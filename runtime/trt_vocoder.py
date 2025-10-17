# runtime/trt_vocoder.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)
from __future__ import annotations
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import tensorrt as trt

from trt_utils import create_model_runner


_TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


class _TensorRTPlanRunner:
    """Низкоуровневая обёртка над .plan движком (без ModelRunner)."""
    
    def __init__(self, engine_path: Path, device: torch.device):
        if device.type != "cuda":
            raise ValueError("TRT vocoder требует CUDA")
        
        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        
        if engine is None:
            raise RuntimeError(f"Не удалось десериализовать TRT engine: {engine_path}")
        
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Не удалось создать execution context")
        
        self.engine = engine
        self.context = context
        
        # Определяем API версию
        self._uses_tensor_addresses = hasattr(self.context, "set_tensor_address")
        self._supports_input_api = hasattr(self.context, "set_input_shape")
        
        # Находим входы/выходы
        num_bindings = getattr(engine, "num_bindings", getattr(engine, "num_io_tensors", 0))
        
        if hasattr(engine, "binding_is_input"):
            # Legacy API
            input_indices = [i for i in range(num_bindings) if engine.binding_is_input(i)]
            output_indices = [i for i in range(num_bindings) if not engine.binding_is_input(i)]
            
            if len(input_indices) != 1 or len(output_indices) != 1:
                raise ValueError("Vocoder engine должен иметь ровно 1 вход и 1 выход")
            
            self.input_index = input_indices[0]
            self.output_index = output_indices[0]
            self.input_name = engine.get_binding_name(self.input_index)
            self.output_name = engine.get_binding_name(self.output_index)
            
            self.input_dtype = _TRT_TO_TORCH_DTYPE[engine.get_binding_dtype(self.input_index)]
            self.output_dtype = _TRT_TO_TORCH_DTYPE[engine.get_binding_dtype(self.output_index)]
        
        else:
            # Modern API
            tensor_names = [engine.get_tensor_name(i) for i in range(num_bindings)]
            input_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            output_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            
            if len(input_names) != 1 or len(output_names) != 1:
                raise ValueError("Vocoder engine должен иметь ровно 1 вход и 1 выход")
            
            self.input_name = input_names[0]
            self.output_name = output_names[0]
            
            try:
                self.input_index = tensor_names.index(self.input_name)
                self.output_index = tensor_names.index(self.output_name)
            except ValueError as e:
                raise RuntimeError("Не удалось найти индексы входов/выходов") from e
            
            self.input_dtype = _TRT_TO_TORCH_DTYPE[engine.get_tensor_dtype(self.input_name)]
            self.output_dtype = _TRT_TO_TORCH_DTYPE[engine.get_tensor_dtype(self.output_name)]
        
        self._bindings_template = [0] * num_bindings
    
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: [B, n_mels, T] → audio: [B, n_samples]
        """
        mel = mel.to(self.device).to(self.input_dtype).contiguous()
        batch_shape = tuple(mel.shape)
        
        stream = torch.cuda.current_stream(self.device)
        
        # Установка оптимизационного профиля
        if hasattr(self.context, "set_optimization_profile_async"):
            self.context.set_optimization_profile_async(0, stream.cuda_stream)
        
        # Установка формы входа
        if self._supports_input_api:
            self.context.set_input_shape(self.input_name, batch_shape)
        elif hasattr(self.context, "set_binding_shape"):
            self.context.set_binding_shape(self.input_index, batch_shape)
        elif hasattr(self.context, "set_tensor_shape"):
            self.context.set_tensor_shape(self.input_name, batch_shape)
        else:
            raise AttributeError("TRT context не поддерживает установку форм")
        
        # Получаем форму выхода
        if hasattr(self.context, "get_binding_shape"):
            output_shape = tuple(self.context.get_binding_shape(self.output_index))
        elif hasattr(self.context, "get_tensor_shape"):
            output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        else:
            raise AttributeError("TRT context не поддерживает получение форм выхода")
        
        # Аллокация выхода
        audio = torch.empty(output_shape, device=self.device, dtype=self.output_dtype)
        
        # Выполнение
        if self._uses_tensor_addresses:
            self.context.set_tensor_address(self.input_name, mel.data_ptr())
            self.context.set_tensor_address(self.output_name, audio.data_ptr())
            ok = self.context.execute_async_v3(stream.cuda_stream)
        else:
            bindings = list(self._bindings_template)
            bindings[self.input_index] = mel.data_ptr()
            bindings[self.output_index] = audio.data_ptr()
            
            if hasattr(self.context, "execute_async_v3"):
                ok = self.context.execute_async_v3(stream.cuda_stream)
            elif hasattr(self.context, "execute_async_v2"):
                ok = self.context.execute_async_v2(bindings, stream.cuda_stream)
            else:
                ok = self.context.execute_v2(bindings)
        
        if not ok:
            raise RuntimeError("TRT vocoder execution failed")
        
        return audio


class VocoderTRT(nn.Module):
    """
    Универсальный TRT Vocoder:
      - mel [B, n_mels, T] → audio [B, n_samples]
      - Поддерживает как .plan файлы, так и ModelRunner
    """
    
    def __init__(
        self,
        engine_dir: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.dtype = dtype
        
        engine_path = Path(engine_dir)
        
        # Определяем тип движка
        if engine_path.is_file() and engine_path.suffix == ".plan":
            # Голый .plan файл → используем низкоуровневый runner
            self.runner = _TensorRTPlanRunner(engine_path, self.device)
            self._is_plan_runner = True
            self._input_dtype = self.runner.input_dtype
        
        elif engine_path.is_dir():
            # Директория с TRT-LLM engine → используем ModelRunner
            self.runner = create_model_runner(str(engine_path), device=device)
            self._is_plan_runner = False
            self._input_dtype = dtype
        
        else:
            raise ValueError(
                f"engine_dir должен быть либо .plan файлом, либо директорией с TRT-LLM engine: {engine_dir}"
            )
    
    @torch.inference_mode()
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: [B, n_mels, T] → audio: [B, n_samples]
        """
        if self._is_plan_runner:
            return self.runner(mel)
        
        # ModelRunner API
        feeds = {"mel": mel.to(self._input_dtype)}
        out = self.runner(feeds)
        
        # Попытка извлечь аудио из разных возможных ключей
        for key in ["audio", "waveform", "output", "wav"]:
            if key in out:
                return out[key]
        
        # Fallback: берём первый выход
        if len(out) == 1:
            return next(iter(out.values()))
        
        raise KeyError(f"Не удалось найти аудио в выходе TRT vocoder. Доступные ключи: {list(out.keys())}")