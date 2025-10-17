# runtime/trt_vocoder.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import tensorrt as trt
import torch

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
    """Minimal runtime wrapper for TensorRT `.plan` engines."""

    def __init__(self, engine_path: Path, device: torch.device):
        if device.type != "cuda":
            raise ValueError("TensorRT vocoder requires a CUDA device")

        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from '{engine_path}'")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context for vocoder engine")

        self.engine = engine
        self.context = context
        self.profile_index = 0

        if hasattr(self.engine, "num_bindings"):
            num_bindings = self.engine.num_bindings
        elif hasattr(self.engine, "num_io_tensors"):
            num_bindings = self.engine.num_io_tensors
        else:
            raise AttributeError(
                "TensorRT engine does not expose 'num_bindings' or 'num_io_tensors' attributes"
            )

        self._bindings_template: List[int] = [0] * num_bindings
        self._uses_tensor_addresses = hasattr(self.context, "set_tensor_address")
        self._supports_input_api = hasattr(self.context, "set_input_shape")

        if hasattr(self.engine, "binding_is_input"):
            input_indices = [idx for idx in range(num_bindings) if self.engine.binding_is_input(idx)]
            output_indices = [idx for idx in range(num_bindings) if not self.engine.binding_is_input(idx)]

            if len(input_indices) != 1 or len(output_indices) != 1:
                raise ValueError(
                    "The TensorRT vocoder engine must expose exactly one input and one output binding"
                )

            self.input_index = input_indices[0]
            self.output_index = output_indices[0]
            self.input_name = self.engine.get_binding_name(self.input_index)
            self.output_name = self.engine.get_binding_name(self.output_index)

            input_dtype = self.engine.get_binding_dtype(self.input_index)
            output_dtype = self.engine.get_binding_dtype(self.output_index)
        else:
            tensor_names = [self.engine.get_tensor_name(i) for i in range(num_bindings)]
            input_names = [name for name in tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
            output_names = [name for name in tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]

            if len(input_names) != 1 or len(output_names) != 1:
                raise ValueError(
                    "The TensorRT vocoder engine must expose exactly one input and one output tensor"
                )

            self.input_name = input_names[0]
            self.output_name = output_names[0]

            try:
                self.input_index = tensor_names.index(self.input_name)
            except ValueError as exc:
                raise RuntimeError(
                    "Failed to locate TensorRT input tensor in engine bindings"
                ) from exc

            try:
                self.output_index = tensor_names.index(self.output_name)
            except ValueError as exc:
                raise RuntimeError(
                    "Failed to locate TensorRT output tensor in engine bindings"
                ) from exc

            input_dtype = self.engine.get_tensor_dtype(self.input_name)
            output_dtype = self.engine.get_tensor_dtype(self.output_name)
        if input_dtype not in _TRT_TO_TORCH_DTYPE or output_dtype not in _TRT_TO_TORCH_DTYPE:
            raise TypeError("Unsupported TensorRT binding data type for vocoder engine")

        self.input_dtype = _TRT_TO_TORCH_DTYPE[input_dtype]
        self.output_dtype = _TRT_TO_TORCH_DTYPE[output_dtype]

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.to(self.device).to(self.input_dtype).contiguous()
        batch_shape = tuple(mel.shape)

        stream = torch.cuda.current_stream(self.device)

        if hasattr(self.context, "set_optimization_profile_async"):
            self.context.set_optimization_profile_async(self.profile_index, stream.cuda_stream)
        elif hasattr(self.context, "active_optimization_profile"):
            self.context.active_optimization_profile = self.profile_index

        if self._supports_input_api:
            self.context.set_input_shape(self.input_name, batch_shape)
        elif hasattr(self.context, "set_binding_shape"):
            self.context.set_binding_shape(self.input_index, batch_shape)
        elif hasattr(self.context, "set_tensor_shape"):
            self.context.set_tensor_shape(self.input_name, batch_shape)
        else:
            raise AttributeError("TensorRT execution context does not support setting input shapes")

        if hasattr(self.context, "all_binding_shapes_specified") and not self.context.all_binding_shapes_specified:
            raise RuntimeError("Not all TensorRT binding shapes were specified for the vocoder engine")

        if hasattr(self.context, "get_binding_shape"):
            output_shape = tuple(self.context.get_binding_shape(self.output_index))
        elif hasattr(self.context, "get_tensor_shape"):
            output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        else:
            raise AttributeError("TensorRT execution context does not expose output shape retrieval APIs")
        audio = torch.empty(output_shape, device=self.device, dtype=self.output_dtype)

        if self._uses_tensor_addresses:
            self.context.set_tensor_address(self.input_name, mel.data_ptr())
            self.context.set_tensor_address(self.output_name, audio.data_ptr())
            ok = self.context.execute_async_v3(stream.cuda_stream)
        else:
            bindings = list(self._bindings_template)
            bindings[self.input_index] = mel.data_ptr()
            bindings[self.output_index] = audio.data_ptr()

            execute = None
            if hasattr(self.context, "execute_async_v3"):
                execute = self.context.execute_async_v3
            elif hasattr(self.context, "execute_async_v2"):
                execute = self.context.execute_async_v2
            else:
                execute = self.context.execute_v2

            try:
                ok = execute(stream.cuda_stream, bindings)  # type: ignore[arg-type]
            except TypeError:
                try:
                    ok = execute(bindings, stream.cuda_stream)  # type: ignore[arg-type]
                except TypeError:
                    ok = execute(bindings)  # type: ignore[arg-type]

        if not ok:
            raise RuntimeError("TensorRT execution failed for the vocoder engine")

        return audio


class VocoderTRT:
    """Mel [B, D, N] -> wav [B, nw] через TRT-движок."""

    def __init__(self, engine_dir: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = torch.device(device)
        engine_path = Path(engine_dir)

        if engine_path.is_file():
            self.runner = _TensorRTPlanRunner(engine_path, self.device)
            self._plan_runner = True
            self._input_dtype = self.runner.input_dtype
        else:
            self.runner = create_model_runner(engine_path, device=device)
            self._plan_runner = False
            self._input_dtype = dtype

        self.dtype = dtype

    @torch.inference_mode()
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if self._plan_runner:
            return self.runner(mel)

        feeds: Dict[str, torch.Tensor] = {"mel": mel.to(self._input_dtype)}
        out = self.runner(feeds)
        audio = out["audio"]  # имя выхода подставь своё
        return audio

