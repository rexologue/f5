from __future__ import annotations

import os
import sys
import ctypes
import sysconfig
from pathlib import Path

import torch

try:
    import tensorrt as trt
except Exception as e:
    trt = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa

except Exception:
    cuda = None

def preload_libpython():
    major, minor = sys.version_info[:2]

    # собрать кандидатные директории вокруг текущего интерпретатора
    libdirs = []
    for key in ("LIBDIR", "LIBPL"):
        d = sysconfig.get_config_var(key)
        if d: libdirs.append(Path(d))
    exe_prefix = Path(sys.executable).resolve().parents[1]  # .../env/bin/python -> .../env
    libdirs += [exe_prefix / "lib", exe_prefix / "lib64", Path(sys.prefix) / "lib", Path(sys.prefix) / "lib64"]
    libdirs = [p for p in dict.fromkeys(libdirs) if p.exists()]  # уникализуем и фильтруем существующие

    # имена по приоритету (только shared)
    names = [
        f"libpython{major}.{minor}.so.1.0",
        f"libpython{major}.{minor}.so",
        "libpython3.so",
        "libpython.so",
    ]

    candidates = []

    # если LDLIBRARY указывает на .so и он существует — используем первым
    ldlib = sysconfig.get_config_var("LDLIBRARY")
    if ldlib and ldlib.endswith(".so"):
        for d in libdirs:
            p = d / ldlib
            if p.exists():
                candidates.append(p)
                break

    # пройтись по именам с приоритетом
    for d in libdirs:
        for n in names:
            p = d / n
            if p.exists():
                candidates.append(p)

    # последняя попытка: glob по шаблону .so*
    if not candidates:
        for d in libdirs:
            for p in sorted(d.glob(f"libpython{major}.{minor}*.so*")):
                candidates.append(p)

    if not candidates:
        raise RuntimeError(f"libpython .so not found in: {', '.join(map(str, libdirs))}")

    # взять первый валидный shared объект
    libpath = str(candidates[0])
    ctypes.CDLL(libpath, mode=ctypes.RTLD_GLOBAL)


class TrtEngineRunner:
    """
    Универсальный раннер для .plan-движков TensorRT.
    - поддерживает динамические размеры (set_binding_shape)
    - имена входов/выходов — по биндингам engine
    - принимает/возвращает torch.Tensor на текущем CUDA-устройстве

    ВАЖНО: Нужен pycuda для аллокаций. Если его нет — кинем понятную ошибку.
    """
    def __init__(self, engine_path: str, device_id: int = 0):
        if trt is None or cuda is None:
            raise RuntimeError("TensorRT / pycuda не найдены. Установите tensorrt и pycuda.")
        assert os.path.exists(engine_path), f"Engine file not found: {engine_path}"

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine is not None, "Failed to deserialize engine"

        self.context = self.engine.create_execution_context()
        self.device_id = device_id

        # карта имён биндингов
        self.bindings_idx = {self.engine.get_binding_name(i): i for i in range(self.engine.num_bindings)}
        self.is_input = {name: self.engine.binding_is_input(i) for name, i in self.bindings_idx.items()}

        self.stream = cuda.Stream()

        # кэш под аллоцированные буферы (на случай многих вызовов с одинаковыми размерами)
        self._dev_ptrs = {}   # name -> device allocation
        self._host_bufs = {}  # name -> pinned host buffer (np)

    def _alloc_binding(self, name: str, shape, dtype: trt.DataType, for_input: bool):
        i = self.bindings_idx[name]
        # выставляем динамическую форму
        self.context.set_binding_shape(i, shape)
        nbytes = int(trt.volume(shape) * trt.nptype(dtype).itemsize)

        # входам не делаем host буфер, пишем сразу из torch
        if for_input:
            if name in self._dev_ptrs:
                return
            self._dev_ptrs[name] = cuda.mem_alloc(nbytes)
            return

        # выходы — заведём host+dev
        if name not in self._host_bufs or self._host_bufs[name].nbytes != nbytes:
            self._host_bufs[name] = cuda.pagelocked_empty(shape=trt.volume(shape), dtype=trt.nptype(dtype))
        if name not in self._dev_ptrs or self._dev_ptrs[name].size < nbytes:
            self._dev_ptrs[name] = cuda.mem_alloc(nbytes)

    def __call__(self, inputs: dict, output_names: list[str], dynamic_shapes: dict | None = None) -> dict:
        # выставляем формы
        if dynamic_shapes is not None:
            for name, shape in dynamic_shapes.items():
                if name in self.bindings_idx:
                    dtype = self.engine.get_binding_dtype(self.bindings_idx[name])
                    self._alloc_binding(name, tuple(shape), dtype, self.is_input[name])

        # собираем вектор bindings
        bindings = [None] * self.engine.num_bindings

        # заливаем входы
        for name, tensor in inputs.items():
            assert name in self.bindings_idx, f"Unknown binding: {name}"
            i = self.bindings_idx[name]
            assert self.is_input[name]
            # аллокация (если нужно)
            dtype = self.engine.get_binding_dtype(i)
            self._alloc_binding(name, tuple(tensor.shape), dtype, True)

            # копирование: torch.Tensor (cuda) -> device ptr
            assert tensor.is_cuda, "Input tensor must be on CUDA"
            # гарантировать плотный layout
            t = tensor.contiguous()
            cuda.memcpy_dtod_async(int(self._dev_ptrs[name]), int(t.data_ptr()), t.numel() * t.element_size(), self.stream)
            bindings[i] = int(self._dev_ptrs[name])

        # выходы
        for name in output_names:
            assert name in self.bindings_idx, f"Unknown output binding: {name}"
            i = self.bindings_idx[name]
            assert not self.is_input[name]
            dtype = self.engine.get_binding_dtype(i)
            shape = self.context.get_binding_shape(i)
            # если форму не задали заранее, всё равно аллоцируем по текущей (может быть -1 → уже конкретизирована)
            self._alloc_binding(name, tuple(shape), dtype, False)
            bindings[i] = int(self._dev_ptrs[name])

        # запуск
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()

        # копируем выходы на host и собираем torch cuda-тензоры
        outputs = {}
        for name in output_names:
            i = self.bindings_idx[name]
            dtype = self.engine.get_binding_dtype(i)
            shape = tuple(self.context.get_binding_shape(i))
            host_buf = self._host_bufs[name]
            cuda.memcpy_dtoh_async(host_buf, self._dev_ptrs[name], self.stream)
            self.stream.synchronize()
            np_out = host_buf.reshape(shape)
            # вернём CUDA тензор (можно возвращать CPU при желании)
            torch_out = torch.from_numpy(np_out).to(device="cuda")
            outputs[name] = torch_out
        return outputs