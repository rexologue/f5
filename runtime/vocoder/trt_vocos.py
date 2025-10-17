# runtime/vocoder/trt_vocos.py
from __future__ import annotations
import torch
from torch import nn
from typing import Dict, Any
from runtime.setup.utils import TrtEngineRunner

class TRTVocosBackend(nn.Module):
    """
    Совместимый интерфейс с Vocos: .decode(mel:[B, C, T]) -> wav:[B, samples]
    Внутри — TensorRT-план вокодера.
    """
    def __init__(self, engine_path: str, prefer_dtype: torch.dtype = torch.float16, device=None):
        super().__init__()
        self.runner = TrtEngineRunner(engine_path)
        self.prefer_dtype = prefer_dtype
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @torch.no_grad()
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        # ожидаем [B, C, T]
        assert mel.ndim == 3, f"mel must be [B, C, T], got {mel.shape}"
        mel = mel.to(self.device, dtype=self.prefer_dtype).contiguous()

        out = self.runner(
            inputs={"mel": mel},
            output_names=["wave"],
            dynamic_shapes={
                "mel": mel.shape,
                # "wave": будет выведен из план-графа (можно опционально задать ожидаемую форму)
            },
        )
        wave = out["wave"]
        return wave
