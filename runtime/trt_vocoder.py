# runtime/trt_vocoder.py
from __future__ import annotations
from typing import Dict
import torch
from tensorrt_llm.runtime import ModelRunner

class VocoderTRT:
    """Mel [B, D, N] -> wav [B, nw] через TRT-движок."""
    def __init__(self, engine_dir: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.runner = ModelRunner(engine_dir, device=device)
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.inference_mode()
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        feeds: Dict[str, torch.Tensor] = {"mel": mel.to(self.dtype)}
        out = self.runner(feeds)
        return out["audio"]  # имя выхода подставь своё
