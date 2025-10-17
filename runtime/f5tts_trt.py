# runtime/f5tts_trt.py
from __future__ import annotations

import os
import sys
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from pathlib import Path

import torch
import soundfile as sf
import tqdm

from setup.utils import preload_libpython
preload_libpython()

from src.utils_infer import (
    infer_process,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from src.core.utils import seed_everything
from src.settings.structure import load_settings

from trt_transformer import F5DiTTRT
from trt_vocoder import VocoderTRT
from src.core.cfm import CFM  # твой CFM из src/core/cfm.py (если путь другой — поправь импорт)
from src.core.utils import get_tokenizer


class F5TTSTRT:
    """
    Аналог src/f5tts.py, но:
      - трансформер = TRT-DiT + эмбеддинги на Torch
      - вокодер = TRT
      - для инференса используем исходный utils_infer.infer_process(...)
    """
    def __init__(
        self,
        config_path: Path,
        vocab_file: Path,
        ckpt_file: Path,
        trt_dit_dir: Path,        # <- папка с TRT DiT (из convert_checkpoint.py)
        trt_vocoder_dir: Path,    # <- папка с TRT вокодером
        *,
        ode_method: str = "euler",
        use_ema: bool = True,     # флаг сохранён для совместимости
        device: str | None = None,
    ) -> None:

        model_cfg = load_settings(config_path)
        self.target_sample_rate = model_cfg.mel_spec.target_sample_rate

        # устройство
        self.device = self._resolve_device(device)

        # --- TRT Vocoder ---
        self.vocoder = VocoderTRT(str(trt_vocoder_dir), device=self.device, dtype=torch.float16)

        # --- TRT DiT transformer (эмбеддинги/финал — на Torch) ---
        hidden = model_cfg.arch.dim
        num_heads = model_cfg.arch.heads
        mel_dim = model_cfg.mel_spec.n_mel_channels

        vocab_char_map, vocab_size = get_tokenizer(vocab_file)

        transformer_trt = F5DiTTRT(
            ckpt_path=str(ckpt_file),
            trt_dit_dir=str(trt_dit_dir),
            mel_dim=mel_dim,
            hidden=hidden,
            num_heads=num_heads,
            dtype=torch.float16,
            device=self.device,
            tokenizer_vocab_size=vocab_size,
        )

        # --- CFM, куда подсовываем наш transformer ---
        self.ema_model = CFM(
            transformer=transformer_trt,
            odeint_kwargs=dict(method=ode_method),
            num_channels=mel_dim,
            mel_spec_kwargs=dict(
                sr=model_cfg.mel_spec.target_sample_rate,
                n_mels=model_cfg.mel_spec.n_mel_channels,
                hop_length=model_cfg.mel_spec.hop_length,
                n_fft=model_cfg.mel_spec.n_fft,
                win_length=model_cfg.mel_spec.win_length,
            ),
            vocab_char_map=vocab_char_map,
        )

        self.use_ema = use_ema
        self.ode_method = ode_method

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        """Validate and normalise device strings before they reach PyTorch/TRT."""
        # Auto-detect when nothing is specified (kept consistent with the original
        # behaviour).
        if device is None:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
                return "xpu"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        normalized = device.strip().lower()

        def _warn_and_fallback(msg: str) -> str:
            warnings.warn(msg, RuntimeWarning)
            if torch.cuda.is_available():
                # Default CUDA device keeps compatibility with TensorRT runtime.
                return "cuda"
            if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
                return "xpu"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if normalized.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device but CUDA is not available in this environment.")

            if ":" in normalized:
                _, index_str = normalized.split(":", 1)
                try:
                    index = int(index_str)
                except ValueError as exc:  # pragma: no cover - defensive branch
                    raise ValueError(f"Invalid CUDA device specification: '{device}'.") from exc

                cuda_count = torch.cuda.device_count()
                if index < 0 or index >= cuda_count:
                    return _warn_and_fallback(
                        f"CUDA device index {index} is out of range for {cuda_count} available device(s); "
                        "falling back to the default GPU."
                    )
                return f"cuda:{index}"
            return "cuda"

        if normalized.startswith("xpu"):
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):  # type: ignore[attr-defined]
                raise RuntimeError("Requested XPU device but XPU is not available in this environment.")
            return "xpu"

        if normalized.startswith("mps"):
            if not torch.backends.mps.is_available():
                raise RuntimeError("Requested MPS device but MPS is not available in this environment.")
            return "mps"

        if normalized == "cpu":
            return "cpu"

        raise ValueError(f"Unsupported device specification: '{device}'.")

    # --- экспорт аналогично src/f5tts.py ---
    def export_wav(self, wav, file_wave: Path, remove_silence: bool = False):
        sf.write(file_wave, wav, self.target_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec: Path):
        save_spectrogram(spec, file_spec)

    # --- инференс, как в src/f5tts.py ---
    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm.tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2.0,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave: Path | None = None,
        file_spec: Path | None = None,
        seed: int | None = None,
    ):
        if seed is None:
            seed = random.randint(0, sys.maxsize)

        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)

        wav, sr, spec = infer_process(  # type: ignore
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,   # <--- наш CFM с TRT-трансформером
            self.vocoder,     # <--- TRT вокодер (callable: mel[B,D,N] -> wav[B,Nw])
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spec is not None:
            self.export_spectrogram(spec, file_spec)

        return wav, sr, spec
