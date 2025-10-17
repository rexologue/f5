# runtime/f5tts_trt.py
from __future__ import annotations
import sys
import random
from pathlib import Path

import torch
import soundfile as sf
import tqdm

from src.utils_infer import (
    infer_process,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from src.core.utils import seed_everything
from src.settings.structure import load_settings

from .trt_transformer import F5DiTTRT
from .trt_vocoder import VocoderTRT
from src.core.cfm import CFM  # твой CFM из src/core/cfm.py (если путь другой — поправь импорт)


def _build_vocab_map(vocab_file: Path) -> dict[str, int]:
    # простая загрузка: по строкам -> id
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            token = line.rstrip("\n")
            if token and token not in vocab:
                vocab[token] = i
    return vocab


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
        if device is not None:
            self.device = device
        else:
            self.device = (
                "cuda" if torch.cuda.is_available() else
                "xpu" if torch.xpu.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )

        # --- TRT Vocoder ---
        self.vocoder = VocoderTRT(str(trt_vocoder_dir), device=self.device, dtype=torch.float16)

        # --- TRT DiT transformer (эмбеддинги/финал — на Torch) ---
        hidden = model_cfg.arch.hidden_size
        num_heads = model_cfg.arch.num_attention_heads
        mel_dim = model_cfg.mel_spec.n_mel_channels

        transformer_trt = F5DiTTRT(
            ckpt_path=str(ckpt_file),
            trt_dit_dir=str(trt_dit_dir),
            mel_dim=mel_dim,
            hidden=hidden,
            num_heads=num_heads,
            dtype=torch.float16,
            device=self.device,
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
            vocab_char_map=_build_vocab_map(vocab_file),
        )

        self.use_ema = use_ema
        self.ode_method = ode_method

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
