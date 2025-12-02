import sys
import random
from pathlib import Path
from importlib.resources import files

import tqdm
import torch
import soundfile as sf

from .utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

from .core.utils import seed_everything
from .settings.structure import load_settings

class F5TTS:
    def __init__(
        self,
        config_path: Path,
        vocab_file: Path,
        ckpt_file: Path,
        vocoder_local_path: Path,
        *,
        ode_method="euler",
        use_ema=True,
        device=None
    ) -> None:
        
        model_cfg = load_settings(config_path)

        model_arc = model_cfg.arch
        self.target_sample_rate = model_cfg.mel_spec.target_sample_rate

        self.use_ema = use_ema
        self.ode_method = ode_method

        if device is not None:
            self.device = device

        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        # Load models
        self.vocoder = load_vocoder(
            vocoder_local_path, 
            self.device, 
        )

        self.ema_model = load_model(
            model_arc.model_dump(), 
            ckpt_file, 
            vocab_file, 
            self.ode_method, 
            model_cfg.mel_spec.n_fft,
            model_cfg.mel_spec.hop_length,
            model_cfg.mel_spec.win_length,
            model_cfg.mel_spec.n_mel_channels,
            model_cfg.mel_spec.target_sample_rate,
            use_ema=self.use_ema, 
            device=self.device
        )

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec):
        save_spectrogram(spec, file_spec)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spec=None,
        seed=None,
    ):
        if seed is None:
            seed = random.randint(0, sys.maxsize)

        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)

        wav, sr, spec = infer_process( # type: ignore
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
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


if __name__ == "__main__":
    f5tts = F5TTS(
        config_path=Path("/home/user5/audio/f5/model.yaml"),
        vocab_file=Path("/home/user5/audio/.models/f5/vocab.txt"),
        ckpt_file=Path("/home/user5/audio/.models/f5/espeech_tts_rlv2.pt"),
        vocoder_local_path=Path("/home/user5/audio/.models/f5"),
        device="cuda:6"
    )

    wav, sr, spec = f5tts.infer(
        ref_file="/home/user5/audio/audio_stuff/ref/ref_anast.mp3",
        ref_text="то есть вы представьте себе планка два и семь, где-то какие-то сучки, на маленьком кусочке это просто будет не видно. ну вы же сами понимаете это.",
        gen_text="""Елена, а как вы относитесь к обучению онлайн? У нас стартует курс по развитию карьеры и повышению дохода, в группе с куратором, и вы как раз подходите под условия участия со скидкой.+ ' \n " . ' n ... \n '""",
        nfe_step=16,
        file_wave="../out1.wav",
        seed=None,
    )

    print("seed :", f5tts.seed)
