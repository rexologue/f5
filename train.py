"""Training entrypoint for F5-TTS using YAML configuration."""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src"))

import torch
from ema_pytorch import EMA

from src.f5_tts import F5TTS
from src.f5_tts.utils_infer import load_model
from src.f5_tts.core.utils import seed_everything
from src.f5_tts.utils.logger import create_logger
from src.f5_tts.settings.structure import load_settings
from src.f5_tts.config import config_as_dict, load_config
from src.f5_tts.model import ManifestDataset, Trainer, build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train F5-TTS")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()

def load_initial_checkpoint(path: Optional[str]):
    if not path:
        return None

    ckpt_path = Path(path)
    if ckpt_path.is_dir():
        state_path = ckpt_path / "trainer_state.pth"
        model_path = ckpt_path / "model.pth"
        state = {}
        
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
        if model_path.exists():
            state["model_state_dict"] = torch.load(model_path, map_location="cpu")
            
        return state or None

    loaded = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in loaded or "ema_model_state_dict" in loaded:
        return loaded
    
    return {"model_state_dict": loaded}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    if cfg.data.batch_type != "dynamic":
        raise ValueError("Only dynamic batching is supported. Please set data.batch_type to 'dynamic'.")

    seed_everything(cfg.env.seed)
    
    tts = F5TTS(
        Path(cfg.paths.model_config),
        Path(cfg.paths.vocab_path),
        Path(cfg.paths.init_checkpoint),
        Path(cfg.paths.vocoder_path),
        use_ema=cfg.training.use_ema,
        device=cfg.env.device,
    )

    # Ensure the model weights match the requested precision. The inference loader
    # defaults to half-precision on capable GPUs, which can clash with full-precision
    # training inputs and cause dtype mismatch errors.
    if not cfg.env.fp16:
        tts.model = tts.model.float()

    train_dataset = ManifestDataset(
        cfg.data.train_manifest,
        target_sample_rate=tts.model.mel_spec.target_sample_rate,
        hop_length=tts.model.mel_spec.hop_length,
        n_mel_channels=tts.model.mel_spec.n_mel_channels,
        n_fft=tts.model.mel_spec.n_fft,
        win_length=tts.model.mel_spec.win_length,
        mel_spec_type=tts.model.mel_spec.mel_spec_type,
        max_audio_length_seconds=cfg.data.max_audio_length_seconds,
        manifest_delimiter=cfg.data.manifest_delimiter,
    )

    val_dataset = None
    if cfg.data.val_manifest:
        val_dataset = ManifestDataset(
            cfg.data.val_manifest,
            target_sample_rate=tts.model.mel_spec.target_sample_rate,
            hop_length=tts.model.mel_spec.hop_length,
            n_mel_channels=tts.model.mel_spec.n_mel_channels,
            n_fft=tts.model.mel_spec.n_fft,
            win_length=tts.model.mel_spec.win_length,
            mel_spec_type=tts.model.mel_spec.mel_spec_type,
            max_audio_length_seconds=cfg.data.max_audio_length_seconds,
            manifest_delimiter=cfg.data.manifest_delimiter,
        )

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        cfg.data,
        batch_seed=cfg.env.seed,
    )

    ema = None
    if cfg.training.use_ema:
        ema = EMA(tts.model, beta=cfg.training.ema_decay, include_online_model=False)

    logger = create_logger(cfg.neptune)

    resume_state = None
    if cfg.paths.init_checkpoint is not None:
        resume_state = load_initial_checkpoint(cfg.paths.init_checkpoint)

    trainer = Trainer(
        tts,
        train_loader,
        val_loader,
        cfg.training,
        logger=logger,
        sampling_cfg=cfg.sampling,
        resume_state=resume_state,
        save_dir=cfg.paths.save_dir,
        use_fp16=cfg.env.fp16,
        use_distributed=cfg.env.distributed,
        ema=ema,
        vocoder_path=cfg.paths.vocoder_path,
        full_config=config_as_dict(cfg),
    )

    trainer.train()


if __name__ == "__main__":
    main()
