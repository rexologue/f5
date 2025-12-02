"""Training entrypoint for F5-TTS using YAML configuration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src"))

import torch

from f5_tts.config import config_as_dict, load_config
from f5_tts.core.dit import DiT
from f5_tts.core.utils import get_tokenizer, seed_everything
from f5_tts.model import CFM, ManifestDataset, Trainer, build_dataloaders
from f5_tts.settings.structure import load_settings
from f5_tts.utils.logger import create_logger
from ema_pytorch import EMA


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


def build_model(cfg: TrainConfig) -> CFM:
    model_settings = load_settings(Path(cfg.paths.model_config))
    vocab_char_map, vocab_size = get_tokenizer(Path(cfg.paths.vocab_path))

    model = CFM(
        transformer=DiT(
            **model_settings.arch.model_dump(),
            text_num_embeds=vocab_size,
            mel_dim=model_settings.mel_spec.n_mel_channels,
        ),
        mel_spec_kwargs=model_settings.mel_spec.model_dump(),
        vocab_char_map=vocab_char_map,
    )
    return model


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if cfg.data.batch_type != "dynamic":
        raise ValueError("Only dynamic batching is supported. Please set data.batch_type to 'dynamic'.")

    seed_everything(cfg.env.seed)

    model = build_model(cfg)

    train_dataset = ManifestDataset(
        cfg.data.train_manifest,
        target_sample_rate=model.mel_spec.target_sample_rate,
        hop_length=model.mel_spec.hop_length,
        n_mel_channels=model.mel_spec.n_mel_channels,
        n_fft=model.mel_spec.n_fft,
        win_length=model.mel_spec.win_length,
        mel_spec_type=model.mel_spec.mel_spec_type,
        max_audio_length_seconds=cfg.data.max_audio_length_seconds,
        manifest_delimiter=cfg.data.manifest_delimiter,
    )

    val_dataset = None
    if cfg.data.val_manifest:
        val_dataset = ManifestDataset(
            cfg.data.val_manifest,
            target_sample_rate=model.mel_spec.target_sample_rate,
            hop_length=model.mel_spec.hop_length,
            n_mel_channels=model.mel_spec.n_mel_channels,
            n_fft=model.mel_spec.n_fft,
            win_length=model.mel_spec.win_length,
            mel_spec_type=model.mel_spec.mel_spec_type,
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
        ema = EMA(model, beta=cfg.training.ema_decay, include_online_model=False)

    logger = create_logger(cfg.neptune)

    resume_state = load_initial_checkpoint(cfg.paths.init_checkpoint)

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg.training,
        logger=logger,
        sampling_cfg=cfg.sampling,
        save_dir=cfg.paths.save_dir,
        resume_state=resume_state,
        use_fp16=cfg.env.fp16,
        use_distributed=cfg.env.distributed,
        ema=ema,
        vocoder_path=cfg.paths.vocoder_path,
        full_config=config_as_dict(cfg),
    )

    trainer.train()


if __name__ == "__main__":
    main()
