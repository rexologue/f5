"""Configuration dataclasses and loader for training."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class PathsConfig:
    save_dir: str
    model_config: str
    vocab_path: str
    init_checkpoint: Optional[str] = None
    vocoder_path: Optional[str] = None


@dataclass
class DataConfig:
    train_manifest: str
    val_manifest: Optional[str] = None
    num_workers: int = 4
    batch_type: str = "dynamic"
    max_tokens_per_batch: int = 32000
    max_samples_per_batch: int = 32
    max_audio_length_seconds: Optional[float] = None
    manifest_delimiter: str = "|"


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    weight_decay: float = 0.0
    grad_accumulation_steps: int = 1
    grad_clip_norm: Optional[float] = None
    num_warmup_updates: int = 0
    save_every_steps: int = 1000
    ema_decay: float = 0.0
    use_ema: bool = False


@dataclass
class NeptuneConfig:
    enabled: bool = False
    project: Optional[str] = None
    experiment_name: Optional[str] = None
    api_token: Optional[str] = None
    env_path: Optional[str] = None
    dependencies_path: Optional[str] = None
    run_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SamplingConfig:
    ref_audio_path: str
    ref_text: str
    sample_texts_csv: str
    nfe_step: int = 32
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0
    duration_multiplier: float = 2.0


@dataclass
class EnvConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    distributed: bool = True


@dataclass
class TrainConfig:
    paths: PathsConfig
    data: DataConfig
    training: TrainingConfig
    sampling: Optional[SamplingConfig] = None
    env: EnvConfig = field(default_factory=EnvConfig)
    neptune: Optional[NeptuneConfig] = None


def _load_yaml(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str) -> TrainConfig:
    raw = _load_yaml(config_path)

    paths = PathsConfig(**raw["paths"])
    data = DataConfig(**raw["data"])
    training = TrainingConfig(**raw["training"])
    env_cfg = EnvConfig(**raw.get("env", {}))
    sampling_cfg = raw.get("sampling")
    sampling = SamplingConfig(**sampling_cfg) if sampling_cfg else None
    neptune_cfg = raw.get("neptune")
    neptune = NeptuneConfig(**neptune_cfg) if neptune_cfg else None

    return TrainConfig(
        paths=paths,
        data=data,
        training=training,
        sampling=sampling,
        env=env_cfg,
        neptune=neptune,
    )


def config_as_dict(config: TrainConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "TrainConfig",
    "load_config",
    "config_as_dict",
    "PathsConfig",
    "DataConfig",
    "TrainingConfig",
    "NeptuneConfig",
    "SamplingConfig",
    "EnvConfig",
]
