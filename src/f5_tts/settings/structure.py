from pathlib import Path

from dynaconf import Dynaconf
from pydantic import BaseModel


class ArchSettings(BaseModel):
    dim: int
    depth: int
    heads: int
    ff_mult: int
    text_dim: int
    text_mask_padding: bool
    qk_norm: str | None
    conv_layers: int
    pe_attn_head: str | None
    attn_backend: str
    attn_mask_enabled: bool
    checkpoint_activations: bool


class MelSettings(BaseModel):
    target_sample_rate: int
    n_mel_channels: int
    hop_length: int
    win_length: int
    n_fft: int


class ModelSettings(BaseModel):
    tokenizer_path: str | None
    arch: ArchSettings
    mel_spec: MelSettings


def load_settings(settings_path: Path) -> ModelSettings:
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    dynaconf_settings = Dynaconf(settings_files=[settings_path])
    config_data = {k.lower(): v for k, v in dynaconf_settings.to_dict(internal=False).items()}
    return ModelSettings.model_validate(config_data)
