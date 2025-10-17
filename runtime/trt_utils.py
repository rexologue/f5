"""Utility helpers for working with TensorRT-LLM runtimes."""
from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path
from typing import Union

from tensorrt_llm.runtime import ModelRunner


def _patch_missing_vocab_size(
    config_path: Path,
    *,
    fallback_vocab_size: int | None = None,
) -> bool:
    """Ensure ``builder_config.vocab_size`` exists in TensorRT engine configs.

    TensorRT-LLM 1.0 tightened validation around the vocabulary size stored in
    the generated ``config.json``.  Engines converted before this requirement
    may omit the field altogether, which causes ``ModelRunner.from_dir`` to
    crash when it attempts to pad the vocabulary size.  The exact value is not
    used by our TTS models, so we fall back to the closest reasonable value we
    can infer from the remaining metadata.

    Parameters
    ----------
    config_path:
        Path to the ``config.json`` file inside the engine directory.

    Returns
    -------
    bool
        ``True`` when the configuration file was updated, ``False`` otherwise.
    """

    if not config_path.exists():
        return False

    try:
        data = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    builder_cfg = data.setdefault("builder_config", {})
    vocab_size = builder_cfg.get("vocab_size")
    if vocab_size is not None:
        return False

    fallback = (
        data.get("pretrained_config", {}).get("vocab_size")
        or data.get("tokenizer", {}).get("vocab_size")
        or data.get("tokenizer", {}).get("model", {}).get("vocab_size")
        or fallback_vocab_size
    )

    if fallback is None:
        fallback = 0

    builder_cfg["vocab_size"] = int(fallback)

    try:
        config_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    except OSError:
        return False

    warnings.warn(
        "TensorRT engine configuration lacked 'builder_config.vocab_size'. "
        "Inserted fallback value %d to keep ModelRunner initialisation "
        "compatible with TensorRT-LLM 1.0." % builder_cfg["vocab_size"],
        RuntimeWarning,
    )
    return True


def create_model_runner(
    engine_path: Union[str, Path],
    *,
    device: Union[str, int] | None = None,
    fallback_vocab_size: int | None = None,
) -> ModelRunner:
    """Create a :class:`ModelRunner` instance handling API differences.

    TensorRT-LLM changed the :class:`ModelRunner` constructor in v1.0 to
    require a ``GenerationSession`` object.  Older releases expected an engine
    directory (or file) directly.  This helper inspects the available API and
    instantiates the runner accordingly so that the rest of the codebase can
    remain version-agnostic.

    Parameters
    ----------
    engine_path:
        Path to the TensorRT-LLM engine directory (or file for older
        versions).
    device:
        Optional device identifier used only by legacy versions that accept a
        ``device`` keyword argument.
    fallback_vocab_size:
        Vocabulary size inferred from the tokenizer.  Used only when the TensorRT
        engine config is missing the ``builder_config.vocab_size`` field.
    """

    engine_path = Path(engine_path)
    init_sig = inspect.signature(ModelRunner.__init__)
    param_names = list(init_sig.parameters.keys())

    # TensorRT-LLM >= 1.0 expects ``session`` as the first argument after
    # ``self`` and provides the ``from_dir`` classmethod for convenience.
    if len(param_names) >= 2 and param_names[1] == "session" and hasattr(ModelRunner, "from_dir"):
        engine_dir = engine_path if engine_path.is_dir() else engine_path.parent
        if not engine_dir.exists():
            raise FileNotFoundError(f"TensorRT engine directory '{engine_dir}' does not exist")

        try:
            return ModelRunner.from_dir(str(engine_dir))
        except TypeError as exc:
            # TensorRT-LLM raises a cryptic ``TypeError`` when ``vocab_size`` is
            # missing.  Patch the config and retry once before surfacing the
            # original failure.
            if "NoneType" not in str(exc):
                raise

            config_path = engine_dir / "config.json"
            if not _patch_missing_vocab_size(config_path, fallback_vocab_size=fallback_vocab_size):
                raise
            return ModelRunner.from_dir(str(engine_dir))

    runner_kwargs = {}
    if device is not None and "device" in init_sig.parameters:
        runner_kwargs["device"] = device

    return ModelRunner(str(engine_path), **runner_kwargs)

