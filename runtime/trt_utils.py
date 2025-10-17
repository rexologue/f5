"""Utility helpers for working with TensorRT-LLM runtimes."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Union

from tensorrt_llm.runtime import ModelRunner


def create_model_runner(engine_path: Union[str, Path], *, device: Union[str, int] | None = None) -> ModelRunner:
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
        return ModelRunner.from_dir(str(engine_dir))

    runner_kwargs = {}
    if device is not None and "device" in init_sig.parameters:
        runner_kwargs["device"] = device

    return ModelRunner(str(engine_path), **runner_kwargs)

