"""Logger wrappers including Neptune implementation and factory."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional, Union

import neptune
from neptune.utils import stringify_unsupported

from f5_tts.config import NeptuneConfig

try:
    from dotenv import load_dotenv

    USE_DOT_ENV = True
except ImportError:  # pragma: no cover - optional dependency
    USE_DOT_ENV = False


class BaseLogger(ABC):
    """A base experiment logger class."""

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def log_hyperparameters(self, params: dict):
        pass

    @abstractmethod
    def save_metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_plot(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self):
        pass


class NeptuneLogger(BaseLogger):
    """A neptune.ai experiment logger class."""

    def __init__(self, config: NeptuneConfig):
        super().__init__(config)

        env_path = getattr(config, "env_path", None)
        if USE_DOT_ENV and env_path:
            load_dotenv(env_path)

        api_token = getattr(config, "api_token", None) or os.environ.get("NEPTUNE_API_TOKEN")
        if api_token:
            os.environ.setdefault("NEPTUNE_API_TOKEN", api_token)

        run_kwargs = {
            "project": getattr(config, "project", None),
            "api_token": os.environ.get("NEPTUNE_API_TOKEN"),
            "name": getattr(config, "experiment_name", None),
            "dependencies": getattr(config, "dependencies_path", None),
            "with_id": getattr(config, "run_id", None),
            "tags": getattr(config, "tags", None),
        }
        run_kwargs = {k: v for k, v in run_kwargs.items() if v is not None}
        self.run = neptune.init_run(**run_kwargs)

    def log_hyperparameters(self, params: dict):
        self.run["hyperparameters"] = stringify_unsupported(params)

    def save_metrics(
        self,
        type_set: str,
        metric_name: Union[list[str], str],
        metric_value: Union[list[float], float],
        step: Optional[int] = None,
    ):
        if isinstance(metric_name, list):
            for p_n, p_v in zip(metric_name, metric_value):
                self.run[f"{type_set}/{p_n}"].log(p_v, step=step)
        else:
            self.run[f"{type_set}/{metric_name}"].log(metric_value, step=step)

    def save_plot(self, type_set, plot_name, plt_fig):
        self.run[f"{type_set}/{plot_name}"].append(plt_fig)

    def add_tag(self, tag: str):
        self.run["sys/tags"].add(tag)

    def stop(self):
        self.run.stop()


def create_logger(neptune_cfg: Optional[NeptuneConfig]) -> Optional[BaseLogger]:
    if not neptune_cfg or not neptune_cfg.enabled:
        return None
    return NeptuneLogger(neptune_cfg)


__all__ = ["NeptuneLogger", "BaseLogger", "create_logger"]
