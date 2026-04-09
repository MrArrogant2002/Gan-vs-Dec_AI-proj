from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from training.utils import ensure_dir, get_env_or_default, to_serializable


LOGGER = logging.getLogger(__name__)


class ExperimentLogger:
    def __init__(
        self,
        config: dict,
        run_name: str,
        job_type: str,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        tracking_config = config.get("tracking", {})
        self.enabled = bool(tracking_config.get("use_wandb", False))
        self.log_tables = bool(tracking_config.get("log_tables", True))
        self.log_images = bool(tracking_config.get("log_images", True))
        self.run = None

        if not self.enabled:
            return

        try:
            import wandb
        except Exception as exc:  # pragma: no cover - depends on target env
            LOGGER.warning("W&B import failed, disabling experiment logging: %s", exc)
            self.enabled = False
            return

        try:
            wandb_dir = ensure_dir(tracking_config.get("wandb_dir", "experiments/wandb"))
            self.run = wandb.init(
                project=get_env_or_default("WANDB_PROJECT", tracking_config.get("wandb_project")),
                entity=get_env_or_default("WANDB_ENTITY", tracking_config.get("wandb_entity") or None),
                mode=get_env_or_default("WANDB_MODE", tracking_config.get("wandb_mode", "offline")),
                dir=str(wandb_dir),
                name=run_name,
                job_type=job_type,
                tags=list(tags or []),
                config=to_serializable(config),
                reinit=True,
            )
        except Exception as exc:  # pragma: no cover - depends on target env
            LOGGER.warning("W&B init failed, continuing without it: %s", exc)
            self.enabled = False
            self.run = None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: Optional[str] = None) -> None:
        if not self.run:
            return
        payload = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                payload[f"{prefix}/{key}" if prefix else key] = value
        if payload:
            self.run.log(payload, step=step)

    def log_dataframe(self, name: str, frame: pd.DataFrame, max_rows: int = 250) -> None:
        if not self.run or not self.log_tables:
            return
        try:
            import wandb

            limited = frame.head(max_rows).copy()
            self.run.log({name: wandb.Table(dataframe=limited)})
        except Exception as exc:  # pragma: no cover - depends on target env
            LOGGER.warning("Failed to log table %s to W&B: %s", name, exc)

    def log_images_from_paths(self, images: Dict[str, Path]) -> None:
        if not self.run or not self.log_images:
            return
        try:
            import wandb

            payload = {name: wandb.Image(str(path)) for name, path in images.items() if Path(path).exists()}
            if payload:
                self.run.log(payload)
        except Exception as exc:  # pragma: no cover - depends on target env
            LOGGER.warning("Failed to log images to W&B: %s", exc)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
