from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


class ExperimentRun:
    """Manage filesystem artifacts for a single research experiment run."""

    def __init__(
        self,
        cfg: DictConfig,
        root_dir: str | Path = "experiments",
    ) -> None:
        self.cfg = cfg
        self.root_dir = Path(root_dir)

        dataset_name = str(cfg.dataset.name)
        experiment_name = str(cfg.experiment.name)
        active_learning_name = str(cfg.active_learning.name)
        seed = int(cfg.seed)

        self.run_dir = (
            self.root_dir
            / dataset_name
            / experiment_name
            / active_learning_name
            / f"seed_{seed}"
        )

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"

    def create(self) -> None:
        """Create the directory structure for the experiment run."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self) -> Path:
        """Save the resolved Hydra configuration."""
        self.create()

        config_path = self.run_dir / "config.yaml"
        OmegaConf.save(
            config=self.cfg,
            f=config_path,
            resolve=True,
        )

        return config_path

    def save_metadata(
        self,
        device: str,
    ) -> Path:
        """Save metadata describing the experiment environment."""
        self.create()

        metadata: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(self.cfg.dataset.name),
            "model": str(self.cfg.model.name),
            "experiment": str(self.cfg.experiment.name),
            "active_learning": str(
                self.cfg.active_learning.name
            ),
            "seed": int(self.cfg.seed),
            "device": device,
        }

        metadata_path = self.run_dir / "metadata.json"

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )
            file.write("\n")

        return metadata_path

    def save_summary(
        self,
        summary: dict[str, Any],
    ) -> Path:
        """Save a JSON summary of the experiment."""
        self.create()

        summary_path = self.run_dir / "summary.json"

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                default=str,
            )
            file.write("\n")

        return summary_path
