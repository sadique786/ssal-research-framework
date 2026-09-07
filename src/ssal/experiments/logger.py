from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Persist experiment metrics and history."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history: list[dict[str, Any]] = []

    def log_round(
        self,
        round_index: int,
        metrics: dict[str, float],
        labeled_samples: int,
        unlabeled_samples: int,
    ) -> None:
        """Record metrics for one active learning round."""
        record: dict[str, Any] = {
            "round": round_index,
            "labeled_samples": labeled_samples,
            "unlabeled_samples": unlabeled_samples,
            **metrics,
        }

        self.history.append(record)

    def save_csv(self) -> Path:
        """Save round metrics as CSV."""
        path = self.run_dir / "metrics.csv"

        if not self.history:
            return path

        fieldnames = list(self.history[0].keys())

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(self.history)

        return path

    def save_history(self) -> Path:
        """Save complete experiment history as JSON."""
        path = self.run_dir / "history.json"

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "rounds": self.history,
                },
                file,
                indent=2,
            )
            file.write("\n")

        return path

    def save_summary(
        self,
        summary: dict[str, Any],
    ) -> Path:
        """Save experiment summary as JSON."""
        path = self.run_dir / "summary.json"

        with path.open(
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

        return path

    def save(self) -> None:
        """Persist all currently recorded metrics."""
        self.save_csv()
        self.save_history()
