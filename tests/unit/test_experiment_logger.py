from __future__ import annotations

import csv
import json

from ssal.experiments import ExperimentLogger


def test_logger_records_round_metrics(tmp_path) -> None:
    """Test that round metrics are recorded."""
    logger = ExperimentLogger(tmp_path)

    logger.log_round(
        round_index=1,
        metrics={
            "train_loss": 0.25,
            "train_accuracy": 0.91,
            "test_loss": 1.07,
            "test_accuracy": 0.73,
        },
        labeled_samples=5000,
        unlabeled_samples=45000,
    )

    assert len(logger.history) == 1
    assert logger.history[0]["round"] == 1
    assert logger.history[0]["labeled_samples"] == 5000
    assert logger.history[0]["test_accuracy"] == 0.73


def test_logger_saves_csv(tmp_path) -> None:
    """Test that metrics are persisted as CSV."""
    logger = ExperimentLogger(tmp_path)

    logger.log_round(
        round_index=1,
        metrics={
            "train_loss": 0.25,
            "train_accuracy": 0.91,
            "test_loss": 1.07,
            "test_accuracy": 0.73,
        },
        labeled_samples=5000,
        unlabeled_samples=45000,
    )

    path = logger.save_csv()

    assert path.exists()

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["round"] == "1"
    assert rows[0]["labeled_samples"] == "5000"


def test_logger_saves_history(tmp_path) -> None:
    """Test that complete history is persisted as JSON."""
    logger = ExperimentLogger(tmp_path)

    logger.log_round(
        round_index=1,
        metrics={
            "train_loss": 0.25,
            "train_accuracy": 0.91,
            "test_loss": 1.07,
            "test_accuracy": 0.73,
        },
        labeled_samples=5000,
        unlabeled_samples=45000,
    )

    path = logger.save_history()

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        history = json.load(file)

    assert len(history["rounds"]) == 1
    assert history["rounds"][0]["round"] == 1
    assert history["rounds"][0]["test_accuracy"] == 0.73
