from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset, Subset

from ssal.active_learning.base import BaseQueryStrategy
from ssal.active_learning.pool import ActiveLearningPool
from ssal.training import SupervisedTrainer


class ActiveLearningExperiment:
    """Coordinate training, evaluation, and querying for active learning."""

    def __init__(
        self,
        dataset: Dataset[Any],
        pool: ActiveLearningPool,
        strategy: BaseQueryStrategy,
    ) -> None:
        if len(dataset) != pool.dataset_size:
            raise ValueError(
                "Dataset size must match the active learning pool size."
            )

        self.dataset = dataset
        self.pool = pool
        self.strategy = strategy
        self.history: list[dict[str, float]] = []

    def create_labeled_dataloader(
        self,
        batch_size: int,
    ) -> DataLoader[Any]:
        """Create a data loader containing only labeled samples."""

        labeled_dataset = Subset(
            self.dataset,
            self.pool.labeled_indices,
        )

        return DataLoader(
            labeled_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    def run_round(
        self,
        trainer: SupervisedTrainer,
        test_loader: DataLoader[Any],
        epochs: int,
        batch_size: int,
    ) -> dict[str, float]:
        """Train and evaluate for one active learning round."""

        if epochs <= 0:
            raise ValueError("Epochs must be greater than zero.")

        train_loader = self.create_labeled_dataloader(
            batch_size=batch_size,
        )

        train_metrics: dict[str, float] = {}

        for _ in range(epochs):
            train_metrics = trainer.train_epoch(
                train_loader,
            )

        test_metrics = trainer.evaluate(
            test_loader,
        )

        metrics = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }

        self.history.append(metrics)

        return metrics

    def query(
        self,
        query_size: int,
    ) -> list[int]:
        """Select samples from the unlabeled pool."""

        return self.strategy.query(
            self.pool,
            query_size=query_size,
        )

    def update_pool(
        self,
        selected_indices: list[int],
    ) -> None:
        """Move selected samples into the labeled pool."""

        self.pool.update(
            selected_indices,
        )
