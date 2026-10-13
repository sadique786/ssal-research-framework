from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from ssal.active_learning.base import BaseQueryStrategy
from ssal.active_learning.pool import ActiveLearningPool


class LeastConfidenceStrategy(BaseQueryStrategy):
    """Select samples with the lowest maximum predicted class probability."""

    def query(
        self,
        pool: ActiveLearningPool,
        query_size: int,
        *,
        dataset: Dataset[Any] | None = None,
        model: nn.Module | None = None,
        device: torch.device | None = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        if query_size <= 0:
            raise ValueError("query_size must be greater than zero.")
        if query_size > pool.num_unlabeled:
            raise ValueError(
                "query_size cannot exceed the number of unlabeled samples."
            )
        if dataset is None or model is None or device is None:
            raise ValueError(
                "Least confidence requires dataset, model, and device."
            )
        if len(dataset) != pool.dataset_size:
            raise ValueError("Dataset size must match the active learning pool.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        unlabeled_indices = pool.unlabeled_indices
        loader = DataLoader(
            Subset(dataset, unlabeled_indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
        )

        was_training = model.training
        model.eval()
        confidence_batches: list[torch.Tensor] = []

        try:
            with torch.no_grad():
                for batch in loader:
                    images = batch[0].to(device)
                    logits = model(images)
                    probabilities = torch.softmax(logits, dim=1)
                    confidence_batches.append(probabilities.max(dim=1).values.cpu())
        finally:
            model.train(was_training)

        confidence = torch.cat(confidence_batches).numpy()
        chosen_positions = np.argsort(confidence, kind="stable")[:query_size]
        return np.sort(unlabeled_indices[chosen_positions])
