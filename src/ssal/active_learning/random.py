from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Any

from ssal.active_learning.base import BaseQueryStrategy
from ssal.active_learning.pool import ActiveLearningPool


class RandomSamplingStrategy(BaseQueryStrategy):
    """Randomly select samples from the unlabeled pool."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

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
        """Randomly select indices from the unlabeled pool."""

        if query_size <= 0:
            raise ValueError("query_size must be greater than zero.")
        if query_size > pool.num_unlabeled:
            raise ValueError(
                "query_size cannot exceed the number of unlabeled samples."
            )

        return np.sort(
            self._rng.choice(
                pool.unlabeled_indices,
                size=query_size,
                replace=False,
            )
        )
