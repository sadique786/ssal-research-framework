from __future__ import annotations

import numpy as np

from ssal.active_learning.base import BaseQueryStrategy
from ssal.active_learning.pool import ActiveLearningPool


class RandomSamplingStrategy(BaseQueryStrategy):
    """Randomly select samples from the unlabeled pool."""

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)

    def query(
        self,
        pool: ActiveLearningPool,
        query_size: int,
    ) -> np.ndarray:
        """Randomly select indices from the unlabeled pool."""

        if query_size <= 0:
            raise ValueError(
                "query_size must be greater than zero."
            )

        if query_size > pool.num_unlabeled:
            raise ValueError(
                "query_size cannot exceed the number of unlabeled samples."
            )

        selected_indices = self._rng.choice(
            pool.unlabeled_indices,
            size=query_size,
            replace=False,
        )

        return np.sort(selected_indices)
