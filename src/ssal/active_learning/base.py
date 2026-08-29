from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ssal.active_learning.pool import ActiveLearningPool


class BaseQueryStrategy(ABC):
    """Base interface for active learning query strategies."""

    @abstractmethod
    def query(
        self,
        pool: ActiveLearningPool,
        query_size: int,
    ) -> np.ndarray:
        """Select sample indices from the unlabeled pool."""
