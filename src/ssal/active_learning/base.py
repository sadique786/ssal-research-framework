from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ssal.active_learning.pool import ActiveLearningPool


class BaseQueryStrategy(ABC):
    """Base interface for active learning query strategies."""

    @abstractmethod
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
        """Select original dataset indices from the unlabeled pool."""
