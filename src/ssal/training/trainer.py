from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BaseTrainer(ABC):
    """Base interface for model trainers."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device

        self.model.to(self.device)

    @abstractmethod
    def train_epoch(self, loader: Any) -> dict[str, float]:
        """Train for one epoch."""

    @abstractmethod
    def evaluate(self, loader: Any) -> dict[str, float]:
        """Evaluate the model."""
