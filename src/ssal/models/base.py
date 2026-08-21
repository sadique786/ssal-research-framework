from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Base interface for SSAL models."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero.")

        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
