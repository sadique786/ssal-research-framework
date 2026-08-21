from __future__ import annotations

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, Dataset


class BaseDataModule(ABC):
    """Abstract interface for dataset modules."""

    @abstractmethod
    def train_dataset(self) -> Dataset:
        """Return the training dataset."""

    @abstractmethod
    def test_dataset(self) -> Dataset:
        """Return the test dataset."""

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return the training data loader."""

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Return the test data loader."""
