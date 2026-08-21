from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from ssal.data.base import BaseDataModule
from ssal.data.transforms import (
    get_cifar10_test_transform,
    get_cifar10_train_transform,
)


class CIFAR10DataModule(BaseDataModule):
    """Data module for the CIFAR-10 dataset."""

    def __init__(
        self,
        root: str | Path = "data",
        train_batch_size: int = 128,
        test_batch_size: int = 256,
        num_workers: int = 0,
        download: bool = True,
    ) -> None:
        self.root = Path(root)

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.download = download

        self._train_dataset: CIFAR10 | None = None
        self._test_dataset: CIFAR10 | None = None

    def setup(self) -> None:
        """Load the CIFAR-10 training and test datasets."""

        self._train_dataset = CIFAR10(
            root=self.root,
            train=True,
            transform=get_cifar10_train_transform(),
            download=self.download,
        )

        self._test_dataset = CIFAR10(
            root=self.root,
            train=False,
            transform=get_cifar10_test_transform(),
            download=self.download,
        )

    def train_dataset(self) -> Dataset:
        """Return the CIFAR-10 training dataset."""

        if self._train_dataset is None:
            self.setup()

        return self._train_dataset

    def test_dataset(self) -> Dataset:
        """Return the CIFAR-10 test dataset."""

        if self._test_dataset is None:
            self.setup()

        return self._test_dataset

    def train_dataloader(self) -> DataLoader:
        """Return the CIFAR-10 training data loader."""

        return DataLoader(
            self.train_dataset(),
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the CIFAR-10 test data loader."""

        return DataLoader(
            self.test_dataset(),
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
