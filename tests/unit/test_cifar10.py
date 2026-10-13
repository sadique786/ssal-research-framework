from pathlib import Path
from unittest.mock import patch

import torch

from ssal.data import CIFAR10DataModule


class FakeCIFAR10:
    """Small stand-in that records CIFAR-10 constructor arguments."""

    instances: list["FakeCIFAR10"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.classes = list(range(10))
        self.__class__.instances.append(self)

    def __len__(self) -> int:
        return 50_000 if self.kwargs["train"] else 10_000

    def __getitem__(self, index: int):
        return torch.zeros(3, 32, 32), index % 10



def test_cifar10_datasets() -> None:
    with patch("ssal.data.cifar10.CIFAR10", FakeCIFAR10):
        data = CIFAR10DataModule(root=Path("data/cifar10"), download=False)
        train_dataset = data.train_dataset()
        test_dataset = data.test_dataset()

    assert len(train_dataset) == 50_000
    assert len(test_dataset) == 10_000


def test_cifar10_classes() -> None:
    with patch("ssal.data.cifar10.CIFAR10", FakeCIFAR10):
        data = CIFAR10DataModule(root=Path("data/cifar10"), download=False)
        train_dataset = data.train_dataset()

    image, label = train_dataset[0]

    assert len(train_dataset.classes) == 10
    assert list(image.shape) == [3, 32, 32]
    assert 0 <= label <= 9


def test_cifar10_acquisition_dataset_uses_deterministic_transform() -> None:
    """Acquisition uses CIFAR-10's training split with a distinct transform."""

    root = Path("data/cifar10")
    FakeCIFAR10.instances.clear()
    with patch("ssal.data.cifar10.CIFAR10", FakeCIFAR10), patch(
        "ssal.data.cifar10.get_cifar10_train_transform",
        return_value="train-transform",
    ), patch(
        "ssal.data.cifar10.get_cifar10_test_transform",
        return_value="deterministic-transform",
    ):
        data = CIFAR10DataModule(root=root, download=False)
        data.train_dataset()
        data.acquisition_dataset()

    train_call = FakeCIFAR10.instances[0].kwargs
    acquisition_call = FakeCIFAR10.instances[2].kwargs
    assert train_call["root"] == root
    assert acquisition_call["root"] == root
    assert train_call["train"] is True
    assert acquisition_call["train"] is True
    assert train_call["transform"] == "train-transform"
    assert acquisition_call["transform"] == "deterministic-transform"
