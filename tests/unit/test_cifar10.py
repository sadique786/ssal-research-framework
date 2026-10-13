from pathlib import Path

from ssal.data import CIFAR10DataModule


def test_cifar10_datasets() -> None:
    data = CIFAR10DataModule(
        root=Path.home() / "datasets" / "CIFAR10",
        num_workers=0,
        download=False,
    )

    train_dataset = data.train_dataset()
    test_dataset = data.test_dataset()

    assert len(train_dataset) == 50_000
    assert len(test_dataset) == 10_000


def test_cifar10_sample_shape_and_label() -> None:
    data = CIFAR10DataModule(
        root=Path.home() / "datasets" / "CIFAR10",
        num_workers=0,
        download=False,
    )

    train_dataset = data.train_dataset()

    image, label = train_dataset[0]

    assert list(image.shape) == [3, 32, 32]
    assert 0 <= label <= 9


def test_cifar10_classes() -> None:
    data = CIFAR10DataModule(
        root=Path.home() / "datasets" / "CIFAR10",
        num_workers=0,
        download=False,
    )

    train_dataset = data.train_dataset()

    assert len(train_dataset.classes) == 10
