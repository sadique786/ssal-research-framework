import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ssal.models import ResNet18Classifier
from ssal.training import SupervisedTrainer


def create_test_loader() -> DataLoader:
    images = torch.randn(
        8,
        3,
        32,
        32,
    )

    labels = torch.randint(
        low=0,
        high=10,
        size=(8,),
    )

    dataset = TensorDataset(
        images,
        labels,
    )

    return DataLoader(
        dataset,
        batch_size=4,
    )


def create_trainer() -> SupervisedTrainer:
    model = ResNet18Classifier(
        num_classes=10,
        pretrained=False,
    )

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    return SupervisedTrainer(
        model=model,
        device=torch.device("cpu"),
        optimizer=optimizer,
    )


def test_train_epoch_returns_metrics() -> None:
    trainer = create_trainer()

    loader = create_test_loader()

    metrics = trainer.train_epoch(loader)

    assert "loss" in metrics
    assert "accuracy" in metrics

    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_returns_metrics() -> None:
    trainer = create_trainer()

    loader = create_test_loader()

    metrics = trainer.evaluate(loader)

    assert "loss" in metrics
    assert "accuracy" in metrics

    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_training_updates_parameters() -> None:
    trainer = create_trainer()

    loader = create_test_loader()

    parameters_before = [
        parameter.detach().clone()
        for parameter in trainer.model.parameters()
    ]

    trainer.train_epoch(loader)

    parameters_after = list(
        trainer.model.parameters()
    )

    changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after,
        )
    )

    assert changed
