from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from ssal.data import CIFAR10DataModule
from ssal.models import ResNet18Classifier
from ssal.training import SupervisedTrainer
from ssal.utils import get_device, set_seed


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Run the SSAL supervised baseline experiment."""

    set_seed(cfg.seed)

    device = get_device(cfg.device)

    print("=" * 60)
    print("SSAL Research Framework")
    print("=" * 60)
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Model: {cfg.model.name}")
    print(f"Device: {device}")
    print(f"Seed: {cfg.seed}")
    print("=" * 60)

    data = CIFAR10DataModule(
        root=cfg.dataset.root,
        train_batch_size=cfg.dataset.train.batch_size,
        test_batch_size=cfg.dataset.test.batch_size,
        num_workers=cfg.dataset.train.num_workers,
        download=True,
    )

    model = ResNet18Classifier(
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.experiment.training.learning_rate,
    )

    trainer = SupervisedTrainer(
        model=model,
        device=device,
        optimizer=optimizer,
    )

    train_loader = data.train_dataloader()

    test_loader = data.test_dataloader()

    epochs = cfg.experiment.training.epochs

    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)

        test_metrics = trainer.evaluate(test_loader)

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy'] * 100:.2f}% | "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Acc: {test_metrics['accuracy'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
