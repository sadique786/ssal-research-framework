from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from ssal.active_learning import (
    ActiveLearningExperiment,
    ActiveLearningPool,
    RandomSamplingStrategy,
)
from ssal.data import CIFAR10DataModule
from ssal.models import ResNet18Classifier
from ssal.training import SupervisedTrainer
from ssal.utils import get_device, set_seed


def run_active_learning(
    cfg: DictConfig,
    device: torch.device | None = None,
) -> ActiveLearningExperiment:
    """Run the configured active learning experiment."""

    if device is None:
        device = get_device(cfg.device)

    data = CIFAR10DataModule(
        root=cfg.dataset.root,
        train_batch_size=cfg.dataset.train.batch_size,
        test_batch_size=cfg.dataset.test.batch_size,
        num_workers=cfg.dataset.train.num_workers,
        download=True,
    )

    train_dataset = data.train_dataset()
    test_loader = data.test_dataloader()

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

    pool = ActiveLearningPool(
        dataset_size=len(train_dataset),
        initial_labeled_size=cfg.active_learning.initial_labeled_size,
        seed=cfg.active_learning.seed,
    )

    strategy = RandomSamplingStrategy(
        seed=cfg.active_learning.seed,
    )

    experiment = ActiveLearningExperiment(
        dataset=train_dataset,
        pool=pool,
        strategy=strategy,
    )

    num_rounds = cfg.active_learning.num_rounds
    epochs = cfg.experiment.training.epochs
    batch_size = cfg.dataset.train.batch_size
    query_size = cfg.active_learning.query_size

    print(
        f"Initial labeled: {pool.num_labeled:,}"
    )
    print(
        f"Initial unlabeled: {pool.num_unlabeled:,}"
    )

    for round_index in range(1, num_rounds + 1):
        metrics = experiment.run_round(
            trainer=trainer,
            test_loader=test_loader,
            epochs=epochs,
            batch_size=batch_size,
        )

        print(
            f"Round [{round_index}/{num_rounds}] | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Train Acc: "
            f"{metrics['train_accuracy'] * 100:.2f}% | "
            f"Test Loss: {metrics['test_loss']:.4f} | "
            f"Test Acc: "
            f"{metrics['test_accuracy'] * 100:.2f}%"
        )

        if round_index < num_rounds:
            selected = experiment.query(
                query_size=query_size,
            )

            experiment.update_pool(selected)

            print(
                f"After query | "
                f"Labeled: {pool.num_labeled:,} | "
                f"Unlabeled: {pool.num_unlabeled:,}"
            )

    return experiment


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Run the SSAL active learning experiment."""

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

    run_active_learning(
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    main()