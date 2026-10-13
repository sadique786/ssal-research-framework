from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from ssal.active_learning import (
    ActiveLearningExperiment,
    ActiveLearningPool,
    RandomSamplingStrategy,
)
from ssal.training import SupervisedTrainer


class RecordingStrategy(RandomSamplingStrategy):
    """Record the dataset supplied by the experiment query path."""

    def __init__(self) -> None:
        super().__init__(seed=42)
        self.dataset = None

    def query(self, pool, query_size, **kwargs):
        self.dataset = kwargs["dataset"]
        return pool.unlabeled_indices[1 : query_size + 1]


def create_dataset() -> TensorDataset:
    """Create a small synthetic classification dataset."""

    torch.manual_seed(42)

    features = torch.randn(100, 4)
    labels = torch.randint(0, 2, (100,))

    return TensorDataset(features, labels)


def create_model() -> nn.Module:
    """Create a small test model."""

    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def create_trainer(model: nn.Module) -> SupervisedTrainer:
    """Create a supervised trainer for testing."""

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    return SupervisedTrainer(
        model=model,
        device=torch.device("cpu"),
        optimizer=optimizer,
    )


def test_experiment_initializes_correctly() -> None:
    """Test experiment initialization."""

    dataset = create_dataset()

    pool = ActiveLearningPool(
        dataset_size=len(dataset),
        initial_labeled_size=20,
        seed=42,
    )

    strategy = RandomSamplingStrategy(
        seed=42,
    )

    experiment = ActiveLearningExperiment(
        dataset=dataset,
        pool=pool,
        strategy=strategy,
    )

    assert experiment.dataset is dataset
    assert experiment.acquisition_dataset is dataset
    assert experiment.pool is pool
    assert experiment.strategy is strategy


def test_experiment_creates_labeled_dataloader() -> None:
    """Test that only labeled samples are used for training."""

    dataset = create_dataset()

    pool = ActiveLearningPool(
        dataset_size=len(dataset),
        initial_labeled_size=20,
        seed=42,
    )

    experiment = ActiveLearningExperiment(
        dataset=dataset,
        pool=pool,
        strategy=RandomSamplingStrategy(seed=42),
    )

    loader = experiment.create_labeled_dataloader(
        batch_size=5,
    )

    assert len(loader.dataset) == 20
    assert loader.dataset.dataset is dataset


def test_experiment_uses_acquisition_dataset_for_query() -> None:
    """Training and acquisition datasets remain separate in their paths."""

    training_dataset = create_dataset()
    acquisition_dataset = TensorDataset(
        torch.arange(100, dtype=torch.float32).unsqueeze(1),
        torch.zeros(100, dtype=torch.long),
    )
    pool = ActiveLearningPool(dataset_size=100, initial_labeled_size=20, seed=42)
    strategy = RecordingStrategy()
    experiment = ActiveLearningExperiment(
        dataset=training_dataset,
        acquisition_dataset=acquisition_dataset,
        pool=pool,
        strategy=strategy,
    )

    train_loader = experiment.create_labeled_dataloader(batch_size=5)
    selected = experiment.query(query_size=3)

    expected = pool.unlabeled_indices[1:4]
    assert train_loader.dataset.dataset is training_dataset
    assert strategy.dataset is acquisition_dataset
    assert selected == expected.tolist()
    assert np.all(np.isin(selected, pool.unlabeled_indices))


def test_experiment_rejects_mismatched_acquisition_dataset_size() -> None:
    dataset = create_dataset()
    pool = ActiveLearningPool(dataset_size=100, initial_labeled_size=20, seed=42)
    mismatched_dataset = TensorDataset(
        torch.randn(99, 4), torch.zeros(99, dtype=torch.long)
    )

    with pytest.raises(ValueError, match="Acquisition dataset size"):
        ActiveLearningExperiment(
            dataset=dataset,
            acquisition_dataset=mismatched_dataset,
            pool=pool,
            strategy=RandomSamplingStrategy(seed=42),
        )


def test_experiment_runs_one_round() -> None:
    """Test a complete active learning round."""

    dataset = create_dataset()

    pool = ActiveLearningPool(
        dataset_size=len(dataset),
        initial_labeled_size=20,
        seed=42,
    )

    experiment = ActiveLearningExperiment(
        dataset=dataset,
        pool=pool,
        strategy=RandomSamplingStrategy(seed=42),
    )

    model = create_model()

    trainer = create_trainer(model)

    test_loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
    )

    metrics = experiment.run_round(
        trainer=trainer,
        test_loader=test_loader,
        epochs=1,
        batch_size=5,
    )

    assert "train_loss" in metrics
    assert "train_accuracy" in metrics
    assert "test_loss" in metrics
    assert "test_accuracy" in metrics


def test_experiment_updates_pool_after_query() -> None:
    """Test that querying updates the active learning pool."""

    dataset = create_dataset()

    pool = ActiveLearningPool(
        dataset_size=len(dataset),
        initial_labeled_size=20,
        seed=42,
    )

    experiment = ActiveLearningExperiment(
        dataset=dataset,
        pool=pool,
        strategy=RandomSamplingStrategy(seed=42),
    )

    initial_labeled = pool.num_labeled

    selected = experiment.query(
        query_size=10,
    )

    assert len(selected) == 10

    experiment.update_pool(selected)

    assert pool.num_labeled == initial_labeled + 10
    assert pool.num_unlabeled == 70


def test_experiment_records_history() -> None:
    """Test experiment metric history."""

    dataset = create_dataset()

    pool = ActiveLearningPool(
        dataset_size=len(dataset),
        initial_labeled_size=20,
        seed=42,
    )

    experiment = ActiveLearningExperiment(
        dataset=dataset,
        pool=pool,
        strategy=RandomSamplingStrategy(seed=42),
    )

    model = create_model()

    trainer = create_trainer(model)

    test_loader = DataLoader(
        dataset,
        batch_size=10,
    )

    experiment.run_round(
        trainer=trainer,
        test_loader=test_loader,
        epochs=1,
        batch_size=5,
    )

    assert len(experiment.history) == 1
