import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import TensorDataset

from ssal.active_learning import (
    ActiveLearningExperiment,
    ActiveLearningPool,
    LeastConfidenceStrategy,
)
from ssal.training import SupervisedTrainer


class FixedLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "logits",
            torch.tensor([
                [8.0, 0.0],
                [1.0, 0.9],
                [0.0, 8.0],
                [1.0, 1.0],
                [5.0, 0.0],
                [0.5, 0.4],
            ]),
        )

    def forward(self, x):
        indices = x[:, 0].long()
        return self.logits[indices] + self.anchor * 0


def make_case():
    features = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    labels = torch.zeros(6, dtype=torch.long)
    dataset = TensorDataset(features, labels)
    pool = ActiveLearningPool(dataset_size=6, initial_labeled_size=1, seed=10)
    model = FixedLogitModel()
    return dataset, pool, model


def test_least_confidence_returns_unique_unlabeled_indices():
    dataset, pool, model = make_case()
    strategy = LeastConfidenceStrategy()

    selected = strategy.query(
        pool,
        2,
        dataset=dataset,
        model=model,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert len(selected) == 2
    assert len(np.unique(selected)) == 2
    assert np.all(np.isin(selected, pool.unlabeled_indices))


def test_least_confidence_preserves_model_training_state():
    dataset, pool, model = make_case()
    model.train()

    LeastConfidenceStrategy().query(
        pool,
        1,
        dataset=dataset,
        model=model,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert model.training


def test_least_confidence_requires_model_context():
    _, pool, _ = make_case()

    with pytest.raises(ValueError, match="requires dataset, model, and device"):
        LeastConfidenceStrategy().query(pool, 1)


def test_least_confidence_rejects_invalid_query_size():
    dataset, pool, model = make_case()

    with pytest.raises(ValueError, match="greater than zero"):
        LeastConfidenceStrategy().query(
            pool, 0, dataset=dataset, model=model,
            device=torch.device("cpu"),
        )


def test_least_confidence_cpu_smoke_uses_original_pool_indices():
    """Exercise the real pool, experiment, and strategy on CPU-only data."""
    acquisition_dataset, pool, model = make_case()
    training_dataset = TensorDataset(
        torch.full((6, 1), 99.0),
        torch.zeros(6, dtype=torch.long),
    )
    trainer = SupervisedTrainer(
        model=model,
        device=torch.device("cpu"),
        optimizer=SGD(model.parameters(), lr=0.1),
    )
    experiment = ActiveLearningExperiment(
        dataset=training_dataset,
        acquisition_dataset=acquisition_dataset,
        pool=pool,
        strategy=LeastConfidenceStrategy(),
    )

    selected = experiment.query(query_size=3, trainer=trainer, batch_size=2)

    assert selected == [1, 3, 5]
    assert len(selected) == len(set(selected))
    assert np.all(np.isin(selected, pool.unlabeled_indices))
