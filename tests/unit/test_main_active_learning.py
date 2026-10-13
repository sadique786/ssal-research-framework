from __future__ import annotations

from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from ssal.experiments import ExperimentLogger
from ssal.main import run_active_learning


def create_config():
    """Create a minimal configuration for testing."""
    return OmegaConf.create(
        {
            "device": "cpu",
            "dataset": {
                "root": "data",
                "train": {
                    "batch_size": 10,
                    "num_workers": 0,
                },
                "test": {
                    "batch_size": 10,
                    "num_workers": 0,
                },
            },
            "model": {
                "num_classes": 2,
                "pretrained": False,
            },
            "active_learning": {
		"name": "random",
                "initial_labeled_size": 20,
                "query_size": 10,
                "num_rounds": 2,
                "seed": 42,
            },
            "experiment": {
                "training": {
                    "epochs": 1,
                    "learning_rate": 0.001,
                },
            },
        }
    )


def create_dataset() -> TensorDataset:
    """Create a small synthetic binary-classification dataset."""
    torch.manual_seed(42)
    features = torch.randn(100, 4)
    labels = torch.randint(0, 2, (100,))
    return TensorDataset(features, labels)


def test_run_active_learning_executes_configured_rounds(tmp_path) -> None:
    """Test that the runner executes configured AL rounds."""
    cfg = create_config()
    train_dataset = create_dataset()
    acquisition_dataset = create_dataset()

    logger = ExperimentLogger(
        run_dir=tmp_path,
    )

    with patch("ssal.main.CIFAR10DataModule") as data_cls, patch(
        "ssal.main.ResNet18Classifier"
    ) as model_cls, patch(
        "ssal.main.SupervisedTrainer"
    ), patch(
        "ssal.main.ActiveLearningExperiment"
    ) as experiment_cls:
        data_cls.return_value.train_dataset.return_value = train_dataset
        data_cls.return_value.acquisition_dataset.return_value = acquisition_dataset
        data_cls.return_value.test_dataloader.return_value = DataLoader(
            train_dataset,
            batch_size=10,
            shuffle=False,
        )

        model_cls.return_value = torch.nn.Linear(4, 2)

        experiment = experiment_cls.return_value

        experiment.run_round.return_value = {
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "test_loss": 1.1,
            "test_accuracy": 0.4,
        }

        experiment.query.return_value = list(range(10))

        run_active_learning(
            cfg,
            logger=logger,
        )

    assert experiment_cls.call_count == 1
    data_cls.return_value.acquisition_dataset.assert_called_once_with()
    assert experiment_cls.call_args.kwargs["dataset"] is train_dataset
    assert experiment_cls.call_args.kwargs["acquisition_dataset"] is acquisition_dataset
    assert experiment.run_round.call_count == 2

    # Query happens only between rounds.
    assert experiment.query.call_count == 1
    assert experiment.update_pool.call_count == 1

    experiment.query.assert_called_once()
    assert experiment.query.call_args.kwargs["query_size"] == 10
    assert experiment.query.call_args.kwargs["batch_size"] == 10
    assert "trainer" in experiment.query.call_args.kwargs

    experiment.update_pool.assert_called_once_with(
        list(range(10)),
    )
