from __future__ import annotations

import json

from omegaconf import OmegaConf

from ssal.experiments import ExperimentRun


def create_config():
    """Create a minimal experiment configuration."""
    return OmegaConf.create(
        {
            "seed": 42,
            "dataset": {
                "name": "cifar10",
            },
            "model": {
                "name": "resnet18",
            },
            "experiment": {
                "name": "l0_cifar10",
            },
            "active_learning": {
                "name": "random",
            },
        }
    )


def test_experiment_run_creates_expected_directories(tmp_path) -> None:
    """Test that the expected experiment directory structure is created."""
    cfg = create_config()

    run = ExperimentRun(
        cfg=cfg,
        root_dir=tmp_path,
    )

    run.create()

    assert run.run_dir == (
        tmp_path
        / "cifar10"
        / "l0_cifar10"
        / "random"
        / "seed_42"
    )
    assert run.run_dir.is_dir()
    assert run.checkpoint_dir.is_dir()
    assert run.log_dir.is_dir()


def test_experiment_run_saves_resolved_config(tmp_path) -> None:
    """Test that the experiment configuration is saved."""
    cfg = create_config()

    run = ExperimentRun(
        cfg=cfg,
        root_dir=tmp_path,
    )

    config_path = run.save_config()

    assert config_path.exists()

    saved_config = OmegaConf.load(config_path)

    assert saved_config.seed == 42
    assert saved_config.dataset.name == "cifar10"
    assert saved_config.model.name == "resnet18"


def test_experiment_run_saves_metadata(tmp_path) -> None:
    """Test that experiment metadata is saved."""
    cfg = create_config()

    run = ExperimentRun(
        cfg=cfg,
        root_dir=tmp_path,
    )

    metadata_path = run.save_metadata(
        device="cpu",
    )

    assert metadata_path.exists()

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["dataset"] == "cifar10"
    assert metadata["model"] == "resnet18"
    assert metadata["experiment"] == "l0_cifar10"
    assert metadata["active_learning"] == "random"
    assert metadata["seed"] == 42
    assert metadata["device"] == "cpu"
    assert "created_at" in metadata
