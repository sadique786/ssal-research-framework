from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_random_active_learning_config() -> None:
    """Test that the random active learning configuration loads correctly."""
    config_dir = Path("configs").resolve()

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(config_dir),
    ):
        cfg = compose(config_name="config")

    assert cfg.active_learning.name == "random"
    assert cfg.active_learning.initial_labeled_size == 5000
    assert cfg.active_learning.query_size == 2500
    assert cfg.active_learning.num_rounds == 5
    assert cfg.active_learning.seed == 42
