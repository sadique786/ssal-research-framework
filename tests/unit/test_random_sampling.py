import numpy as np
import pytest

from ssal.active_learning import (
    ActiveLearningPool,
    RandomSamplingStrategy,
)


def test_random_sampling_returns_correct_number_of_samples() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
        seed=42,
    )

    strategy = RandomSamplingStrategy(
        seed=42,
    )

    selected_indices = strategy.query(
        pool,
        query_size=10,
    )

    assert len(selected_indices) == 10


def test_random_sampling_selects_only_unlabeled_samples() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
        seed=42,
    )

    strategy = RandomSamplingStrategy(
        seed=42,
    )

    selected_indices = strategy.query(
        pool,
        query_size=10,
    )

    assert np.all(
        np.isin(
            selected_indices,
            pool.unlabeled_indices,
        )
    )


def test_random_sampling_contains_no_duplicates() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
        seed=42,
    )

    strategy = RandomSamplingStrategy(
        seed=42,
    )

    selected_indices = strategy.query(
        pool,
        query_size=50,
    )

    assert len(
        np.unique(selected_indices)
    ) == len(selected_indices)


def test_random_sampling_is_reproducible() -> None:
    pool_one = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
        seed=42,
    )

    pool_two = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
        seed=42,
    )

    strategy_one = RandomSamplingStrategy(
        seed=123,
    )

    strategy_two = RandomSamplingStrategy(
        seed=123,
    )

    selected_one = strategy_one.query(
        pool_one,
        query_size=10,
    )

    selected_two = strategy_two.query(
        pool_two,
        query_size=10,
    )

    assert np.array_equal(
        selected_one,
        selected_two,
    )


def test_random_sampling_rejects_zero_query_size() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
    )

    strategy = RandomSamplingStrategy()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        strategy.query(
            pool,
            query_size=0,
        )


def test_random_sampling_rejects_query_size_larger_than_pool() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
    )

    strategy = RandomSamplingStrategy()

    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        strategy.query(
            pool,
            query_size=81,
        )


def test_random_sampling_integrates_with_pool_update() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
        seed=42,
    )

    strategy = RandomSamplingStrategy(
        seed=123,
    )

    selected_indices = strategy.query(
        pool,
        query_size=10,
    )

    pool.update(selected_indices)

    assert pool.num_labeled == 30
    assert pool.num_unlabeled == 70

    assert np.all(
        np.isin(
            selected_indices,
            pool.labeled_indices,
        )
    )
