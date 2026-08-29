import numpy as np
import pytest

from ssal.active_learning import ActiveLearningPool


def test_pool_initialization() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
        seed=42,
    )

    assert pool.num_labeled == 20
    assert pool.num_unlabeled == 80


def test_pool_contains_all_samples() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
        seed=42,
    )

    all_indices = np.concatenate(
        [
            pool.labeled_indices,
            pool.unlabeled_indices,
        ]
    )

    assert len(np.unique(all_indices)) == 100
    assert set(all_indices) == set(range(100))


def test_labeled_and_unlabeled_pools_do_not_overlap() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
        seed=42,
    )

    overlap = np.intersect1d(
        pool.labeled_indices,
        pool.unlabeled_indices,
    )

    assert len(overlap) == 0


def test_pool_initialization_is_reproducible() -> None:
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

    assert np.array_equal(
        pool_one.labeled_indices,
        pool_two.labeled_indices,
    )


def test_update_moves_samples_to_labeled_pool() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
        seed=42,
    )

    selected_indices = pool.unlabeled_indices[:10]

    pool.update(selected_indices)

    assert pool.num_labeled == 30
    assert pool.num_unlabeled == 70

    assert np.all(
        np.isin(
            selected_indices,
            pool.labeled_indices,
        )
    )

    assert not np.any(
        np.isin(
            selected_indices,
            pool.unlabeled_indices,
        )
    )


def test_update_rejects_duplicate_indices() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
    )

    selected_index = int(pool.unlabeled_indices[0])

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        pool.update(
            [
                selected_index,
                selected_index,
            ]
        )


def test_update_rejects_labeled_indices() -> None:
    pool = ActiveLearningPool(
        dataset_size=100,
        initial_labeled_size=20,
    )

    labeled_index = int(pool.labeled_indices[0])

    with pytest.raises(
        ValueError,
        match="unlabeled pool",
    ):
        pool.update([labeled_index])


@pytest.mark.parametrize(
    ("dataset_size", "initial_labeled_size"),
    [
        (0, 1),
        (100, 0),
        (100, 100),
        (100, 101),
    ],
)
def test_invalid_pool_sizes_raise_error(
    dataset_size: int,
    initial_labeled_size: int,
) -> None:
    with pytest.raises(ValueError):
        ActiveLearningPool(
            dataset_size=dataset_size,
            initial_labeled_size=initial_labeled_size,
        )
