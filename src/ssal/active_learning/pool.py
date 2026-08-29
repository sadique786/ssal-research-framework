from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ActiveLearningPool:
    """Manage labeled and unlabeled sample indices for active learning."""

    def __init__(
        self,
        dataset_size: int,
        initial_labeled_size: int,
        seed: int = 42,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be greater than zero.")

        if initial_labeled_size <= 0:
            raise ValueError(
                "initial_labeled_size must be greater than zero."
            )

        if initial_labeled_size >= dataset_size:
            raise ValueError(
                "initial_labeled_size must be smaller than dataset_size."
            )

        self.dataset_size = dataset_size

        rng = np.random.default_rng(seed)

        all_indices = np.arange(dataset_size)

        labeled_indices = rng.choice(
            all_indices,
            size=initial_labeled_size,
            replace=False,
        )

        self._labeled_indices = np.sort(labeled_indices)

        self._unlabeled_indices = np.setdiff1d(
            all_indices,
            self._labeled_indices,
        )

    @property
    def labeled_indices(self) -> np.ndarray:
        """Return the indices currently available for labeling/training."""

        return self._labeled_indices.copy()

    @property
    def unlabeled_indices(self) -> np.ndarray:
        """Return the indices currently available for querying."""

        return self._unlabeled_indices.copy()

    @property
    def num_labeled(self) -> int:
        """Return the number of labeled samples."""

        return len(self._labeled_indices)

    @property
    def num_unlabeled(self) -> int:
        """Return the number of unlabeled samples."""

        return len(self._unlabeled_indices)

    def update(self, indices: Sequence[int] | np.ndarray) -> None:
        """Move selected indices from the unlabeled pool to the labeled pool."""

        selected_indices = np.asarray(indices, dtype=int)

        if selected_indices.ndim != 1:
            raise ValueError(
                "indices must be a one-dimensional sequence."
            )

        if len(selected_indices) == 0:
            raise ValueError(
                "indices must contain at least one sample."
            )

        if len(np.unique(selected_indices)) != len(selected_indices):
            raise ValueError(
                "indices must not contain duplicates."
            )

        if np.any(selected_indices < 0) or np.any(
            selected_indices >= self.dataset_size
        ):
            raise ValueError(
                "indices contain values outside the dataset range."
            )

        is_unlabeled = np.isin(
            selected_indices,
            self._unlabeled_indices,
        )

        if not np.all(is_unlabeled):
            raise ValueError(
                "all selected indices must belong to the unlabeled pool."
            )

        self._labeled_indices = np.sort(
            np.concatenate(
                [
                    self._labeled_indices,
                    selected_indices,
                ]
            )
        )

        self._unlabeled_indices = np.setdiff1d(
            self._unlabeled_indices,
            selected_indices,
        )
