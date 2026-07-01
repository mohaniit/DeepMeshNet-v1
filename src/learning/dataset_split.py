"""
Dataset splitting utilities for DeepMeshNet-v1.

This module splits framework-independent GraphDataset objects into
train/validation/test subsets.

It does not depend on PyTorch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from src.learning.dataset import GraphDataset
from src.learning.graph_sample import GraphSample


@dataclass(frozen=True)
class DatasetSplit:
    """
    Container for train/validation/test dataset splits.
    """

    train: GraphDataset
    validation: GraphDataset
    test: GraphDataset

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
            "total": len(self.train) + len(self.validation) + len(self.test),
        }


def split_dataset(
    dataset: GraphDataset,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int | None = 42,
) -> DatasetSplit:
    """
    Randomly split GraphDataset into train/validation/test subsets.
    """
    _validate_dataset(dataset)
    _validate_ratios(train_ratio, validation_ratio, test_ratio)

    sample_ids = dataset.sample_ids()

    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    total = len(sample_ids)

    train_count = int(total * train_ratio)
    validation_count = int(total * validation_ratio)

    train_ids = sample_ids[:train_count]
    validation_ids = sample_ids[train_count: train_count + validation_count]
    test_ids = sample_ids[train_count + validation_count:]

    return DatasetSplit(
        train=_subset_dataset(dataset, train_ids),
        validation=_subset_dataset(dataset, validation_ids),
        test=_subset_dataset(dataset, test_ids),
    )


def split_dataset_by_counts(
    dataset: GraphDataset,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int | None = 42,
) -> DatasetSplit:
    """
    Randomly split GraphDataset using exact sample counts.
    """
    _validate_dataset(dataset)

    total_requested = train_count + validation_count + test_count

    if total_requested != len(dataset):
        raise ValueError(
            "Split counts must sum to dataset size. "
            f"Requested {total_requested}, got dataset size {len(dataset)}."
        )

    if min(train_count, validation_count, test_count) < 0:
        raise ValueError("Split counts cannot be negative.")

    sample_ids = dataset.sample_ids()

    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    train_ids = sample_ids[:train_count]
    validation_ids = sample_ids[train_count: train_count + validation_count]
    test_ids = sample_ids[train_count + validation_count:]

    return DatasetSplit(
        train=_subset_dataset(dataset, train_ids),
        validation=_subset_dataset(dataset, validation_ids),
        test=_subset_dataset(dataset, test_ids),
    )


def fixed_split_dataset(
    dataset: GraphDataset,
    train_ids: Sequence[str],
    validation_ids: Sequence[str] | None = None,
    test_ids: Sequence[str] | None = None,
) -> DatasetSplit:
    """
    Split GraphDataset using explicit sample IDs.

    Useful for publication benchmarks such as fixed CM500 splits.
    """
    _validate_dataset(dataset)

    validation_ids = validation_ids or []
    test_ids = test_ids or []

    all_ids = list(train_ids) + list(validation_ids) + list(test_ids)
    _validate_unique_ids(all_ids)
    _validate_ids_exist(dataset, all_ids)

    return DatasetSplit(
        train=_subset_dataset(dataset, train_ids),
        validation=_subset_dataset(dataset, validation_ids),
        test=_subset_dataset(dataset, test_ids),
    )


def train_test_split(
    dataset: GraphDataset,
    train_ratio: float = 0.80,
    seed: int | None = 42,
) -> tuple[GraphDataset, GraphDataset]:
    """
    Convenience train/test split.
    """
    _validate_dataset(dataset)

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    sample_ids = dataset.sample_ids()

    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    train_count = int(len(sample_ids) * train_ratio)

    train_ids = sample_ids[:train_count]
    test_ids = sample_ids[train_count:]

    return (
        _subset_dataset(dataset, train_ids),
        _subset_dataset(dataset, test_ids),
    )


def _subset_dataset(
    dataset: GraphDataset,
    sample_ids: Sequence[str],
) -> GraphDataset:
    subset = GraphDataset()

    for sample_id in sample_ids:
        sample = dataset.get_sample(sample_id)
        subset.add_sample(sample)

    return subset


def _validate_dataset(dataset: GraphDataset) -> None:
    if dataset is None:
        raise ValueError("GraphDataset cannot be None.")

    if len(dataset) == 0:
        raise ValueError("GraphDataset cannot be empty.")


def _validate_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    ratios = [train_ratio, validation_ratio, test_ratio]

    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios cannot be negative.")

    total = train_ratio + validation_ratio + test_ratio

    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Got {total}."
        )


def _validate_unique_ids(sample_ids: Sequence[str]) -> None:
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Split sample IDs must be unique.")


def _validate_ids_exist(
    dataset: GraphDataset,
    sample_ids: Sequence[str],
) -> None:
    missing = [
        sample_id
        for sample_id in sample_ids
        if not dataset.has_sample(sample_id)
    ]

    if missing:
        raise KeyError(f"Sample IDs not found: {missing}")


def sample_ids_from_dataset(dataset: GraphDataset) -> list[str]:
    """
    Return sorted sample IDs from dataset.
    """
    _validate_dataset(dataset)
    return dataset.sample_ids()


def make_dataset_from_samples(samples: Sequence[GraphSample]) -> GraphDataset:
    """
    Build GraphDataset from GraphSample sequence.
    """
    dataset = GraphDataset()

    for sample in samples:
        dataset.add_sample(sample)

    return dataset


__all__ = [
    "DatasetSplit",
    "split_dataset",
    "split_dataset_by_counts",
    "fixed_split_dataset",
    "train_test_split",
    "sample_ids_from_dataset",
    "make_dataset_from_samples",
]