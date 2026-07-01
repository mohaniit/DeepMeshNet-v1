"""
DataLoader utilities for DeepMeshNet-v1.

This module creates PyTorch Geometric DataLoader objects from
TorchGraphDataset instances.

PyTorch/PyG dependency is intentionally limited to this adapter layer.
"""

from __future__ import annotations

from typing import Any

from torch_geometric.loader import DataLoader

from src.learning.dataset import GraphDataset
from src.learning.torch_dataset import TorchGraphDataset, make_torch_dataset


def make_graph_dataloader(
    dataset: TorchGraphDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create PyTorch Geometric DataLoader from TorchGraphDataset.
    """
    if dataset is None:
        raise ValueError("TorchGraphDataset cannot be None.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def make_dataloader_from_graph_dataset(
    dataset: GraphDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    include_edge_attr: bool = True,
    validate: bool = True,
) -> DataLoader:
    """
    Create PyTorch Geometric DataLoader directly from GraphDataset.
    """
    if dataset is None:
        raise ValueError("GraphDataset cannot be None.")

    torch_dataset = make_torch_dataset(
        dataset=dataset,
        include_edge_attr=include_edge_attr,
        validate=validate,
    )

    return make_graph_dataloader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def dataloader_summary(loader: DataLoader) -> dict[str, Any]:
    """
    Return basic DataLoader summary.
    """
    if loader is None:
        raise ValueError("DataLoader cannot be None.")

    dataset = loader.dataset

    return {
        "num_samples": len(dataset),
        "batch_size": loader.batch_size,
        "num_batches": len(loader),
        "drop_last": loader.drop_last,
    }


__all__ = [
    "make_graph_dataloader",
    "make_dataloader_from_graph_dataset",
    "dataloader_summary",
]