"""
PyTorch dataset adapter for DeepMeshNet-v1.

This module adapts the framework-independent GraphDataset into a
PyTorch-compatible dataset.

It returns PyTorch Geometric Data objects from GraphSample instances.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from src.learning.dataset import GraphDataset
from src.learning.graph_sample import GraphSample


class TorchGraphDataset(Dataset):
    """
    PyTorch-compatible adapter for GraphDataset.
    """

    def __init__(
        self,
        dataset: GraphDataset,
        include_edge_attr: bool = True,
        validate: bool = True,
    ) -> None:
        if dataset is None:
            raise ValueError("GraphDataset cannot be None.")

        self.dataset = dataset
        self.include_edge_attr = include_edge_attr
        self.validate = validate
        self.sample_ids = dataset.sample_ids()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Any:
        if index < 0 or index >= len(self.sample_ids):
            raise IndexError(index)

        sample_id = self.sample_ids[index]
        sample = self.dataset.get_sample(sample_id)

        return sample.to_pyg_data(
            validate=self.validate,
            include_edge_attr=self.include_edge_attr,
        )

    def get_sample(self, index: int) -> GraphSample:
        if index < 0 or index >= len(self.sample_ids):
            raise IndexError(index)

        return self.dataset.get_sample(self.sample_ids[index])

    def get_sample_id(self, index: int) -> str:
        if index < 0 or index >= len(self.sample_ids):
            raise IndexError(index)

        return self.sample_ids[index]

    def summary(self) -> dict[str, Any]:
        return {
            "num_samples": len(self),
            "include_edge_attr": self.include_edge_attr,
            "validate": self.validate,
            "sample_ids": list(self.sample_ids),
        }

    def __repr__(self) -> str:
        return (
            f"TorchGraphDataset("
            f"samples={len(self)}, "
            f"include_edge_attr={self.include_edge_attr}, "
            f"validate={self.validate})"
        )


def make_torch_dataset(
    dataset: GraphDataset,
    include_edge_attr: bool = True,
    validate: bool = True,
) -> TorchGraphDataset:
    """
    Convenience factory for TorchGraphDataset.
    """
    return TorchGraphDataset(
        dataset=dataset,
        include_edge_attr=include_edge_attr,
        validate=validate,
    )


__all__ = [
    "TorchGraphDataset",
    "make_torch_dataset",
]