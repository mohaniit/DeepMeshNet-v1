"""
Graph dataset for DeepMeshNet-v1.

A GraphDataset is a collection of GraphSample objects representing
multiple CAD models for machine learning.

This module is framework-independent and does not depend on PyTorch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from src.learning.graph_sample import GraphSample


class GraphDataset:
    """
    Repository of GraphSample objects.
    """

    def __init__(self) -> None:
        self._samples: dict[str, GraphSample] = {}

    # ------------------------------------------------------------------
    # Sample management
    # ------------------------------------------------------------------

    def add_sample(self, sample: GraphSample) -> None:
        if sample is None:
            raise ValueError("GraphSample cannot be None.")

        if sample.sample_id in self._samples:
            raise ValueError(
                f"Sample '{sample.sample_id}' already exists."
            )

        self._samples[sample.sample_id] = sample

    def remove_sample(self, sample_id: str) -> None:
        if sample_id not in self._samples:
            raise KeyError(sample_id)

        del self._samples[sample_id]

    def get_sample(self, sample_id: str) -> GraphSample:
        if sample_id not in self._samples:
            raise KeyError(sample_id)

        return self._samples[sample_id]

    def has_sample(self, sample_id: str) -> bool:
        return sample_id in self._samples

    def sample_ids(self) -> list[str]:
        return sorted(self._samples.keys())

    # ------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    @property
    def total_nodes(self) -> int:
        return sum(sample.num_nodes for sample in self._samples.values())

    @property
    def total_edges(self) -> int:
        return sum(sample.num_edges for sample in self._samples.values())

    @property
    def average_nodes(self) -> float:
        if self.num_samples == 0:
            return 0.0

        return self.total_nodes / self.num_samples

    @property
    def average_edges(self) -> float:
        if self.num_samples == 0:
            return 0.0

        return self.total_edges / self.num_samples

    def summary(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "average_nodes": self.average_nodes,
            "average_edges": self.average_edges,
        }

    # ------------------------------------------------------------------
    # Metadata export
    # ------------------------------------------------------------------

    def export_metadata(self) -> list[dict]:
        return [
            sample.summary()
            for sample in self
        ]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_metadata_json(
        self,
        filename: str | Path,
    ) -> Path:
        """
        Save dataset metadata.

        Note:
            This does NOT save CADGraph objects.
            It only exports dataset summaries.
        """

        filename = Path(filename)

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(filename, "w", encoding="utf-8") as fp:
            json.dump(
                self.export_metadata(),
                fp,
                indent=4,
            )

        return filename

    # ------------------------------------------------------------------
    # Python interfaces
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[GraphSample]:
        return iter(self._samples.values())

    def __getitem__(self, sample_id: str) -> GraphSample:
        return self.get_sample(sample_id)

    def __contains__(self, sample_id: str) -> bool:
        return self.has_sample(sample_id)

    def __repr__(self) -> str:
        return (
            f"GraphDataset("
            f"samples={self.num_samples}, "
            f"nodes={self.total_nodes}, "
            f"edges={self.total_edges})"
        )


__all__ = [
    "GraphDataset",
]