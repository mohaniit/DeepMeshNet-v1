"""
Graph Tensor representation for DeepMeshNet-v1.

This module defines a framework-independent numerical representation
of a CAD graph.

GraphTensor serves as the bridge between the CAD Graph Engine (CGE)
and machine learning frameworks such as PyTorch Geometric.

Pipeline
--------
CADModel
    ↓
CADGraph
    ↓
GraphTensor
    ↓
PyTorch Geometric / DGL / ONNX / Future frameworks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GraphTensor:
    """
    Framework-independent graph tensor representation.
    """

    x: np.ndarray
    edge_index: np.ndarray

    y: np.ndarray | None = None
    edge_attr: np.ndarray | None = None

    graph_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        """Number of graph nodes."""
        return int(self.x.shape[0])

    @property
    def num_node_features(self) -> int:
        """Number of node features."""
        if self.x.ndim == 1:
            return 1
        return int(self.x.shape[1])

    @property
    def num_edges(self) -> int:
        """Number of graph edges."""
        if self.edge_index.size == 0:
            return 0
        return int(self.edge_index.shape[1])

    @property
    def has_labels(self) -> bool:
        """Return True if labels exist."""
        return self.y is not None

    @property
    def has_edge_features(self) -> bool:
        """Return True if edge features exist."""
        return self.edge_attr is not None

    def summary(self) -> dict[str, Any]:
        """
        Return tensor summary.
        """
        return {
            "graph_id": self.graph_id,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_node_features": self.num_node_features,
            "has_labels": self.has_labels,
            "has_edge_features": self.has_edge_features,
        }

    def copy(self) -> "GraphTensor":
        """
        Deep copy tensor representation.
        """
        return GraphTensor(
            x=self.x.copy(),
            edge_index=self.edge_index.copy(),
            y=None if self.y is None else self.y.copy(),
            edge_attr=None if self.edge_attr is None else self.edge_attr.copy(),
            graph_id=self.graph_id,
            metadata=dict(self.metadata),
        )

    def __len__(self) -> int:
        return self.num_nodes

    def __repr__(self) -> str:
        return (
            f"GraphTensor("
            f"nodes={self.num_nodes}, "
            f"edges={self.num_edges}, "
            f"features={self.num_node_features}, "
            f"labels={self.has_labels})"
        )