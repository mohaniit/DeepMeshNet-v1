"""
Graph sample representation for DeepMeshNet-v1 Learning Engine.

A GraphSample represents one CAD model as a learning-ready sample.

Pipeline
--------
CADGraph
    ↓
GraphTensor
    ↓
PyTorch Geometric Data

Conversions are lazy. GraphTensor and PyG Data are created only when
requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.graph.graph import CADGraph
from src.graph.graph_tensor import GraphTensor
from src.graph.tensor_converter import cad_graph_to_tensor
from src.graph.pyg_converter import graph_tensor_to_pyg_data


@dataclass
class GraphSample:
    """
    One graph-level learning sample.
    """

    graph: CADGraph
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    _tensor: GraphTensor | None = field(default=None, init=False, repr=False)
    _pyg_data: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.graph is None:
            raise ValueError("CADGraph cannot be None.")

        if self.sample_id is None:
            self.sample_id = self.graph.graph_id or "graph_sample"

        if not self.metadata:
            self.metadata = dict(self.graph.metadata)

    @property
    def graph_id(self) -> str | None:
        return self.graph.graph_id

    @property
    def num_nodes(self) -> int:
        return self.graph.node_count()

    @property
    def num_edges(self) -> int:
        return self.graph.edge_count()

    @property
    def has_tensor(self) -> bool:
        return self._tensor is not None

    @property
    def has_pyg_data(self) -> bool:
        return self._pyg_data is not None

    def to_tensor(
        self,
        validate: bool = True,
        include_edge_attr: bool = True,
        refresh: bool = False,
    ) -> GraphTensor:
        """
        Convert sample graph to GraphTensor lazily.
        """
        if self._tensor is None or refresh:
            self._tensor = cad_graph_to_tensor(
                graph=self.graph,
                validate=validate,
                include_edge_attr=include_edge_attr,
            )

        return self._tensor

    def to_pyg_data(
        self,
        validate: bool = True,
        include_edge_attr: bool = True,
        refresh: bool = False,
    ) -> Any:
        """
        Convert sample graph to PyTorch Geometric Data lazily.
        """
        if self._pyg_data is None or refresh:
            tensor = self.to_tensor(
                validate=validate,
                include_edge_attr=include_edge_attr,
                refresh=refresh,
            )

            self._pyg_data = graph_tensor_to_pyg_data(
                tensor=tensor,
                include_edge_attr=include_edge_attr,
            )

        return self._pyg_data

    def clear_cache(self) -> None:
        """
        Clear cached tensor and PyG representations.
        """
        self._tensor = None
        self._pyg_data = None

    def summary(self) -> dict[str, Any]:
        """
        Return sample summary.
        """
        return {
            "sample_id": self.sample_id,
            "graph_id": self.graph_id,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "has_tensor": self.has_tensor,
            "has_pyg_data": self.has_pyg_data,
            "metadata": dict(self.metadata),
        }

    def __len__(self) -> int:
        return self.num_nodes

    def __repr__(self) -> str:
        return (
            f"GraphSample("
            f"sample_id={self.sample_id!r}, "
            f"nodes={self.num_nodes}, "
            f"edges={self.num_edges})"
        )


__all__ = [
    "GraphSample",
]