"""
CAD graph data structures for DeepMeshNet-v1.

This module defines a neutral engineering graph representation.
It does not depend on PyTorch Geometric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    """
    Graph node representing one CAD face.
    """

    node_id: int
    face_id: int | None = None
    features: list[float] = field(default_factory=list)
    label: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """
    Graph edge representing adjacency between two CAD faces.
    """

    source: int
    target: int
    features: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[int, int]:
        """Return edge as tuple."""
        return self.source, self.target


@dataclass
class CADGraph:
    """
    Neutral CAD graph representation.

    Nodes represent CAD faces.
    Edges represent face adjacency.
    """

    graph_id: str | None = None
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        """Add graph node."""
        if self.has_node(node.node_id):
            raise ValueError(f"Duplicate node ID: {node.node_id}")

        self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add graph edge."""
        if not self.has_node(edge.source):
            raise ValueError(f"Source node does not exist: {edge.source}")

        if not self.has_node(edge.target):
            raise ValueError(f"Target node does not exist: {edge.target}")

        self.edges.append(edge)

    def has_node(self, node_id: int) -> bool:
        """Return True if graph contains node ID."""
        return any(node.node_id == node_id for node in self.nodes)

    def get_node(self, node_id: int) -> GraphNode:
        """Return node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node

        raise KeyError(f"Node not found: {node_id}")

    def node_count(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Return number of edges."""
        return len(self.edges)

    def edge_index(self) -> list[tuple[int, int]]:
        """Return edge list as source-target tuples."""
        return [edge.as_tuple() for edge in self.edges]

    def node_feature_matrix(self) -> list[list[float]]:
        """Return node feature matrix."""
        return [node.features for node in self.nodes]

    def labels(self) -> list[int | None]:
        """Return node labels."""
        return [node.label for node in self.nodes]

    def summary(self) -> dict[str, Any]:
        """Return graph summary."""
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "has_node_features": all(len(node.features) > 0 for node in self.nodes),
            "has_labels": all(node.label is not None for node in self.nodes),
        }