"""
CADGraph to GraphTensor converter for DeepMeshNet-v1.

This module converts the neutral object-oriented CADGraph representation
into a framework-independent numerical GraphTensor representation.

It does not depend on PyTorch or PyTorch Geometric.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.graph.graph import CADGraph
from src.graph.graph_tensor import GraphTensor
from src.graph.graph_validator import validate_graph


def cad_graph_to_tensor(
    graph: CADGraph,
    validate: bool = True,
    include_edge_attr: bool = True,
) -> GraphTensor:
    """
    Convert CADGraph to GraphTensor.

    Parameters
    ----------
    graph:
        CADGraph object.
    validate:
        If True, validate graph before conversion.
    include_edge_attr:
        If True, include edge features when available.

    Returns
    -------
    GraphTensor
        Framework-independent numerical graph tensor.
    """
    if graph is None:
        raise ValueError("CADGraph cannot be None.")

    if validate:
        result = validate_graph(graph)
        if not result.is_valid:
            raise ValueError(
                "Invalid CADGraph: " + "; ".join(result.errors)
            )

    x = _build_node_feature_matrix(graph)
    edge_index = _build_edge_index(graph)
    y = _build_labels(graph)
    edge_attr = (
        _build_edge_attr(graph)
        if include_edge_attr and graph.edge_count() > 0
        else None
    )

    return GraphTensor(
        x=x,
        edge_index=edge_index,
        y=y,
        edge_attr=edge_attr,
        graph_id=graph.graph_id,
        metadata=dict(graph.metadata),
    )


def _build_node_feature_matrix(graph: CADGraph) -> np.ndarray:
    """
    Build node feature matrix.
    """
    if graph.node_count() == 0:
        raise ValueError("CADGraph contains no nodes.")

    features = [node.features for node in graph.nodes]

    return np.asarray(features, dtype=np.float32)


def _build_edge_index(graph: CADGraph) -> np.ndarray:
    """
    Build edge index array with shape [2, num_edges].
    """
    if graph.edge_count() == 0:
        return np.empty((2, 0), dtype=np.int64)

    edges = [[edge.source, edge.target] for edge in graph.edges]

    return np.asarray(edges, dtype=np.int64).T


def _build_labels(graph: CADGraph) -> np.ndarray | None:
    """
    Build node label vector.

    Returns None if any label is missing.
    """
    labels: list[int] = []

    for node in graph.nodes:
        if node.label is None:
            return None

        labels.append(int(node.label))

    return np.asarray(labels, dtype=np.int64)


def _build_edge_attr(graph: CADGraph) -> np.ndarray | None:
    """
    Build edge feature matrix.

    Returns None if any edge has missing/empty features.
    """
    if graph.edge_count() == 0:
        return None

    edge_features: list[list[float]] = []

    for edge in graph.edges:
        if not edge.features:
            return None

        edge_features.append([float(value) for value in edge.features])

    return np.asarray(edge_features, dtype=np.float32)


def tensor_summary(tensor: GraphTensor) -> dict[str, Any]:
    """
    Return GraphTensor summary.
    """
    if tensor is None:
        raise ValueError("GraphTensor cannot be None.")

    return tensor.summary()


__all__ = [
    "cad_graph_to_tensor",
    "tensor_summary",
]