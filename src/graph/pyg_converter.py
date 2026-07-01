"""
PyTorch Geometric converter for DeepMeshNet-v1.

This module converts the framework-independent GraphTensor representation
into a PyTorch Geometric Data object.

Pipeline
--------
CADGraph
    ↓
GraphTensor
    ↓
torch_geometric.data.Data

This module is intentionally isolated from CADModel, OpenCascade, EMKE,
and engineering logic.
"""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data

from src.graph.graph_tensor import GraphTensor
from src.graph.tensor_converter import cad_graph_to_tensor
from src.graph.graph import CADGraph


def graph_tensor_to_pyg_data(
    tensor: GraphTensor,
    include_edge_attr: bool = True,
) -> Data:
    """
    Convert GraphTensor to PyTorch Geometric Data.

    Parameters
    ----------
    tensor:
        Framework-independent graph tensor.
    include_edge_attr:
        If True, include edge attributes when available.

    Returns
    -------
    torch_geometric.data.Data
    """
    if tensor is None:
        raise ValueError("GraphTensor cannot be None.")

    x = torch.tensor(tensor.x, dtype=torch.float32)
    edge_index = torch.tensor(tensor.edge_index, dtype=torch.long)

    data_kwargs: dict[str, Any] = {
        "x": x,
        "edge_index": edge_index,
    }

    if tensor.y is not None:
        data_kwargs["y"] = torch.tensor(tensor.y, dtype=torch.long)

    if include_edge_attr and tensor.edge_attr is not None:
        data_kwargs["edge_attr"] = torch.tensor(
            tensor.edge_attr,
            dtype=torch.float32,
        )

    data = Data(**data_kwargs)

    data.graph_id = tensor.graph_id
    data.metadata = dict(tensor.metadata)

    return data


def cad_graph_to_pyg_data(
    graph: CADGraph,
    validate: bool = True,
    include_edge_attr: bool = True,
) -> Data:
    """
    Convert CADGraph directly to PyTorch Geometric Data.

    Internally:
        CADGraph → GraphTensor → PyG Data
    """
    tensor = cad_graph_to_tensor(
        graph=graph,
        validate=validate,
        include_edge_attr=include_edge_attr,
    )

    return graph_tensor_to_pyg_data(
        tensor=tensor,
        include_edge_attr=include_edge_attr,
    )


def pyg_data_summary(data: Data) -> dict[str, Any]:
    """
    Return summary of PyTorch Geometric Data object.
    """
    if data is None:
        raise ValueError("PyG Data cannot be None.")

    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_node_features": int(data.num_node_features),
        "has_labels": hasattr(data, "y") and data.y is not None,
        "has_edge_attr": hasattr(data, "edge_attr") and data.edge_attr is not None,
        "graph_id": getattr(data, "graph_id", None),
    }


__all__ = [
    "graph_tensor_to_pyg_data",
    "cad_graph_to_pyg_data",
    "pyg_data_summary",
]