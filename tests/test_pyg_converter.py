import numpy as np
import pytest
import torch

from torch_geometric.data import Data

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.graph.graph_tensor import GraphTensor
from src.graph.pyg_converter import (
    cad_graph_to_pyg_data,
    graph_tensor_to_pyg_data,
    pyg_data_summary,
)


def test_graph_tensor_to_pyg_data_basic():
    tensor = GraphTensor(
        x=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        y=np.array([0, 1], dtype=np.int64),
        edge_attr=np.array([[1.0], [1.0]], dtype=np.float32),
        graph_id="pyg_basic",
        metadata={"source": "unit_test"},
    )

    data = graph_tensor_to_pyg_data(tensor)

    assert isinstance(data, Data)
    assert data.graph_id == "pyg_basic"
    assert data.metadata == {"source": "unit_test"}

    assert torch.equal(
        data.x,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )

    assert torch.equal(
        data.edge_index,
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )

    assert torch.equal(
        data.y,
        torch.tensor([0, 1], dtype=torch.long),
    )

    assert torch.equal(
        data.edge_attr,
        torch.tensor([[1.0], [1.0]], dtype=torch.float32),
    )


def test_graph_tensor_to_pyg_data_without_labels():
    tensor = GraphTensor(
        x=np.array([[1.0], [2.0]], dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        y=None,
        graph_id="no_labels",
    )

    data = graph_tensor_to_pyg_data(tensor)

    assert isinstance(data, Data)
    assert not hasattr(data, "y") or data.y is None


def test_graph_tensor_to_pyg_data_without_edge_attr():
    tensor = GraphTensor(
        x=np.array([[1.0], [2.0]], dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        y=np.array([0, 1], dtype=np.int64),
        edge_attr=np.array([[1.0]], dtype=np.float32),
    )

    data = graph_tensor_to_pyg_data(
        tensor,
        include_edge_attr=False,
    )

    assert not hasattr(data, "edge_attr") or data.edge_attr is None


def test_graph_tensor_to_pyg_data_rejects_none_tensor():
    with pytest.raises(ValueError, match="GraphTensor cannot be None"):
        graph_tensor_to_pyg_data(None)


def test_cad_graph_to_pyg_data_basic():
    graph = CADGraph(graph_id="cad_to_pyg", metadata={"source": "cad_graph"})

    graph.add_node(GraphNode(node_id=0, features=[1.0, 2.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[3.0, 4.0], label=1))

    graph.add_edge(GraphEdge(source=0, target=1, features=[1.0]))
    graph.add_edge(GraphEdge(source=1, target=0, features=[1.0]))

    data = cad_graph_to_pyg_data(graph)

    assert isinstance(data, Data)
    assert data.graph_id == "cad_to_pyg"
    assert data.metadata == {"source": "cad_graph"}

    assert data.x.shape == (2, 2)
    assert data.edge_index.shape == (2, 2)
    assert data.y.shape == (2,)
    assert data.edge_attr.shape == (2, 1)


def test_cad_graph_to_pyg_data_without_edge_attr():
    graph = CADGraph(graph_id="cad_no_edge_attr")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=1))
    graph.add_edge(GraphEdge(source=0, target=1, features=[1.0]))

    data = cad_graph_to_pyg_data(
        graph,
        include_edge_attr=False,
    )

    assert not hasattr(data, "edge_attr") or data.edge_attr is None


def test_cad_graph_to_pyg_data_invalid_graph_raises():
    graph = CADGraph(graph_id="invalid_graph")

    graph.nodes.append(GraphNode(node_id=0, features=[1.0], label=0))
    graph.nodes.append(GraphNode(node_id=0, features=[2.0], label=1))

    with pytest.raises(ValueError, match="Invalid CADGraph"):
        cad_graph_to_pyg_data(graph, validate=True)


def test_pyg_data_summary():
    tensor = GraphTensor(
        x=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        y=np.array([0, 1], dtype=np.int64),
        edge_attr=np.array([[1.0], [1.0]], dtype=np.float32),
        graph_id="summary_graph",
    )

    data = graph_tensor_to_pyg_data(tensor)

    summary = pyg_data_summary(data)

    assert summary == {
        "num_nodes": 2,
        "num_edges": 2,
        "num_node_features": 2,
        "has_labels": True,
        "has_edge_attr": True,
        "graph_id": "summary_graph",
    }


def test_pyg_data_summary_without_labels_or_edge_attr():
    tensor = GraphTensor(
        x=np.array([[1.0], [2.0]], dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        y=None,
        edge_attr=None,
        graph_id="summary_no_labels",
    )

    data = graph_tensor_to_pyg_data(tensor)

    summary = pyg_data_summary(data)

    assert summary["num_nodes"] == 2
    assert summary["num_edges"] == 1
    assert summary["num_node_features"] == 1
    assert summary["has_labels"] is False
    assert summary["has_edge_attr"] is False
    assert summary["graph_id"] == "summary_no_labels"


def test_pyg_data_summary_rejects_none():
    with pytest.raises(ValueError, match="PyG Data cannot be None"):
        pyg_data_summary(None)