import numpy as np
import pytest

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.graph.graph_tensor import GraphTensor
from src.graph.tensor_converter import cad_graph_to_tensor, tensor_summary


def test_cad_graph_to_tensor_basic():
    graph = CADGraph(graph_id="basic", metadata={"source": "unit_test"})

    graph.add_node(GraphNode(node_id=0, features=[1.0, 2.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[3.0, 4.0], label=1))

    graph.add_edge(GraphEdge(source=0, target=1, features=[1.0]))
    graph.add_edge(GraphEdge(source=1, target=0, features=[1.0]))

    tensor = cad_graph_to_tensor(graph)

    assert isinstance(tensor, GraphTensor)
    assert tensor.graph_id == "basic"
    assert tensor.metadata == {"source": "unit_test"}

    assert tensor.x.shape == (2, 2)
    assert tensor.edge_index.shape == (2, 2)
    assert tensor.y.shape == (2,)
    assert tensor.edge_attr.shape == (2, 1)

    np.testing.assert_array_equal(
        tensor.x,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        tensor.edge_index,
        np.array([[0, 1], [1, 0]], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        tensor.y,
        np.array([0, 1], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        tensor.edge_attr,
        np.array([[1.0], [1.0]], dtype=np.float32),
    )


def test_cad_graph_to_tensor_without_edge_attr():
    graph = CADGraph(graph_id="no_edge_attr")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=1))

    graph.add_edge(GraphEdge(source=0, target=1, features=[1.0]))

    tensor = cad_graph_to_tensor(graph, include_edge_attr=False)

    assert tensor.edge_attr is None


def test_cad_graph_to_tensor_missing_labels_returns_none_y():
    graph = CADGraph(graph_id="missing_labels")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=None))

    graph.add_edge(GraphEdge(source=0, target=1, features=[1.0]))

    tensor = cad_graph_to_tensor(graph)

    assert tensor.y is None


def test_cad_graph_to_tensor_empty_edges():
    graph = CADGraph(graph_id="empty_edges")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))

    tensor = cad_graph_to_tensor(graph)

    assert tensor.edge_index.shape == (2, 0)
    assert tensor.edge_attr is None


def test_cad_graph_to_tensor_rejects_none_graph():
    with pytest.raises(ValueError, match="CADGraph cannot be None"):
        cad_graph_to_tensor(None)


def test_cad_graph_to_tensor_validation_rejects_invalid_graph():
    graph = CADGraph(graph_id="invalid")

    graph.nodes.append(GraphNode(node_id=0, features=[1.0], label=0))
    graph.nodes.append(GraphNode(node_id=0, features=[2.0], label=1))

    with pytest.raises(ValueError, match="Invalid CADGraph"):
        cad_graph_to_tensor(graph, validate=True)


def test_cad_graph_to_tensor_without_validation_allows_structural_conversion():
    graph = CADGraph(graph_id="skip_validation")

    graph.nodes.append(GraphNode(node_id=0, features=[1.0], label=0))
    graph.nodes.append(GraphNode(node_id=0, features=[2.0], label=1))

    tensor = cad_graph_to_tensor(graph, validate=False)

    assert tensor.x.shape == (2, 1)
    assert tensor.edge_index.shape == (2, 0)
    assert tensor.y.shape == (2,)


def test_cad_graph_to_tensor_edge_attr_none_when_edge_features_missing():
    graph = CADGraph(graph_id="missing_edge_features")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=1))

    graph.add_edge(GraphEdge(source=0, target=1, features=[]))

    tensor = cad_graph_to_tensor(graph)

    assert tensor.edge_attr is None


def test_tensor_summary():
    tensor = GraphTensor(
        x=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        y=np.array([0, 1], dtype=np.int64),
        graph_id="summary_tensor",
    )

    summary = tensor_summary(tensor)

    assert summary == {
        "graph_id": "summary_tensor",
        "num_nodes": 2,
        "num_edges": 2,
        "num_node_features": 2,
        "has_labels": True,
        "has_edge_features": False,
    }


def test_tensor_summary_rejects_none():
    with pytest.raises(ValueError, match="GraphTensor cannot be None"):
        tensor_summary(None)