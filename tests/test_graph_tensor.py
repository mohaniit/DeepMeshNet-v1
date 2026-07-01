import numpy as np

from src.graph.graph_tensor import GraphTensor


def test_graph_tensor_basic_properties():
    tensor = GraphTensor(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=np.array([[0, 1], [1, 0]]),
        y=np.array([0, 1]),
        graph_id="test_graph",
    )

    assert tensor.num_nodes == 2
    assert tensor.num_node_features == 2
    assert tensor.num_edges == 2
    assert tensor.has_labels is True
    assert tensor.has_edge_features is False
    assert len(tensor) == 2


def test_graph_tensor_with_edge_attr():
    tensor = GraphTensor(
        x=np.array([[1.0], [2.0]]),
        edge_index=np.array([[0, 1], [1, 0]]),
        y=np.array([0, 1]),
        edge_attr=np.array([[1.0], [1.0]]),
    )

    assert tensor.has_edge_features is True


def test_graph_tensor_without_labels():
    tensor = GraphTensor(
        x=np.array([[1.0], [2.0]]),
        edge_index=np.array([[0], [1]]),
        y=None,
    )

    assert tensor.has_labels is False


def test_graph_tensor_empty_edges():
    tensor = GraphTensor(
        x=np.array([[1.0], [2.0]]),
        edge_index=np.empty((2, 0), dtype=int),
    )

    assert tensor.num_edges == 0


def test_graph_tensor_one_dimensional_features():
    tensor = GraphTensor(
        x=np.array([1.0, 2.0, 3.0]),
        edge_index=np.empty((2, 0), dtype=int),
    )

    assert tensor.num_nodes == 3
    assert tensor.num_node_features == 1


def test_graph_tensor_summary():
    tensor = GraphTensor(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=np.array([[0, 1], [1, 0]]),
        y=np.array([0, 1]),
        edge_attr=np.array([[1.0], [1.0]]),
        graph_id="summary_graph",
        metadata={"source": "unit_test"},
    )

    summary = tensor.summary()

    assert summary == {
        "graph_id": "summary_graph",
        "num_nodes": 2,
        "num_edges": 2,
        "num_node_features": 2,
        "has_labels": True,
        "has_edge_features": True,
    }


def test_graph_tensor_copy_is_independent():
    tensor = GraphTensor(
        x=np.array([[1.0, 2.0]]),
        edge_index=np.array([[0], [0]]),
        y=np.array([1]),
        edge_attr=np.array([[1.0]]),
        graph_id="copy_graph",
        metadata={"a": 1},
    )

    copied = tensor.copy()

    assert copied is not tensor
    assert copied.graph_id == tensor.graph_id
    assert copied.metadata == tensor.metadata

    copied.x[0, 0] = 99.0
    copied.edge_index[0, 0] = 99
    copied.y[0] = 99
    copied.edge_attr[0, 0] = 99.0
    copied.metadata["a"] = 99

    assert tensor.x[0, 0] == 1.0
    assert tensor.edge_index[0, 0] == 0
    assert tensor.y[0] == 1
    assert tensor.edge_attr[0, 0] == 1.0
    assert tensor.metadata["a"] == 1


def test_graph_tensor_repr():
    tensor = GraphTensor(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=np.array([[0, 1], [1, 0]]),
        y=np.array([0, 1]),
    )

    text = repr(tensor)

    assert "GraphTensor" in text
    assert "nodes=2" in text
    assert "edges=2" in text
    assert "features=2" in text
    assert "labels=True" in text