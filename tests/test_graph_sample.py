from torch_geometric.data import Data
import pytest

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.graph.graph_tensor import GraphTensor
from src.learning.graph_sample import GraphSample


def make_sample_graph():
    graph = CADGraph(graph_id="sample_graph", metadata={"source": "unit_test"})

    graph.add_node(GraphNode(node_id=0, features=[1.0, 2.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[3.0, 4.0], label=1))

    graph.add_edge(GraphEdge(source=0, target=1, features=[1.0]))
    graph.add_edge(GraphEdge(source=1, target=0, features=[1.0]))

    return graph


def test_graph_sample_initialization():
    graph = make_sample_graph()

    sample = GraphSample(graph=graph)

    assert sample.graph is graph
    assert sample.sample_id == "sample_graph"
    assert sample.graph_id == "sample_graph"
    assert sample.metadata == {"source": "unit_test"}
    assert sample.num_nodes == 2
    assert sample.num_edges == 2
    assert sample.has_tensor is False
    assert sample.has_pyg_data is False


def test_graph_sample_custom_sample_id_and_metadata():
    graph = make_sample_graph()

    sample = GraphSample(
        graph=graph,
        sample_id="custom_id",
        metadata={"split": "train"},
    )

    assert sample.sample_id == "custom_id"
    assert sample.metadata == {"split": "train"}


def test_graph_sample_rejects_none_graph():
    with pytest.raises(ValueError, match="CADGraph cannot be None"):
        GraphSample(graph=None)


def test_graph_sample_to_tensor_lazy_conversion():
    sample = GraphSample(graph=make_sample_graph())

    assert sample.has_tensor is False

    tensor = sample.to_tensor()

    assert isinstance(tensor, GraphTensor)
    assert sample.has_tensor is True
    assert tensor.num_nodes == 2
    assert tensor.num_edges == 2
    assert tensor.graph_id == "sample_graph"


def test_graph_sample_to_tensor_cache_reuse():
    sample = GraphSample(graph=make_sample_graph())

    tensor_1 = sample.to_tensor()
    tensor_2 = sample.to_tensor()

    assert tensor_1 is tensor_2


def test_graph_sample_to_tensor_refresh():
    sample = GraphSample(graph=make_sample_graph())

    tensor_1 = sample.to_tensor()
    tensor_2 = sample.to_tensor(refresh=True)

    assert tensor_1 is not tensor_2


def test_graph_sample_to_pyg_data_lazy_conversion():
    sample = GraphSample(graph=make_sample_graph())

    assert sample.has_pyg_data is False

    data = sample.to_pyg_data()

    assert isinstance(data, Data)
    assert sample.has_tensor is True
    assert sample.has_pyg_data is True
    assert data.graph_id == "sample_graph"
    assert data.x.shape == (2, 2)
    assert data.edge_index.shape == (2, 2)
    assert data.y.shape == (2,)


def test_graph_sample_to_pyg_data_cache_reuse():
    sample = GraphSample(graph=make_sample_graph())

    data_1 = sample.to_pyg_data()
    data_2 = sample.to_pyg_data()

    assert data_1 is data_2


def test_graph_sample_to_pyg_data_refresh():
    sample = GraphSample(graph=make_sample_graph())

    data_1 = sample.to_pyg_data()
    data_2 = sample.to_pyg_data(refresh=True)

    assert data_1 is not data_2


def test_graph_sample_clear_cache():
    sample = GraphSample(graph=make_sample_graph())

    sample.to_tensor()
    sample.to_pyg_data()

    assert sample.has_tensor is True
    assert sample.has_pyg_data is True

    sample.clear_cache()

    assert sample.has_tensor is False
    assert sample.has_pyg_data is False


def test_graph_sample_summary():
    sample = GraphSample(graph=make_sample_graph())

    summary = sample.summary()

    assert summary == {
        "sample_id": "sample_graph",
        "graph_id": "sample_graph",
        "num_nodes": 2,
        "num_edges": 2,
        "has_tensor": False,
        "has_pyg_data": False,
        "metadata": {"source": "unit_test"},
    }


def test_graph_sample_summary_after_conversion():
    sample = GraphSample(graph=make_sample_graph())

    sample.to_tensor()
    sample.to_pyg_data()

    summary = sample.summary()

    assert summary["has_tensor"] is True
    assert summary["has_pyg_data"] is True


def test_graph_sample_len():
    sample = GraphSample(graph=make_sample_graph())

    assert len(sample) == 2


def test_graph_sample_repr():
    sample = GraphSample(graph=make_sample_graph())

    text = repr(sample)

    assert "GraphSample" in text
    assert "sample_graph" in text
    assert "nodes=2" in text
    assert "edges=2" in text