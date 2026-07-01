import pytest
from torch_geometric.data import Data

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.learning.dataset import GraphDataset
from src.learning.graph_sample import GraphSample
from src.learning.torch_dataset import TorchGraphDataset, make_torch_dataset


def make_sample(sample_id: str) -> GraphSample:
    graph = CADGraph(graph_id=sample_id)

    graph.add_node(GraphNode(node_id=0, features=[1.0, 2.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[3.0, 4.0], label=1))

    graph.add_edge(GraphEdge(source=0, target=1, features=[1.0]))
    graph.add_edge(GraphEdge(source=1, target=0, features=[1.0]))

    return GraphSample(graph=graph)


def make_dataset(size: int = 3) -> GraphDataset:
    dataset = GraphDataset()

    for index in range(size):
        dataset.add_sample(make_sample(f"model_{index:03d}"))

    return dataset


def test_torch_graph_dataset_initialization():
    dataset = make_dataset(3)

    torch_dataset = TorchGraphDataset(dataset)

    assert len(torch_dataset) == 3
    assert torch_dataset.include_edge_attr is True
    assert torch_dataset.validate is True
    assert torch_dataset.sample_ids == [
        "model_000",
        "model_001",
        "model_002",
    ]


def test_torch_graph_dataset_rejects_none_dataset():
    with pytest.raises(ValueError, match="GraphDataset cannot be None"):
        TorchGraphDataset(None)


def test_torch_graph_dataset_getitem_returns_pyg_data():
    dataset = make_dataset(1)

    torch_dataset = TorchGraphDataset(dataset)

    data = torch_dataset[0]

    assert isinstance(data, Data)
    assert data.graph_id == "model_000"
    assert data.x.shape == (2, 2)
    assert data.edge_index.shape == (2, 2)
    assert data.y.shape == (2,)
    assert data.edge_attr.shape == (2, 1)


def test_torch_graph_dataset_getitem_without_edge_attr():
    dataset = make_dataset(1)

    torch_dataset = TorchGraphDataset(
        dataset,
        include_edge_attr=False,
    )

    data = torch_dataset[0]

    assert isinstance(data, Data)
    assert not hasattr(data, "edge_attr") or data.edge_attr is None


def test_torch_graph_dataset_getitem_out_of_range():
    dataset = make_dataset(1)

    torch_dataset = TorchGraphDataset(dataset)

    with pytest.raises(IndexError):
        _ = torch_dataset[1]


def test_torch_graph_dataset_getitem_negative_index():
    dataset = make_dataset(1)

    torch_dataset = TorchGraphDataset(dataset)

    with pytest.raises(IndexError):
        _ = torch_dataset[-1]


def test_torch_graph_dataset_get_sample():
    dataset = make_dataset(2)

    torch_dataset = TorchGraphDataset(dataset)

    sample = torch_dataset.get_sample(1)

    assert isinstance(sample, GraphSample)
    assert sample.sample_id == "model_001"


def test_torch_graph_dataset_get_sample_out_of_range():
    dataset = make_dataset(1)

    torch_dataset = TorchGraphDataset(dataset)

    with pytest.raises(IndexError):
        torch_dataset.get_sample(1)


def test_torch_graph_dataset_get_sample_id():
    dataset = make_dataset(2)

    torch_dataset = TorchGraphDataset(dataset)

    assert torch_dataset.get_sample_id(0) == "model_000"
    assert torch_dataset.get_sample_id(1) == "model_001"


def test_torch_graph_dataset_get_sample_id_out_of_range():
    dataset = make_dataset(1)

    torch_dataset = TorchGraphDataset(dataset)

    with pytest.raises(IndexError):
        torch_dataset.get_sample_id(1)


def test_torch_graph_dataset_summary():
    dataset = make_dataset(2)

    torch_dataset = TorchGraphDataset(
        dataset,
        include_edge_attr=False,
        validate=False,
    )

    summary = torch_dataset.summary()

    assert summary == {
        "num_samples": 2,
        "include_edge_attr": False,
        "validate": False,
        "sample_ids": [
            "model_000",
            "model_001",
        ],
    }


def test_torch_graph_dataset_repr():
    dataset = make_dataset(2)

    torch_dataset = TorchGraphDataset(dataset)

    text = repr(torch_dataset)

    assert "TorchGraphDataset" in text
    assert "samples=2" in text
    assert "include_edge_attr=True" in text
    assert "validate=True" in text


def test_make_torch_dataset_factory():
    dataset = make_dataset(1)

    torch_dataset = make_torch_dataset(
        dataset,
        include_edge_attr=False,
        validate=False,
    )

    assert isinstance(torch_dataset, TorchGraphDataset)
    assert len(torch_dataset) == 1
    assert torch_dataset.include_edge_attr is False
    assert torch_dataset.validate is False