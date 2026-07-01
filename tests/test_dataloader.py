import pytest
from torch_geometric.loader import DataLoader

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.learning.dataloader import (
    dataloader_summary,
    make_dataloader_from_graph_dataset,
    make_graph_dataloader,
)
from src.learning.dataset import GraphDataset
from src.learning.graph_sample import GraphSample
from src.learning.torch_dataset import TorchGraphDataset


def make_sample(sample_id: str) -> GraphSample:
    graph = CADGraph(graph_id=sample_id)

    graph.add_node(GraphNode(node_id=0, features=[1.0, 2.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[3.0, 4.0], label=1))

    graph.add_edge(GraphEdge(source=0, target=1, features=[1.0]))
    graph.add_edge(GraphEdge(source=1, target=0, features=[1.0]))

    return GraphSample(graph=graph)


def make_dataset(size: int = 4) -> GraphDataset:
    dataset = GraphDataset()

    for index in range(size):
        dataset.add_sample(make_sample(f"model_{index:03d}"))

    return dataset


def test_make_graph_dataloader():
    dataset = make_dataset(4)
    torch_dataset = TorchGraphDataset(dataset)

    loader = make_graph_dataloader(
        dataset=torch_dataset,
        batch_size=2,
        shuffle=False,
    )

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 2
    assert len(loader) == 2


def test_make_graph_dataloader_batch_iteration():
    dataset = make_dataset(4)
    torch_dataset = TorchGraphDataset(dataset)

    loader = make_graph_dataloader(
        dataset=torch_dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch.x.shape[1] == 2
    assert batch.edge_index.shape[0] == 2
    assert batch.y is not None


def test_make_graph_dataloader_rejects_none_dataset():
    with pytest.raises(ValueError, match="TorchGraphDataset cannot be None"):
        make_graph_dataloader(None)


def test_make_graph_dataloader_rejects_bad_batch_size():
    dataset = make_dataset(1)
    torch_dataset = TorchGraphDataset(dataset)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        make_graph_dataloader(torch_dataset, batch_size=0)


def test_make_graph_dataloader_rejects_negative_workers():
    dataset = make_dataset(1)
    torch_dataset = TorchGraphDataset(dataset)

    with pytest.raises(ValueError, match="num_workers cannot be negative"):
        make_graph_dataloader(torch_dataset, num_workers=-1)


def test_make_dataloader_from_graph_dataset():
    dataset = make_dataset(3)

    loader = make_dataloader_from_graph_dataset(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 2
    assert len(loader) == 2


def test_make_dataloader_from_graph_dataset_without_edge_attr():
    dataset = make_dataset(2)

    loader = make_dataloader_from_graph_dataset(
        dataset=dataset,
        batch_size=1,
        include_edge_attr=False,
    )

    batch = next(iter(loader))

    assert not hasattr(batch, "edge_attr") or batch.edge_attr is None


def test_make_dataloader_from_graph_dataset_rejects_none_dataset():
    with pytest.raises(ValueError, match="GraphDataset cannot be None"):
        make_dataloader_from_graph_dataset(None)


def test_dataloader_summary():
    dataset = make_dataset(4)
    torch_dataset = TorchGraphDataset(dataset)

    loader = make_graph_dataloader(
        dataset=torch_dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    summary = dataloader_summary(loader)

    assert summary == {
        "num_samples": 4,
        "batch_size": 2,
        "num_batches": 2,
        "drop_last": False,
    }


def test_dataloader_summary_drop_last():
    dataset = make_dataset(5)
    torch_dataset = TorchGraphDataset(dataset)

    loader = make_graph_dataloader(
        dataset=torch_dataset,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )

    summary = dataloader_summary(loader)

    assert summary == {
        "num_samples": 5,
        "batch_size": 2,
        "num_batches": 2,
        "drop_last": True,
    }


def test_dataloader_summary_rejects_none_loader():
    with pytest.raises(ValueError, match="DataLoader cannot be None"):
        dataloader_summary(None)