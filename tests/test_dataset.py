import json

import pytest

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.learning.dataset import GraphDataset
from src.learning.graph_sample import GraphSample


def make_sample(sample_id: str):
    graph = CADGraph(
        graph_id=sample_id,
        metadata={"split": "train"},
    )

    graph.add_node(
        GraphNode(
            node_id=0,
            features=[1.0, 2.0],
            label=0,
        )
    )

    graph.add_node(
        GraphNode(
            node_id=1,
            features=[3.0, 4.0],
            label=1,
        )
    )

    graph.add_edge(
        GraphEdge(
            source=0,
            target=1,
            features=[1.0],
        )
    )

    return GraphSample(graph=graph)


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def test_dataset_initialization():
    dataset = GraphDataset()

    assert dataset.num_samples == 0
    assert dataset.total_nodes == 0
    assert dataset.total_edges == 0
    assert len(dataset) == 0


# ---------------------------------------------------------
# Add / Get
# ---------------------------------------------------------

def test_add_sample():
    dataset = GraphDataset()

    sample = make_sample("model_001")

    dataset.add_sample(sample)

    assert dataset.num_samples == 1
    assert dataset.has_sample("model_001")


def test_add_none_sample():
    dataset = GraphDataset()

    with pytest.raises(ValueError):
        dataset.add_sample(None)


def test_add_duplicate_sample():
    dataset = GraphDataset()

    sample = make_sample("model_001")

    dataset.add_sample(sample)

    with pytest.raises(ValueError):
        dataset.add_sample(sample)


def test_get_sample():
    dataset = GraphDataset()

    sample = make_sample("model_001")

    dataset.add_sample(sample)

    loaded = dataset.get_sample("model_001")

    assert loaded is sample


def test_get_missing_sample():
    dataset = GraphDataset()

    with pytest.raises(KeyError):
        dataset.get_sample("missing")


# ---------------------------------------------------------
# Remove
# ---------------------------------------------------------

def test_remove_sample():
    dataset = GraphDataset()

    sample = make_sample("model_001")

    dataset.add_sample(sample)

    dataset.remove_sample("model_001")

    assert dataset.num_samples == 0


def test_remove_missing_sample():
    dataset = GraphDataset()

    with pytest.raises(KeyError):
        dataset.remove_sample("missing")


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def test_dataset_statistics():
    dataset = GraphDataset()

    dataset.add_sample(make_sample("a"))
    dataset.add_sample(make_sample("b"))

    assert dataset.num_samples == 2
    assert dataset.total_nodes == 4
    assert dataset.total_edges == 2
    assert dataset.average_nodes == 2
    assert dataset.average_edges == 1


def test_dataset_summary():
    dataset = GraphDataset()

    dataset.add_sample(make_sample("a"))
    dataset.add_sample(make_sample("b"))

    summary = dataset.summary()

    assert summary == {
        "num_samples": 2,
        "total_nodes": 4,
        "total_edges": 2,
        "average_nodes": 2,
        "average_edges": 1,
    }


# ---------------------------------------------------------
# Sample IDs
# ---------------------------------------------------------

def test_sample_ids():
    dataset = GraphDataset()

    dataset.add_sample(make_sample("c"))
    dataset.add_sample(make_sample("a"))
    dataset.add_sample(make_sample("b"))

    assert dataset.sample_ids() == [
        "a",
        "b",
        "c",
    ]


# ---------------------------------------------------------
# Metadata Export
# ---------------------------------------------------------

def test_export_metadata():
    dataset = GraphDataset()

    dataset.add_sample(make_sample("sample"))

    metadata = dataset.export_metadata()

    assert len(metadata) == 1

    assert metadata[0]["sample_id"] == "sample"
    assert metadata[0]["graph_id"] == "sample"


def test_save_metadata_json(tmp_path):
    dataset = GraphDataset()

    dataset.add_sample(make_sample("sample"))

    output = tmp_path / "dataset_metadata.json"

    dataset.save_metadata_json(output)

    assert output.exists()

    with open(output, encoding="utf-8") as fp:
        data = json.load(fp)

    assert len(data) == 1

    assert data[0]["sample_id"] == "sample"


# ---------------------------------------------------------
# Python Protocols
# ---------------------------------------------------------

def test_dataset_iteration():
    dataset = GraphDataset()

    dataset.add_sample(make_sample("a"))
    dataset.add_sample(make_sample("b"))

    ids = [sample.sample_id for sample in dataset]

    assert sorted(ids) == ["a", "b"]


def test_dataset_getitem():
    dataset = GraphDataset()

    sample = make_sample("abc")

    dataset.add_sample(sample)

    assert dataset["abc"] is sample


def test_dataset_contains():
    dataset = GraphDataset()

    dataset.add_sample(make_sample("abc"))

    assert "abc" in dataset
    assert "xyz" not in dataset


def test_dataset_repr():
    dataset = GraphDataset()

    dataset.add_sample(make_sample("sample"))

    text = repr(dataset)

    assert "GraphDataset" in text
    assert "samples=1" in text
    assert "nodes=2" in text
    assert "edges=1" in text