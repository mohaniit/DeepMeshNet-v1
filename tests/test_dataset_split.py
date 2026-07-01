import pytest

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.learning.dataset import GraphDataset
from src.learning.dataset_split import (
    DatasetSplit,
    fixed_split_dataset,
    make_dataset_from_samples,
    sample_ids_from_dataset,
    split_dataset,
    split_dataset_by_counts,
    train_test_split,
)
from src.learning.graph_sample import GraphSample


def make_sample(sample_id: str) -> GraphSample:
    graph = CADGraph(graph_id=sample_id)

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=1))
    graph.add_edge(GraphEdge(source=0, target=1, features=[1.0]))

    return GraphSample(graph=graph)


def make_dataset(size: int = 10) -> GraphDataset:
    dataset = GraphDataset()

    for index in range(size):
        dataset.add_sample(make_sample(f"model_{index:03d}"))

    return dataset


def test_dataset_split_summary():
    split = DatasetSplit(
        train=make_dataset(6),
        validation=make_dataset(2),
        test=make_dataset(2),
    )

    assert split.summary() == {
        "train": 6,
        "validation": 2,
        "test": 2,
        "total": 10,
    }


def test_split_dataset_default_ratios():
    dataset = make_dataset(10)

    split = split_dataset(dataset, seed=42)

    assert len(split.train) == 7
    assert len(split.validation) == 1
    assert len(split.test) == 2
    assert split.summary()["total"] == 10


def test_split_dataset_is_reproducible():
    dataset = make_dataset(10)

    split_1 = split_dataset(dataset, seed=123)
    split_2 = split_dataset(dataset, seed=123)

    assert split_1.train.sample_ids() == split_2.train.sample_ids()
    assert split_1.validation.sample_ids() == split_2.validation.sample_ids()
    assert split_1.test.sample_ids() == split_2.test.sample_ids()


def test_split_dataset_different_seed_changes_split():
    dataset = make_dataset(20)

    split_1 = split_dataset(dataset, seed=1)
    split_2 = split_dataset(dataset, seed=2)

    assert split_1.train.sample_ids() != split_2.train.sample_ids()


def test_split_dataset_custom_ratios():
    dataset = make_dataset(20)

    split = split_dataset(
        dataset,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
        seed=42,
    )

    assert len(split.train) == 12
    assert len(split.validation) == 4
    assert len(split.test) == 4


def test_split_dataset_rejects_none_dataset():
    with pytest.raises(ValueError, match="GraphDataset cannot be None"):
        split_dataset(None)


def test_split_dataset_rejects_empty_dataset():
    with pytest.raises(ValueError, match="GraphDataset cannot be empty"):
        split_dataset(GraphDataset())


def test_split_dataset_rejects_negative_ratio():
    dataset = make_dataset(10)

    with pytest.raises(ValueError, match="negative"):
        split_dataset(
            dataset,
            train_ratio=-0.1,
            validation_ratio=0.6,
            test_ratio=0.5,
        )


def test_split_dataset_rejects_bad_ratio_sum():
    dataset = make_dataset(10)

    with pytest.raises(ValueError, match="sum to 1.0"):
        split_dataset(
            dataset,
            train_ratio=0.5,
            validation_ratio=0.3,
            test_ratio=0.3,
        )


def test_split_dataset_by_counts():
    dataset = make_dataset(10)

    split = split_dataset_by_counts(
        dataset,
        train_count=6,
        validation_count=2,
        test_count=2,
        seed=42,
    )

    assert len(split.train) == 6
    assert len(split.validation) == 2
    assert len(split.test) == 2
    assert split.summary()["total"] == 10


def test_split_dataset_by_counts_rejects_bad_total():
    dataset = make_dataset(10)

    with pytest.raises(ValueError, match="sum to dataset size"):
        split_dataset_by_counts(
            dataset,
            train_count=6,
            validation_count=2,
            test_count=1,
        )


def test_split_dataset_by_counts_rejects_negative_count():
    dataset = make_dataset(10)

    with pytest.raises(ValueError, match="cannot be negative"):
        split_dataset_by_counts(
            dataset,
            train_count=6,
            validation_count=-1,
            test_count=5,
        )


def test_fixed_split_dataset():
    dataset = make_dataset(5)

    split = fixed_split_dataset(
        dataset,
        train_ids=["model_000", "model_001", "model_002"],
        validation_ids=["model_003"],
        test_ids=["model_004"],
    )

    assert split.train.sample_ids() == [
        "model_000",
        "model_001",
        "model_002",
    ]
    assert split.validation.sample_ids() == ["model_003"]
    assert split.test.sample_ids() == ["model_004"]


def test_fixed_split_dataset_allows_empty_validation_and_test():
    dataset = make_dataset(3)

    split = fixed_split_dataset(
        dataset,
        train_ids=["model_000", "model_001", "model_002"],
    )

    assert len(split.train) == 3
    assert len(split.validation) == 0
    assert len(split.test) == 0


def test_fixed_split_dataset_rejects_duplicate_ids():
    dataset = make_dataset(3)

    with pytest.raises(ValueError, match="unique"):
        fixed_split_dataset(
            dataset,
            train_ids=["model_000", "model_000"],
        )


def test_fixed_split_dataset_rejects_missing_ids():
    dataset = make_dataset(3)

    with pytest.raises(KeyError, match="not found"):
        fixed_split_dataset(
            dataset,
            train_ids=["missing"],
        )


def test_train_test_split():
    dataset = make_dataset(10)

    train, test = train_test_split(
        dataset,
        train_ratio=0.8,
        seed=42,
    )

    assert len(train) == 8
    assert len(test) == 2


def test_train_test_split_rejects_bad_ratio():
    dataset = make_dataset(10)

    with pytest.raises(ValueError, match="between 0 and 1"):
        train_test_split(dataset, train_ratio=1.0)


def test_sample_ids_from_dataset():
    dataset = make_dataset(3)

    assert sample_ids_from_dataset(dataset) == [
        "model_000",
        "model_001",
        "model_002",
    ]


def test_make_dataset_from_samples():
    samples = [
        make_sample("a"),
        make_sample("b"),
    ]

    dataset = make_dataset_from_samples(samples)

    assert len(dataset) == 2
    assert dataset.sample_ids() == ["a", "b"]


def test_splits_share_original_sample_objects():
    dataset = make_dataset(5)

    split = fixed_split_dataset(
        dataset,
        train_ids=["model_000"],
        validation_ids=["model_001"],
        test_ids=["model_002"],
    )

    assert split.train.get_sample("model_000") is dataset.get_sample("model_000")
    assert split.validation.get_sample("model_001") is dataset.get_sample("model_001")
    assert split.test.get_sample("model_002") is dataset.get_sample("model_002")