import pytest
import torch

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.learning.dataloader import make_dataloader_from_graph_dataset
from src.learning.dataset import GraphDataset
from src.learning.graph_sample import GraphSample
from src.learning.predictor import PredictionResult, Predictor, predict


class DummyNodeClassifier(torch.nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.linear(x)


def make_sample(sample_id: str) -> GraphSample:
    graph = CADGraph(graph_id=sample_id)

    graph.add_node(GraphNode(node_id=0, features=[1.0, 0.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[0.0, 1.0], label=1))

    graph.add_edge(GraphEdge(source=0, target=1, features=[1.0]))
    graph.add_edge(GraphEdge(source=1, target=0, features=[1.0]))

    return GraphSample(graph=graph)


def make_loader(size: int = 3, batch_size: int = 2):
    dataset = GraphDataset()

    for index in range(size):
        dataset.add_sample(make_sample(f"model_{index:03d}"))

    return make_dataloader_from_graph_dataset(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def make_model():
    model = DummyNodeClassifier()

    with torch.no_grad():
        model.linear.weight[:] = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        model.linear.bias[:] = torch.tensor([0.0, 0.0])

    return model


def test_prediction_result_to_dict():
    result = PredictionResult(
        y_pred=[0, 1],
        y_true=[0, 1],
        probabilities=[[0.8, 0.2], [0.1, 0.9]],
        graph_ids=["g1"],
        metadata={"a": 1},
    )

    assert result.to_dict() == {
        "y_pred": [0, 1],
        "y_true": [0, 1],
        "probabilities": [[0.8, 0.2], [0.1, 0.9]],
        "graph_ids": ["g1"],
        "metadata": {"a": 1},
    }


def test_predictor_initialization():
    model = make_model()

    predictor = Predictor(model)

    assert predictor.model is model
    assert predictor.device.type == "cpu"
    assert predictor.model.training is False


def test_predictor_rejects_none_model():
    with pytest.raises(ValueError, match="Model cannot be None"):
        Predictor(None)


def test_predict_loader():
    model = make_model()
    loader = make_loader(size=2, batch_size=1)

    predictor = Predictor(model)

    result = predictor.predict_loader(loader)

    assert isinstance(result, PredictionResult)
    assert result.y_pred == [0, 1, 0, 1]
    assert result.y_true == [0, 1, 0, 1]
    assert result.probabilities is None
    assert result.metadata["num_predictions"] == 4


def test_predict_loader_with_probabilities():
    model = make_model()
    loader = make_loader(size=1, batch_size=1)

    predictor = Predictor(model)

    result = predictor.predict_loader(
        loader,
        return_probabilities=True,
    )

    assert result.y_pred == [0, 1]
    assert result.y_true == [0, 1]
    assert result.probabilities is not None
    assert len(result.probabilities) == 2
    assert len(result.probabilities[0]) == 2
    assert result.metadata["return_probabilities"] is True


def test_predict_loader_graph_ids():
    model = make_model()
    loader = make_loader(size=2, batch_size=1)

    predictor = Predictor(model)

    result = predictor.predict_loader(loader)

    assert result.graph_ids == ["model_000", "model_001"]


def test_predict_loader_rejects_none_loader():
    predictor = Predictor(make_model())

    with pytest.raises(ValueError, match="DataLoader cannot be None"):
        predictor.predict_loader(None)


def test_predict_batch():
    model = make_model()
    loader = make_loader(size=1, batch_size=1)
    batch = next(iter(loader))

    predictor = Predictor(model)

    result = predictor.predict_batch(batch)

    assert result.y_pred == [0, 1]
    assert result.y_true == [0, 1]
    assert result.probabilities is None
    assert result.metadata["num_predictions"] == 2


def test_predict_batch_with_probabilities():
    model = make_model()
    loader = make_loader(size=1, batch_size=1)
    batch = next(iter(loader))

    predictor = Predictor(model)

    result = predictor.predict_batch(
        batch,
        return_probabilities=True,
    )

    assert result.probabilities is not None
    assert len(result.probabilities) == 2


def test_predict_batch_rejects_none_batch():
    predictor = Predictor(make_model())

    with pytest.raises(ValueError, match="Batch cannot be None"):
        predictor.predict_batch(None)


def test_predict_convenience_function():
    model = make_model()
    loader = make_loader(size=1, batch_size=1)

    result = predict(model, loader)

    assert isinstance(result, PredictionResult)
    assert result.y_pred == [0, 1]
    assert result.y_true == [0, 1]