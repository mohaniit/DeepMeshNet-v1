import torch

from src.models.base_model import ModelConfig
from src.models.gcn import GCN, create_gcn


def make_graph():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )

    return x, edge_index


def test_gcn_initialization():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        dropout=0.1,
    )

    model = GCN(config)

    assert model.model_name == "gcn"
    assert model.config is config
    assert model.conv1.in_channels == 2
    assert model.conv1.out_channels == 8
    assert model.conv2.in_channels == 8
    assert model.conv2.out_channels == 4


def test_gcn_forward_shape():
    model = create_gcn(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        dropout=0.1,
    )

    x, edge_index = make_graph()

    logits = model(x, edge_index)

    assert logits.shape == (3, 4)


def test_gcn_predict_classes():
    model = create_gcn(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
    )

    x, edge_index = make_graph()

    predictions = model.predict_classes(x, edge_index)

    assert predictions.shape == (3,)
    assert predictions.dtype == torch.long


def test_gcn_model_summary():
    model = create_gcn(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        dropout=0.2,
    )

    summary = model.model_summary()

    assert summary["model_name"] == "gcn"
    assert summary["config"] == {
        "in_channels": 2,
        "hidden_channels": 8,
        "out_channels": 4,
        "dropout": 0.2,
    }
    assert summary["num_parameters"] == model.num_parameters()
    assert summary["num_parameters"] > 0


def test_create_gcn_factory():
    model = create_gcn(
        in_channels=3,
        hidden_channels=16,
        out_channels=5,
        dropout=0.3,
    )

    assert isinstance(model, GCN)
    assert model.config.in_channels == 3
    assert model.config.hidden_channels == 16
    assert model.config.out_channels == 5
    assert model.config.dropout == 0.3