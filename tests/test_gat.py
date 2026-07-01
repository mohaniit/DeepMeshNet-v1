import pytest
import torch

from src.models.base_model import ModelConfig
from src.models.gat import GAT, create_gat


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


def test_gat_initialization():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        dropout=0.1,
    )

    model = GAT(
        config=config,
        heads=2,
    )

    assert model.model_name == "gat"
    assert model.config is config
    assert model.heads == 2
    assert model.conv1.in_channels == 2
    assert model.conv1.out_channels == 8
    assert model.conv1.heads == 2
    assert model.conv2.in_channels == 16
    assert model.conv2.out_channels == 4
    assert model.conv2.heads == 1


def test_gat_rejects_invalid_heads():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
    )

    with pytest.raises(ValueError, match="heads must be positive"):
        GAT(
            config=config,
            heads=0,
        )


def test_gat_forward_shape():
    model = create_gat(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        dropout=0.1,
        heads=2,
    )

    x, edge_index = make_graph()

    logits = model(x, edge_index)

    assert logits.shape == (3, 4)


def test_gat_predict_classes():
    model = create_gat(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        heads=2,
    )

    x, edge_index = make_graph()

    predictions = model.predict_classes(x, edge_index)

    assert predictions.shape == (3,)
    assert predictions.dtype == torch.long


def test_gat_model_summary():
    model = create_gat(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        dropout=0.2,
        heads=3,
    )

    summary = model.model_summary()

    assert summary["model_name"] == "gat"
    assert summary["config"] == {
        "in_channels": 2,
        "hidden_channels": 8,
        "out_channels": 4,
        "dropout": 0.2,
    }
    assert summary["heads"] == 3
    assert summary["num_parameters"] == model.num_parameters()
    assert summary["num_parameters"] > 0


def test_create_gat_factory():
    model = create_gat(
        in_channels=3,
        hidden_channels=16,
        out_channels=5,
        dropout=0.3,
        heads=2,
    )

    assert isinstance(model, GAT)
    assert model.config.in_channels == 3
    assert model.config.hidden_channels == 16
    assert model.config.out_channels == 5
    assert model.config.dropout == 0.3
    assert model.heads == 2