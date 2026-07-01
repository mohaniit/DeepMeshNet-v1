import pytest
import torch

from src.models.base_model import ModelConfig
from src.models.gine import GINE, create_gine


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

    edge_attr = torch.tensor(
        [
            [2.0, 3.0, 1.0, 1.414, 1.0],
            [3.0, 2.0, 1.0, 1.414, 1.0],
            [3.0, 1.0, 2.0, 1.000, 1.0],
            [1.0, 3.0, 2.0, 1.000, 1.0],
        ],
        dtype=torch.float32,
    )

    return x, edge_index, edge_attr


def test_gine_initialization():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        dropout=0.1,
    )

    model = GINE(
        config=config,
        edge_dim=5,
    )

    assert model.model_name == "gine"
    assert model.config is config
    assert model.edge_dim == 5
    assert model.mlp1 is not None
    assert model.conv1 is not None
    assert model.mlp2 is not None
    assert model.conv2 is not None


def test_gine_rejects_invalid_edge_dim():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
    )

    with pytest.raises(ValueError, match="edge_dim must be positive"):
        GINE(
            config=config,
            edge_dim=0,
        )


def test_gine_forward_shape():
    model = create_gine(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        edge_dim=5,
        dropout=0.1,
    )

    x, edge_index, edge_attr = make_graph()

    logits = model(x, edge_index, edge_attr)

    assert logits.shape == (3, 4)


def test_gine_requires_edge_attr():
    model = create_gine(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        edge_dim=5,
    )

    x, edge_index, _ = make_graph()

    with pytest.raises(ValueError, match="edge_attr"):
        model(x, edge_index)


def test_gine_predict_logits():
    model = create_gine(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        edge_dim=5,
    )

    x, edge_index, edge_attr = make_graph()

    logits = model.predict_logits(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    assert logits.shape == (3, 4)


def test_gine_predict_classes():
    model = create_gine(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        edge_dim=5,
    )

    x, edge_index, edge_attr = make_graph()

    predictions = model.predict_classes(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    assert predictions.shape == (3,)
    assert predictions.dtype == torch.long


def test_gine_model_summary():
    model = create_gine(
        in_channels=2,
        hidden_channels=8,
        out_channels=4,
        edge_dim=5,
        dropout=0.2,
    )

    summary = model.model_summary()

    assert summary["model_name"] == "gine"
    assert summary["config"] == {
        "in_channels": 2,
        "hidden_channels": 8,
        "out_channels": 4,
        "dropout": 0.2,
    }
    assert summary["edge_dim"] == 5
    assert summary["num_parameters"] == model.num_parameters()
    assert summary["num_parameters"] > 0


def test_create_gine_factory():
    model = create_gine(
        in_channels=3,
        hidden_channels=16,
        out_channels=5,
        edge_dim=5,
        dropout=0.3,
    )

    assert isinstance(model, GINE)
    assert model.config.in_channels == 3
    assert model.config.hidden_channels == 16
    assert model.config.out_channels == 5
    assert model.config.dropout == 0.3
    assert model.edge_dim == 5