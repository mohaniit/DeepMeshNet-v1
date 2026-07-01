import pytest
import torch

from src.models.base_model import BaseGraphModel, ModelConfig


class DummyGraphModel(BaseGraphModel):
    model_name = "dummy"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.in_channels, config.out_channels)

    def forward(self, x, edge_index):
        return self.linear(x)


def test_model_config_to_dict():
    config = ModelConfig(
        in_channels=4,
        hidden_channels=8,
        out_channels=3,
        dropout=0.2,
    )

    assert config.to_dict() == {
        "in_channels": 4,
        "hidden_channels": 8,
        "out_channels": 3,
        "dropout": 0.2,
    }


def test_model_config_validate_success():
    config = ModelConfig(
        in_channels=4,
        hidden_channels=8,
        out_channels=3,
        dropout=0.5,
    )

    config.validate()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"in_channels": 0, "hidden_channels": 8, "out_channels": 3}, "in_channels"),
        ({"in_channels": 4, "hidden_channels": 0, "out_channels": 3}, "hidden_channels"),
        ({"in_channels": 4, "hidden_channels": 8, "out_channels": 0}, "out_channels"),
        ({"in_channels": 4, "hidden_channels": 8, "out_channels": 3, "dropout": -0.1}, "dropout"),
        ({"in_channels": 4, "hidden_channels": 8, "out_channels": 3, "dropout": 1.0}, "dropout"),
    ],
)
def test_model_config_validate_rejects_invalid_values(kwargs, message):
    config = ModelConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_base_graph_model_initialization():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=4,
        out_channels=2,
    )

    model = DummyGraphModel(config)

    assert model.config is config
    assert model.model_name == "dummy"


def test_base_graph_model_rejects_none_config():
    with pytest.raises(ValueError, match="ModelConfig cannot be None"):
        DummyGraphModel(None)


def test_forward_output_shape():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=4,
        out_channels=3,
    )

    model = DummyGraphModel(config)

    x = torch.randn(5, 2)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    logits = model(x, edge_index)

    assert logits.shape == (5, 3)


def test_predict_logits():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=4,
        out_channels=3,
    )

    model = DummyGraphModel(config)

    x = torch.randn(5, 2)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    logits = model.predict_logits(x, edge_index)

    assert logits.shape == (5, 3)
    assert model.training is False


def test_predict_classes():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=4,
        out_channels=3,
    )

    model = DummyGraphModel(config)

    x = torch.randn(5, 2)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    predictions = model.predict_classes(x, edge_index)

    assert predictions.shape == (5,)
    assert predictions.dtype == torch.long


def test_num_parameters():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=4,
        out_channels=3,
    )

    model = DummyGraphModel(config)

    assert model.num_parameters() > 0


def test_model_summary():
    config = ModelConfig(
        in_channels=2,
        hidden_channels=4,
        out_channels=3,
        dropout=0.1,
    )

    model = DummyGraphModel(config)

    summary = model.model_summary()

    assert summary["model_name"] == "dummy"
    assert summary["config"] == config.to_dict()
    assert summary["num_parameters"] == model.num_parameters()