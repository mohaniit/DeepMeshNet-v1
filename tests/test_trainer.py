import pytest
import torch

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.learning.dataloader import make_dataloader_from_graph_dataset
from src.learning.dataset import GraphDataset
from src.learning.graph_sample import GraphSample
from src.learning.trainer import (
    Trainer,
    TrainingConfig,
    TrainingHistory,
    TrainingResult,
)


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


def make_loader(size: int = 4, batch_size: int = 2):
    dataset = GraphDataset()

    for index in range(size):
        dataset.add_sample(make_sample(f"model_{index:03d}"))

    return make_dataloader_from_graph_dataset(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def make_trainer(epochs: int = 2, checkpoint_dir=None):
    model = DummyNodeClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    config = TrainingConfig(
        epochs=epochs,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        save_best=checkpoint_dir is not None,
        metric_name="macro_f1",
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
    )


def test_training_config_defaults():
    config = TrainingConfig()

    assert config.epochs == 100
    assert config.device == "cpu"
    assert config.checkpoint_dir is None
    assert config.save_best is True
    assert config.metric_name == "macro_f1"
    assert config.log_every == 10


def test_training_history_to_dict():
    history = TrainingHistory(
        train_loss=[1.0],
        validation_loss=[0.8],
        validation_accuracy=[0.5],
        validation_macro_f1=[0.4],
        validation_weighted_f1=[0.45],
    )

    assert history.to_dict() == {
        "train_loss": [1.0],
        "validation_loss": [0.8],
        "validation_accuracy": [0.5],
        "validation_macro_f1": [0.4],
        "validation_weighted_f1": [0.45],
    }


def test_training_result_to_dict():
    history = TrainingHistory(train_loss=[1.0])
    result = TrainingResult(
        best_epoch=1,
        best_metric=0.5,
        history=history,
        best_checkpoint_path=None,
    )

    data = result.to_dict()

    assert data["best_epoch"] == 1
    assert data["best_metric"] == 0.5
    assert data["best_checkpoint_path"] is None
    assert data["history"] == history.to_dict()


def test_trainer_initialization():
    trainer = make_trainer()

    assert isinstance(trainer.model, torch.nn.Module)
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)
    assert isinstance(trainer.loss_fn, torch.nn.Module)
    assert trainer.device.type == "cpu"


def test_trainer_rejects_none_model():
    optimizer = torch.optim.Adam(DummyNodeClassifier().parameters(), lr=0.01)

    with pytest.raises(ValueError, match="Model cannot be None"):
        Trainer(
            model=None,
            optimizer=optimizer,
            loss_fn=torch.nn.CrossEntropyLoss(),
        )


def test_trainer_rejects_none_optimizer():
    with pytest.raises(ValueError, match="Optimizer cannot be None"):
        Trainer(
            model=DummyNodeClassifier(),
            optimizer=None,
            loss_fn=torch.nn.CrossEntropyLoss(),
        )


def test_trainer_rejects_none_loss_fn():
    model = DummyNodeClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="Loss function cannot be None"):
        Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=None,
        )


def test_train_one_epoch():
    trainer = make_trainer()
    loader = make_loader()

    loss = trainer._train_one_epoch(loader)

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_evaluate():
    trainer = make_trainer()
    loader = make_loader()

    loss, metrics = trainer.evaluate(loader)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert 0.0 <= metrics.weighted_f1 <= 1.0


def test_train_with_validation():
    trainer = make_trainer(epochs=2)
    train_loader = make_loader()
    validation_loader = make_loader()

    result = trainer.train(
        train_loader=train_loader,
        validation_loader=validation_loader,
    )

    assert isinstance(result, TrainingResult)
    assert result.best_epoch >= 1
    assert len(result.history.train_loss) == 2
    assert len(result.history.validation_loss) == 2
    assert len(result.history.validation_accuracy) == 2
    assert len(result.history.validation_macro_f1) == 2
    assert len(result.history.validation_weighted_f1) == 2


def test_train_without_validation():
    trainer = make_trainer(epochs=2)
    train_loader = make_loader()

    result = trainer.train(train_loader=train_loader)

    assert isinstance(result, TrainingResult)
    assert result.best_epoch >= 1
    assert len(result.history.train_loss) == 2
    assert result.history.validation_loss == []


def test_train_saves_best_checkpoint(tmp_path):
    trainer = make_trainer(
        epochs=2,
        checkpoint_dir=tmp_path,
    )

    train_loader = make_loader()
    validation_loader = make_loader()

    result = trainer.train(
        train_loader=train_loader,
        validation_loader=validation_loader,
    )

    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.exists()
    assert result.best_checkpoint_path.name == "best_checkpoint.pt"


def test_train_rejects_none_loader():
    trainer = make_trainer()

    with pytest.raises(ValueError, match="train_loader cannot be None"):
        trainer.train(None)


def test_evaluate_rejects_none_loader():
    trainer = make_trainer()

    with pytest.raises(ValueError, match="loader cannot be None"):
        trainer.evaluate(None)


def test_select_metric_accuracy():
    trainer = make_trainer()
    loader = make_loader()

    _, metrics = trainer.evaluate(loader)

    trainer.config.metric_name = "accuracy"

    value = trainer._select_metric(metrics)

    assert value == metrics.accuracy


def test_select_metric_macro_f1():
    trainer = make_trainer()
    loader = make_loader()

    _, metrics = trainer.evaluate(loader)

    trainer.config.metric_name = "macro_f1"

    value = trainer._select_metric(metrics)

    assert value == metrics.macro_f1


def test_select_metric_weighted_f1():
    trainer = make_trainer()
    loader = make_loader()

    _, metrics = trainer.evaluate(loader)

    trainer.config.metric_name = "weighted_f1"

    value = trainer._select_metric(metrics)

    assert value == metrics.weighted_f1


def test_select_metric_rejects_unknown_metric():
    trainer = make_trainer()
    loader = make_loader()

    _, metrics = trainer.evaluate(loader)

    trainer.config.metric_name = "unknown"

    with pytest.raises(ValueError, match="Unsupported metric"):
        trainer._select_metric(metrics)