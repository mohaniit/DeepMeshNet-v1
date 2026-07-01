from pathlib import Path
import time

import pytest
import torch

from src.learning.checkpoint import (
    Checkpoint,
    checkpoint_summary,
    latest_checkpoint,
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_model,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)


def make_checkpoint() -> Checkpoint:
    return Checkpoint(
        epoch=25,
        model_state={
            "layer.weight": torch.randn(2, 4),
        },
        optimizer_state={
            "lr": 0.001,
        },
        scheduler_state={
            "step": 25,
        },
        accuracy=0.95,
        macro_f1=0.94,
        weighted_f1=0.95,
        loss=0.123,
        metadata={
            "model": "gcn",
            "dataset": "CM500",
        },
    )


# ------------------------------------------------------------------
# Dataclass
# ------------------------------------------------------------------

def test_checkpoint_to_dict():
    checkpoint = make_checkpoint()

    data = checkpoint.to_dict()

    assert data["epoch"] == 25
    assert data["accuracy"] == 0.95
    assert data["macro_f1"] == 0.94
    assert data["weighted_f1"] == 0.95
    assert data["loss"] == 0.123
    assert data["metadata"]["model"] == "gcn"


# ------------------------------------------------------------------
# Save / Load Checkpoint
# ------------------------------------------------------------------

def test_save_and_load_checkpoint(tmp_path):
    checkpoint = make_checkpoint()

    filename = tmp_path / "checkpoint.pt"

    saved = save_checkpoint(checkpoint, filename)

    assert saved.exists()

    loaded = load_checkpoint(filename)

    assert loaded.epoch == checkpoint.epoch
    assert loaded.accuracy == checkpoint.accuracy
    assert loaded.macro_f1 == checkpoint.macro_f1
    assert loaded.weighted_f1 == checkpoint.weighted_f1
    assert loaded.loss == checkpoint.loss
    assert loaded.metadata == checkpoint.metadata
    assert loaded.optimizer_state == checkpoint.optimizer_state
    assert loaded.scheduler_state == checkpoint.scheduler_state

    assert torch.equal(
        loaded.model_state["layer.weight"],
        checkpoint.model_state["layer.weight"],
    )


def test_save_checkpoint_rejects_none(tmp_path):
    with pytest.raises(ValueError, match="Checkpoint cannot be None"):
        save_checkpoint(None, tmp_path / "a.pt")


def test_load_checkpoint_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


# ------------------------------------------------------------------
# Save / Load Model
# ------------------------------------------------------------------

def test_save_and_load_model(tmp_path):
    model = DummyModel()

    filename = tmp_path / "model.pt"

    save_model(model, filename)

    assert filename.exists()

    restored = DummyModel()

    load_model(restored, filename)

    for p1, p2 in zip(
        model.parameters(),
        restored.parameters(),
    ):
        assert torch.equal(p1, p2)


def test_save_model_rejects_none(tmp_path):
    with pytest.raises(ValueError, match="Model cannot be None"):
        save_model(None, tmp_path / "model.pt")


def test_load_model_rejects_none(tmp_path):
    with pytest.raises(ValueError, match="Model cannot be None"):
        load_model(None, tmp_path / "model.pt")


def test_load_model_missing_file(tmp_path):
    model = DummyModel()

    with pytest.raises(FileNotFoundError):
        load_model(model, tmp_path / "missing.pt")


# ------------------------------------------------------------------
# Latest checkpoint
# ------------------------------------------------------------------

def test_latest_checkpoint(tmp_path):
    first = tmp_path / "epoch_001.pt"
    second = tmp_path / "epoch_002.pt"

    save_checkpoint(make_checkpoint(), first)

    time.sleep(0.02)

    save_checkpoint(make_checkpoint(), second)

    latest = latest_checkpoint(tmp_path)

    assert latest == second


def test_latest_checkpoint_empty_directory(tmp_path):
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_missing_directory():
    assert latest_checkpoint("directory_that_does_not_exist") is None


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def test_checkpoint_summary():
    checkpoint = make_checkpoint()

    summary = checkpoint_summary(checkpoint)

    assert summary == {
        "epoch": 25,
        "accuracy": 0.95,
        "macro_f1": 0.94,
        "weighted_f1": 0.95,
        "loss": 0.123,
        "metadata": {
            "model": "gcn",
            "dataset": "CM500",
        },
    }


def test_checkpoint_summary_rejects_none():
    with pytest.raises(ValueError, match="Checkpoint cannot be None"):
        checkpoint_summary(None)