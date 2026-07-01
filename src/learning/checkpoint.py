"""
Model checkpoint utilities for DeepMeshNet-v1.

This module provides framework utilities for saving and loading
training checkpoints.

It is intentionally independent of any specific GNN architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class Checkpoint:
    """
    Training checkpoint metadata.
    """

    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None

    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    loss: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "loss": self.loss,
            "metadata": self.metadata,
        }


def save_checkpoint(
    checkpoint: Checkpoint,
    filename: str | Path,
) -> Path:
    """
    Save checkpoint to disk.
    """
    if checkpoint is None:
        raise ValueError("Checkpoint cannot be None.")

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint.to_dict(),
        filename,
    )

    return filename


def load_checkpoint(
    filename: str | Path,
    map_location: str | torch.device = "cpu",
) -> Checkpoint:
    """
    Load checkpoint from disk.
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    data = torch.load(
        filename,
        map_location=map_location,
        weights_only=False,
    )

    return Checkpoint(
        epoch=data["epoch"],
        model_state=data["model_state"],
        optimizer_state=data.get("optimizer_state"),
        scheduler_state=data.get("scheduler_state"),
        accuracy=data.get("accuracy", 0.0),
        macro_f1=data.get("macro_f1", 0.0),
        weighted_f1=data.get("weighted_f1", 0.0),
        loss=data.get("loss", 0.0),
        metadata=data.get("metadata", {}),
    )


def save_model(
    model: torch.nn.Module,
    filename: str | Path,
) -> Path:
    """
    Save only model weights.
    """
    if model is None:
        raise ValueError("Model cannot be None.")

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        filename,
    )

    return filename


def load_model(
    model: torch.nn.Module,
    filename: str | Path,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    """
    Load model weights.
    """
    if model is None:
        raise ValueError("Model cannot be None.")

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    state = torch.load(
        filename,
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(state)

    return model


def latest_checkpoint(
    directory: str | Path,
    extension: str = "*.pt",
) -> Path | None:
    """
    Return the most recently modified checkpoint.
    """
    directory = Path(directory)

    if not directory.exists():
        return None

    files = list(directory.glob(extension))

    if not files:
        return None

    return max(
        files,
        key=lambda path: path.stat().st_mtime,
    )


def checkpoint_summary(
    checkpoint: Checkpoint,
) -> dict[str, Any]:
    """
    Return checkpoint summary.
    """
    if checkpoint is None:
        raise ValueError("Checkpoint cannot be None.")

    return {
        "epoch": checkpoint.epoch,
        "accuracy": checkpoint.accuracy,
        "macro_f1": checkpoint.macro_f1,
        "weighted_f1": checkpoint.weighted_f1,
        "loss": checkpoint.loss,
        "metadata": checkpoint.metadata,
    }


__all__ = [
    "Checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "save_model",
    "load_model",
    "latest_checkpoint",
    "checkpoint_summary",
]