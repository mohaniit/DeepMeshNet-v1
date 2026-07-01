"""
Generic training loop for DeepMeshNet-v1.

This trainer is designed for node-level mesh-density classification
using PyTorch Geometric graph data.

It is model-agnostic and works with any model that accepts:
    model(x, edge_index)

and returns node-level logits:
    [num_nodes, num_classes]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from src.learning.checkpoint import Checkpoint, save_checkpoint
from src.learning.metrics import ClassificationMetrics, compute_classification_metrics


@dataclass
class TrainingConfig:
    """
    Training configuration.
    """

    epochs: int = 100
    device: str = "cpu"
    checkpoint_dir: str | Path | None = None
    save_best: bool = True
    metric_name: str = "macro_f1"
    log_every: int = 10


@dataclass
class TrainingHistory:
    """
    Training history container.
    """

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)
    validation_macro_f1: list[float] = field(default_factory=list)
    validation_weighted_f1: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
            "validation_macro_f1": self.validation_macro_f1,
            "validation_weighted_f1": self.validation_weighted_f1,
        }


@dataclass
class TrainingResult:
    """
    Final training result.
    """

    best_epoch: int
    best_metric: float
    history: TrainingHistory
    best_checkpoint_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "best_checkpoint_path": (
                str(self.best_checkpoint_path)
                if self.best_checkpoint_path is not None
                else None
            ),
            "history": self.history.to_dict(),
        }


class Trainer:
    """
    Generic node-classification trainer.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Module,
        config: TrainingConfig | None = None,
    ) -> None:
        if model is None:
            raise ValueError("Model cannot be None.")

        if optimizer is None:
            raise ValueError("Optimizer cannot be None.")

        if loss_fn is None:
            raise ValueError("Loss function cannot be None.")

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)

        self.model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
    ) -> TrainingResult:
        """
        Train model.
        """
        if train_loader is None:
            raise ValueError("train_loader cannot be None.")

        history = TrainingHistory()

        best_epoch = -1
        best_metric = float("-inf")
        best_checkpoint_path: Path | None = None

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            history.train_loss.append(train_loss)

            if validation_loader is not None:
                validation_loss, metrics = self.evaluate(validation_loader)

                history.validation_loss.append(validation_loss)
                history.validation_accuracy.append(metrics.accuracy)
                history.validation_macro_f1.append(metrics.macro_f1)
                history.validation_weighted_f1.append(metrics.weighted_f1)

                current_metric = self._select_metric(metrics)

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch

                    if self.config.save_best and self.config.checkpoint_dir:
                        best_checkpoint_path = self._save_checkpoint(
                            epoch=epoch,
                            loss=validation_loss,
                            metrics=metrics,
                        )

            else:
                current_metric = -train_loss

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch

        return TrainingResult(
            best_epoch=best_epoch,
            best_metric=best_metric,
            history=history,
            best_checkpoint_path=best_checkpoint_path,
        )

    def _train_one_epoch(
        self,
        loader: DataLoader,
    ) -> float:
        """
        Train one epoch.
        """
        self.model.train()

        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(batch.x, batch.edge_index)

            if not hasattr(batch, "y") or batch.y is None:
                raise ValueError("Training batch must contain labels y.")

            loss = self.loss_fn(logits, batch.y)

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        if total_batches == 0:
            raise ValueError("Training loader produced no batches.")

        return total_loss / total_batches

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
    ) -> tuple[float, ClassificationMetrics]:
        """
        Evaluate model.
        """
        if loader is None:
            raise ValueError("loader cannot be None.")

        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        y_true: list[int] = []
        y_pred: list[int] = []

        for batch in loader:
            batch = batch.to(self.device)

            logits = self.model(batch.x, batch.edge_index)

            if not hasattr(batch, "y") or batch.y is None:
                raise ValueError("Evaluation batch must contain labels y.")

            loss = self.loss_fn(logits, batch.y)

            predictions = torch.argmax(logits, dim=1)

            y_true.extend(batch.y.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

            total_loss += float(loss.item())
            total_batches += 1

        if total_batches == 0:
            raise ValueError("Evaluation loader produced no batches.")

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        return total_loss / total_batches, metrics

    def _select_metric(
        self,
        metrics: ClassificationMetrics,
    ) -> float:
        """
        Select metric used for best model tracking.
        """
        metric_name = self.config.metric_name.lower()

        if metric_name == "accuracy":
            return metrics.accuracy

        if metric_name == "macro_f1":
            return metrics.macro_f1

        if metric_name == "weighted_f1":
            return metrics.weighted_f1

        raise ValueError(f"Unsupported metric: {self.config.metric_name}")

    def _save_checkpoint(
        self,
        epoch: int,
        loss: float,
        metrics: ClassificationMetrics,
    ) -> Path:
        """
        Save best checkpoint.
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = Checkpoint(
            epoch=epoch,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            accuracy=metrics.accuracy,
            macro_f1=metrics.macro_f1,
            weighted_f1=metrics.weighted_f1,
            loss=loss,
            metadata={
                "metric_name": self.config.metric_name,
            },
        )

        filename = checkpoint_dir / "best_checkpoint.pt"

        return save_checkpoint(
            checkpoint=checkpoint,
            filename=filename,
        )


__all__ = [
    "TrainingConfig",
    "TrainingHistory",
    "TrainingResult",
    "Trainer",
]