"""
Prediction utilities for DeepMeshNet-v1.

This module runs node-level mesh-density prediction using a trained
PyTorch/PyG-compatible model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader


@dataclass
class PredictionResult:
    """
    Prediction result for one or more graph batches.
    """

    y_pred: list[int]
    y_true: list[int] | None = None
    probabilities: list[list[float]] | None = None
    graph_ids: list[str | None] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "y_pred": self.y_pred,
            "y_true": self.y_true,
            "probabilities": self.probabilities,
            "graph_ids": self.graph_ids,
            "metadata": self.metadata,
        }


class Predictor:
    """
    Generic node-level predictor.
    """

    def __init__(
        self,
        model: Module,
        device: str = "cpu",
    ) -> None:
        if model is None:
            raise ValueError("Model cannot be None.")

        self.model = model
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_loader(
        self,
        loader: DataLoader,
        return_probabilities: bool = False,
    ) -> PredictionResult:
        """
        Predict node labels for all batches in a DataLoader.
        """
        if loader is None:
            raise ValueError("DataLoader cannot be None.")

        y_pred: list[int] = []
        y_true: list[int] = []
        probabilities: list[list[float]] = []
        graph_ids: list[str | None] = []

        for batch in loader:
            batch = batch.to(self.device)

            logits = self.model(batch.x, batch.edge_index)
            preds = torch.argmax(logits, dim=1)

            y_pred.extend(preds.detach().cpu().tolist())

            if hasattr(batch, "y") and batch.y is not None:
                y_true.extend(batch.y.detach().cpu().tolist())

            if return_probabilities:
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.detach().cpu().tolist())

            graph_ids.extend(_extract_graph_ids(batch))

        return PredictionResult(
            y_pred=y_pred,
            y_true=y_true if y_true else None,
            probabilities=probabilities if return_probabilities else None,
            graph_ids=graph_ids,
            metadata={
                "return_probabilities": return_probabilities,
                "num_predictions": len(y_pred),
            },
        )

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Any,
        return_probabilities: bool = False,
    ) -> PredictionResult:
        """
        Predict node labels for a single PyG batch/data object.
        """
        if batch is None:
            raise ValueError("Batch cannot be None.")

        batch = batch.to(self.device)

        logits = self.model(batch.x, batch.edge_index)
        preds = torch.argmax(logits, dim=1)

        y_true = None

        if hasattr(batch, "y") and batch.y is not None:
            y_true = batch.y.detach().cpu().tolist()

        probabilities = None

        if return_probabilities:
            probabilities = torch.softmax(
                logits,
                dim=1,
            ).detach().cpu().tolist()

        return PredictionResult(
            y_pred=preds.detach().cpu().tolist(),
            y_true=y_true,
            probabilities=probabilities,
            graph_ids=_extract_graph_ids(batch),
            metadata={
                "return_probabilities": return_probabilities,
                "num_predictions": int(preds.numel()),
            },
        )


def predict(
    model: Module,
    loader: DataLoader,
    device: str = "cpu",
    return_probabilities: bool = False,
) -> PredictionResult:
    """
    Convenience prediction function.
    """
    predictor = Predictor(
        model=model,
        device=device,
    )

    return predictor.predict_loader(
        loader=loader,
        return_probabilities=return_probabilities,
    )


def _extract_graph_ids(batch: Any) -> list[str | None]:
    """
    Extract graph IDs from PyG Data/Batch objects.

    For batched graphs, PyG may collate graph_id into a list.
    For a single graph, graph_id may be a scalar string.
    """
    graph_id = getattr(batch, "graph_id", None)

    if graph_id is None:
        return []

    if isinstance(graph_id, list):
        return graph_id

    if isinstance(graph_id, tuple):
        return list(graph_id)

    return [graph_id]


__all__ = [
    "PredictionResult",
    "Predictor",
    "predict",
]