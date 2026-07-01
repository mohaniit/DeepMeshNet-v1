"""
Evaluation utilities for DeepMeshNet-v1.

This module evaluates node-level mesh-density predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.learning.metrics import ClassificationMetrics, compute_classification_metrics
from src.learning.predictor import PredictionResult


@dataclass(frozen=True)
class EvaluationReport:
    """
    Evaluation report container.
    """

    metrics: ClassificationMetrics
    num_samples: int
    num_predictions: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_predictions": self.num_predictions,
            "metrics": self.metrics.to_dict(),
            "metadata": self.metadata,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_predictions": self.num_predictions,
            "accuracy": self.metrics.accuracy,
            "macro_f1": self.metrics.macro_f1,
            "weighted_f1": self.metrics.weighted_f1,
            "metadata": self.metadata,
        }


class Evaluator:
    """
    Evaluates PredictionResult objects.
    """

    def __init__(
        self,
        labels: list[int] | None = None,
        target_names: list[str] | None = None,
    ) -> None:
        self.labels = labels
        self.target_names = target_names

    def evaluate_prediction_result(
        self,
        result: PredictionResult,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """
        Evaluate prediction result.
        """
        if result is None:
            raise ValueError("PredictionResult cannot be None.")

        if result.y_true is None:
            raise ValueError("PredictionResult must contain y_true.")

        metrics = compute_classification_metrics(
            y_true=result.y_true,
            y_pred=result.y_pred,
            labels=self.labels,
            target_names=self.target_names,
        )

        report_metadata = dict(result.metadata)

        if metadata:
            report_metadata.update(metadata)

        return EvaluationReport(
            metrics=metrics,
            num_samples=len(result.graph_ids),
            num_predictions=len(result.y_pred),
            metadata=report_metadata,
        )

    def evaluate_arrays(
        self,
        y_true: list[int],
        y_pred: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """
        Evaluate raw label arrays.
        """
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=self.labels,
            target_names=self.target_names,
        )

        return EvaluationReport(
            metrics=metrics,
            num_samples=0,
            num_predictions=len(y_pred),
            metadata=metadata or {},
        )


def evaluate_predictions(
    result: PredictionResult,
    labels: list[int] | None = None,
    target_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """
    Convenience function for evaluating PredictionResult.
    """
    evaluator = Evaluator(
        labels=labels,
        target_names=target_names,
    )

    return evaluator.evaluate_prediction_result(
        result=result,
        metadata=metadata,
    )


def save_evaluation_report(
    report: EvaluationReport,
    filename: str | Path,
) -> Path:
    """
    Save evaluation report as JSON.
    """
    if report is None:
        raise ValueError("EvaluationReport cannot be None.")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(report.to_dict(), fp, indent=4)

    return filename


def load_evaluation_report(
    filename: str | Path,
) -> dict[str, Any]:
    """
    Load evaluation report JSON as dictionary.
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    with open(filename, "r", encoding="utf-8") as fp:
        return json.load(fp)


__all__ = [
    "EvaluationReport",
    "Evaluator",
    "evaluate_predictions",
    "save_evaluation_report",
    "load_evaluation_report",
]