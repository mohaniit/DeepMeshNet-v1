"""
Learning metrics for DeepMeshNet-v1.

This module provides lightweight classification metrics for mesh-density
prediction tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """
    Classification metric container.
    """

    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: list[list[int]]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "confusion_matrix": self.confusion_matrix,
            "report": self.report,
        }


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[int] | None = None,
    target_names: list[str] | None = None,
    zero_division: int = 0,
) -> ClassificationMetrics:
    """
    Compute classification metrics.
    """
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")

    if len(y_true_array) != len(y_pred_array):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(y_true_array) == 0:
        raise ValueError("Metric inputs cannot be empty.")

    accuracy = float(accuracy_score(y_true_array, y_pred_array))

    macro_f1 = float(
        f1_score(
            y_true_array,
            y_pred_array,
            labels=labels,
            average="macro",
            zero_division=zero_division,
        )
    )

    weighted_f1 = float(
        f1_score(
            y_true_array,
            y_pred_array,
            labels=labels,
            average="weighted",
            zero_division=zero_division,
        )
    )

    matrix = confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=labels,
    ).astype(int)

    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=zero_division,
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        confusion_matrix=matrix.tolist(),
        report=report,
    )


def accuracy(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> float:
    """
    Compute accuracy.
    """
    return compute_classification_metrics(y_true, y_pred).accuracy


def macro_f1(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> float:
    """
    Compute macro F1 score.
    """
    return compute_classification_metrics(y_true, y_pred).macro_f1


def weighted_f1(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> float:
    """
    Compute weighted F1 score.
    """
    return compute_classification_metrics(y_true, y_pred).weighted_f1


def confusion_matrix_as_list(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[int] | None = None,
) -> list[list[int]]:
    """
    Compute confusion matrix as nested list.
    """
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")

    if len(y_true_array) != len(y_pred_array):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(y_true_array) == 0:
        raise ValueError("Metric inputs cannot be empty.")

    return confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=labels,
    ).astype(int).tolist()


def _as_1d_array(
    values: list[int] | np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Convert input to 1D numpy array.
    """
    if values is None:
        raise ValueError(f"{name} cannot be None.")

    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    return array


__all__ = [
    "ClassificationMetrics",
    "compute_classification_metrics",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "confusion_matrix_as_list",
]