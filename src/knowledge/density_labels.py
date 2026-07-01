"""
Mesh-density label assignment for DeepMeshNet-v1.

This module converts engineering complexity scores into standardized
mesh-density labels and integer class IDs.
"""

from __future__ import annotations

from typing import Any

from src.geometry.cad_model import CADModel
from src.knowledge.rules import (
    DEFAULT_COMPLEXITY_THRESHOLDS,
    ComplexityThresholds,
    complexity_to_density,
    density_to_id,
    get_density_labels,
    id_to_density,
)


def safe_complexity_score(value: Any, default: float = 0.0) -> float:
    """
    Safely convert complexity score to a clamped float in [0, 1].
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default

    if score != score:  # NaN check
        return default

    return max(0.0, min(1.0, score))


def assign_density_label(
    complexity_score: float,
    thresholds: ComplexityThresholds = DEFAULT_COMPLEXITY_THRESHOLDS,
) -> str:
    """
    Assign mesh-density label from normalized complexity score.
    """
    score = safe_complexity_score(complexity_score)
    return complexity_to_density(score, thresholds)


def assign_density_id(
    complexity_score: float,
    thresholds: ComplexityThresholds = DEFAULT_COMPLEXITY_THRESHOLDS,
) -> int:
    """
    Assign mesh-density integer class ID from normalized complexity score.
    """
    label = assign_density_label(complexity_score, thresholds)
    return density_to_id(label)


def attach_density_labels_to_model(
    model: CADModel,
    thresholds: ComplexityThresholds = DEFAULT_COMPLEXITY_THRESHOLDS,
) -> CADModel:
    """
    Attach density labels and density IDs to all faces in CADModel.

    Requires each face to already contain:
        face.complexity_score

    Populates:
        face.density_label
        face.density_id
    """
    if model is None:
        raise ValueError("CADModel cannot be None.")

    if not hasattr(model, "faces"):
        raise AttributeError("CADModel must contain a 'faces' attribute.")

    for face in model.faces:
        if not hasattr(face, "complexity_score"):
            raise AttributeError(
                "Each face must contain 'complexity_score'. "
                "Run attach_complexity_to_model() first."
            )

        score = safe_complexity_score(face.complexity_score)
        face.density_label = assign_density_label(score, thresholds)
        face.density_id = density_to_id(face.density_label)

    return model


def summarize_density_labels(model: CADModel) -> dict[str, int]:
    """
    Count density labels in a CADModel.
    """
    if model is None:
        raise ValueError("CADModel cannot be None.")

    if not hasattr(model, "faces"):
        raise AttributeError("CADModel must contain a 'faces' attribute.")

    summary = {label: 0 for label in get_density_labels()}

    for face in model.faces:
        label = getattr(face, "density_label", None)

        if label in summary:
            summary[label] += 1

    return summary


def get_density_label_fields() -> list[str]:
    """
    Return fields populated by this module.
    """
    return [
        "density_label",
        "density_id",
    ]


__all__ = [
    "safe_complexity_score",
    "assign_density_label",
    "assign_density_id",
    "attach_density_labels_to_model",
    "summarize_density_labels",
    "get_density_label_fields",
    "density_to_id",
    "id_to_density",
]