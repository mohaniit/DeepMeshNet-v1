"""
Engineering complexity scoring for DeepMeshNet-v1.

This module converts extracted face features into a normalized engineering
complexity score in the range [0, 1].
"""

from __future__ import annotations

from typing import Any

from src.geometry.cad_model import CADModel
from src.knowledge.rules import (
    DEFAULT_FEATURE_THRESHOLDS,
    FeatureThresholds,
    is_curved_surface,
    is_high_complexity_surface,
)


def clamp01(value: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def compute_surface_complexity(surface_type: str | None) -> float:
    """
    Compute base complexity from surface type.
    """
    if surface_type is None:
        return 0.40

    surface = str(surface_type).strip().upper()

    if surface == "PLANE":
        return 0.10

    if surface in {"CYLINDER", "CONE"}:
        return 0.35

    if surface in {"SPHERE", "TORUS"}:
        return 0.55

    if is_high_complexity_surface(surface):
        return 0.75

    return 0.40


def compute_compactness_complexity(compactness: float) -> float:
    """
    Low compactness indicates elongated or irregular faces.
    """
    compactness = clamp01(compactness)
    return clamp01(1.0 - compactness)


def compute_edge_complexity(
    edge_count: int,
    thresholds: FeatureThresholds = DEFAULT_FEATURE_THRESHOLDS,
) -> float:
    """
    Normalize edge count contribution.
    """
    if edge_count <= 0:
        return 0.0

    return clamp01(edge_count / thresholds.high_edge_count)


def compute_neighbor_complexity(
    neighbor_count: int,
    thresholds: FeatureThresholds = DEFAULT_FEATURE_THRESHOLDS,
) -> float:
    """
    Normalize neighbor count contribution.
    """
    if neighbor_count <= 0:
        return 0.0

    return clamp01(neighbor_count / thresholds.high_neighbor_count)


def compute_area_complexity(
    area: float,
    thresholds: FeatureThresholds = DEFAULT_FEATURE_THRESHOLDS,
) -> float:
    """
    Conservative area-scale complexity.

    Very small faces receive higher complexity because they often require
    local mesh refinement. Very large faces receive low complexity here.
    """
    if area <= 0.0:
        return 0.0

    if area <= thresholds.small_area:
        return 1.0

    if area >= thresholds.large_area:
        return 0.0

    normalized = 1.0 - (
        (area - thresholds.small_area)
        / (thresholds.large_area - thresholds.small_area)
    )

    return clamp01(normalized)


def compute_curvature_complexity(surface_type: str | None) -> float:
    """
    Curved surfaces receive additional complexity.
    """
    return 1.0 if is_curved_surface(surface_type) else 0.0


def compute_face_complexity(features: dict[str, Any]) -> float:
    """
    Compute normalized engineering complexity score for one face.
    """
    if features is None:
        raise ValueError("Feature dictionary cannot be None.")

    surface_type = features.get("surface_type", "UNKNOWN")
    compactness = float(features.get("compactness", 0.0))
    edge_count = int(features.get("edge_count", 0))
    neighbor_count = int(features.get("neighbor_count", 0))
    area = float(features.get("area", 0.0))

    score = (
        0.30 * compute_surface_complexity(surface_type)
        + 0.20 * compute_compactness_complexity(compactness)
        + 0.15 * compute_edge_complexity(edge_count)
        + 0.10 * compute_neighbor_complexity(neighbor_count)
        + 0.15 * compute_area_complexity(area)
        + 0.10 * compute_curvature_complexity(surface_type)
    )

    return clamp01(score)


def attach_complexity_to_model(model: CADModel) -> CADModel:
    """
    Compute and attach complexity score to each face.

    Requires each face to already contain:
        face.features
    """
    if model is None:
        raise ValueError("CADModel cannot be None.")

    if not hasattr(model, "faces"):
        raise AttributeError("CADModel must contain a 'faces' attribute.")

    for face in model.faces:
        features = getattr(face, "features", None)

        if features is None:
            raise AttributeError(
                "Each face must contain 'features'. "
                "Run attach_features_to_model() first."
            )

        face.complexity_score = compute_face_complexity(features)

    return model


def get_complexity_components() -> list[str]:
    """
    Return ordered complexity component names.
    """
    return [
        "surface_complexity",
        "compactness_complexity",
        "edge_complexity",
        "neighbor_complexity",
        "area_complexity",
        "curvature_complexity",
    ]