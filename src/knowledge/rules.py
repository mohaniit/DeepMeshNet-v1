"""
Engineering Mesh Knowledge Engine rules for DeepMeshNet-v1.

This module defines the standardized engineering vocabulary used by EMKE:
surface classes, density labels, complexity thresholds, and conservative
rule helpers.
"""

from __future__ import annotations

from dataclasses import dataclass


DENSITY_LOW = "LOW"
DENSITY_MEDIUM = "MEDIUM"
DENSITY_HIGH = "HIGH"
DENSITY_VERY_HIGH = "VERY_HIGH"

DENSITY_LABELS: list[str] = [
    DENSITY_LOW,
    DENSITY_MEDIUM,
    DENSITY_HIGH,
    DENSITY_VERY_HIGH,
]

DENSITY_TO_ID: dict[str, int] = {
    DENSITY_LOW: 0,
    DENSITY_MEDIUM: 1,
    DENSITY_HIGH: 2,
    DENSITY_VERY_HIGH: 3,
}

ID_TO_DENSITY: dict[int, str] = {
    value: key for key, value in DENSITY_TO_ID.items()
}


ANALYTICAL_SURFACES: set[str] = {
    "PLANE",
    "CYLINDER",
    "CONE",
    "SPHERE",
    "TORUS",
}

FREEFORM_SURFACES: set[str] = {
    "BSPLINE",
    "BEZIER",
    "OFFSET",
    "REVOLUTION",
    "EXTRUSION",
    "OTHER",
    "UNKNOWN",
}

CURVED_SURFACES: set[str] = {
    "CYLINDER",
    "CONE",
    "SPHERE",
    "TORUS",
    "BSPLINE",
    "BEZIER",
    "OFFSET",
    "REVOLUTION",
}

HIGH_COMPLEXITY_SURFACES: set[str] = {
    "TORUS",
    "BSPLINE",
    "BEZIER",
    "OFFSET",
    "REVOLUTION",
    "OTHER",
    "UNKNOWN",
}


@dataclass(frozen=True)
class ComplexityThresholds:
    """
    Thresholds for converting normalized complexity score into density labels.
    """

    low_max: float = 0.25
    medium_max: float = 0.50
    high_max: float = 0.75


DEFAULT_COMPLEXITY_THRESHOLDS = ComplexityThresholds()


@dataclass(frozen=True)
class FeatureThresholds:
    """
    Conservative thresholds used by EMKE feature rules.
    """

    small_area: float = 10.0
    large_area: float = 1000.0
    low_compactness: float = 0.25
    high_edge_count: int = 8
    high_neighbor_count: int = 6


DEFAULT_FEATURE_THRESHOLDS = FeatureThresholds()


def normalize_surface_type(surface_type: str | None) -> str:
    """
    Normalize surface type string.
    """
    if surface_type is None:
        return "UNKNOWN"

    normalized = str(surface_type).strip().upper()

    if not normalized:
        return "UNKNOWN"

    return normalized


def is_analytical_surface(surface_type: str | None) -> bool:
    """
    Return True if surface is a basic analytical CAD surface.
    """
    return normalize_surface_type(surface_type) in ANALYTICAL_SURFACES


def is_freeform_surface(surface_type: str | None) -> bool:
    """
    Return True if surface is freeform or non-simple.
    """
    return normalize_surface_type(surface_type) in FREEFORM_SURFACES


def is_curved_surface(surface_type: str | None) -> bool:
    """
    Return True if surface is geometrically curved.
    """
    return normalize_surface_type(surface_type) in CURVED_SURFACES


def is_high_complexity_surface(surface_type: str | None) -> bool:
    """
    Return True if surface type usually requires higher mesh attention.
    """
    return normalize_surface_type(surface_type) in HIGH_COMPLEXITY_SURFACES


def density_to_id(label: str) -> int:
    """
    Convert density label to integer class ID.
    """
    normalized = str(label).strip().upper()

    if normalized not in DENSITY_TO_ID:
        raise ValueError(f"Unknown density label: {label}")

    return DENSITY_TO_ID[normalized]


def id_to_density(class_id: int) -> str:
    """
    Convert integer class ID to density label.
    """
    if class_id not in ID_TO_DENSITY:
        raise ValueError(f"Unknown density class ID: {class_id}")

    return ID_TO_DENSITY[class_id]


def complexity_to_density(
    complexity_score: float,
    thresholds: ComplexityThresholds = DEFAULT_COMPLEXITY_THRESHOLDS,
) -> str:
    """
    Convert normalized complexity score to mesh-density label.

    Expected input range is 0 to 1. Values outside this range are clamped.
    """
    score = max(0.0, min(1.0, float(complexity_score)))

    if score <= thresholds.low_max:
        return DENSITY_LOW

    if score <= thresholds.medium_max:
        return DENSITY_MEDIUM

    if score <= thresholds.high_max:
        return DENSITY_HIGH

    return DENSITY_VERY_HIGH


def get_density_labels() -> list[str]:
    """
    Return supported density labels.
    """
    return list(DENSITY_LABELS)


def get_surface_rule_summary() -> dict[str, list[str]]:
    """
    Return surface rule groups for reporting/debugging.
    """
    return {
        "analytical_surfaces": sorted(ANALYTICAL_SURFACES),
        "freeform_surfaces": sorted(FREEFORM_SURFACES),
        "curved_surfaces": sorted(CURVED_SURFACES),
        "high_complexity_surfaces": sorted(HIGH_COMPLEXITY_SURFACES),
    }