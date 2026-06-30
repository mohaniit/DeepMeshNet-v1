"""
CAD model validation utilities for DeepMeshNet-v1.

This module performs conservative structural checks on CADModel objects.
It does not repair geometry. It only reports validity status, errors,
warnings, and summary counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.geometry.cad_model import CADModel


@dataclass
class ValidationResult:
    """Validation result container."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def validate_cad_model(model: CADModel) -> ValidationResult:
    """
    Validate the structural completeness of a CADModel.

    Parameters
    ----------
    model:
        CADModel instance after STEP loading and topology population.

    Returns
    -------
    ValidationResult
        Validation status, errors, warnings, and summary.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if model is None:
        return ValidationResult(
            is_valid=False,
            errors=["CADModel cannot be None."],
            warnings=[],
            summary={},
        )

    faces = getattr(model, "faces", None)
    edges = getattr(model, "edges", None)
    vertices = getattr(model, "vertices", None)

    if faces is None:
        errors.append("CADModel is missing faces.")
        faces = []

    if edges is None:
        errors.append("CADModel is missing edges.")
        edges = []

    if vertices is None:
        warnings.append("CADModel is missing vertices.")
        vertices = []

    if len(faces) == 0:
        errors.append("CADModel contains no faces.")

    if len(edges) == 0:
        warnings.append("CADModel contains no edges.")

    unknown_surface_count = 0
    zero_area_count = 0
    zero_perimeter_count = 0

    for index, face in enumerate(faces):
        surface_type = getattr(face, "surface_type", None)
        area = getattr(face, "area", None)
        perimeter = getattr(face, "perimeter", None)

        if surface_type in (None, "", "UNKNOWN"):
            unknown_surface_count += 1

        if area is not None and area <= 0.0:
            zero_area_count += 1

        if perimeter is not None and perimeter <= 0.0:
            zero_perimeter_count += 1

        if getattr(face, "shape", None) is None:
            warnings.append(f"Face {index} has no shape reference.")

    if unknown_surface_count > 0:
        warnings.append(
            f"{unknown_surface_count} face(s) have unknown or missing surface type."
        )

    if zero_area_count > 0:
        warnings.append(f"{zero_area_count} face(s) have zero or negative area.")

    if zero_perimeter_count > 0:
        warnings.append(
            f"{zero_perimeter_count} face(s) have zero or negative perimeter."
        )

    summary = {
        "face_count": len(faces),
        "edge_count": len(edges),
        "vertex_count": len(vertices),
        "unknown_surface_count": unknown_surface_count,
        "zero_area_count": zero_area_count,
        "zero_perimeter_count": zero_perimeter_count,
    }

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def require_valid_cad_model(model: CADModel) -> CADModel:
    """
    Validate CADModel and raise ValueError if invalid.

    Returns the original model when valid.
    """
    result = validate_cad_model(model)

    if not result.is_valid:
        message = "Invalid CADModel: " + "; ".join(result.errors)
        raise ValueError(message)

    return model


def get_validation_summary(model: CADModel) -> dict[str, Any]:
    """
    Return only the validation summary dictionary.
    """
    return validate_cad_model(model).summary