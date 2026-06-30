"""
Feature extraction utilities for the Engineering Mesh Knowledge Engine.

This module converts enriched CADModel face data into standardized
feature dictionaries. It extracts only; it does not assign complexity
scores or mesh-density labels.
"""

from __future__ import annotations

from typing import Any

from src.geometry.cad_model import CADModel
from src.knowledge.rules import (
    is_analytical_surface,
    is_curved_surface,
    is_freeform_surface,
    is_high_complexity_surface,
    normalize_surface_type,
)


SURFACE_TYPE_TO_ID: dict[str, int] = {
    "PLANE": 0,
    "CYLINDER": 1,
    "CONE": 2,
    "SPHERE": 3,
    "TORUS": 4,
    "BSPLINE": 5,
    "BEZIER": 6,
    "OFFSET": 7,
    "REVOLUTION": 8,
    "EXTRUSION": 9,
    "OTHER": 10,
    "UNKNOWN": 11,
}

ID_TO_SURFACE_TYPE: dict[int, str] = {
    value: key for key, value in SURFACE_TYPE_TO_ID.items()
}


def surface_type_to_id(surface_type: str | None) -> int:
    """
    Convert surface type label to integer ID.
    """
    normalized = normalize_surface_type(surface_type)
    return SURFACE_TYPE_TO_ID.get(normalized, SURFACE_TYPE_TO_ID["UNKNOWN"])


def id_to_surface_type(surface_type_id: int) -> str:
    """
    Convert surface type ID to label.
    """
    return ID_TO_SURFACE_TYPE.get(surface_type_id, "UNKNOWN")


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default

    if converted != converted:  # NaN check
        return default

    return converted


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_face_features(face: Any, face_id: int | None = None) -> dict[str, Any]:
    """
    Extract standardized features from one CAD face object.

    Parameters
    ----------
    face:
        FaceData-like object.
    face_id:
        Optional fallback face ID.

    Returns
    -------
    dict[str, Any]
        Standardized face feature dictionary.
    """
    if face is None:
        raise ValueError("Face cannot be None.")

    resolved_face_id = getattr(face, "face_id", None)
    if resolved_face_id is None:
        resolved_face_id = getattr(face, "id", None)
    if resolved_face_id is None:
        resolved_face_id = face_id

    surface_type = normalize_surface_type(getattr(face, "surface_type", None))
    surface_type_id = surface_type_to_id(surface_type)

    edges = getattr(face, "edges", [])
    neighbors = getattr(face, "neighbors", [])

    edge_count = safe_int(getattr(face, "edge_count", len(edges)))
    neighbor_count = safe_int(getattr(face, "neighbor_count", len(neighbors)))

    area = safe_float(getattr(face, "area", 0.0))
    perimeter = safe_float(getattr(face, "perimeter", 0.0))
    compactness = safe_float(getattr(face, "compactness", 0.0))

    return {
        "face_id": resolved_face_id,
        "surface_type": surface_type,
        "surface_type_id": surface_type_id,
        "area": area,
        "perimeter": perimeter,
        "compactness": compactness,
        "edge_count": edge_count,
        "neighbor_count": neighbor_count,
        "is_analytical": is_analytical_surface(surface_type),
        "is_freeform": is_freeform_surface(surface_type),
        "is_curved": is_curved_surface(surface_type),
        "is_high_complexity_surface": is_high_complexity_surface(surface_type),
    }


def extract_model_features(model: CADModel) -> list[dict[str, Any]]:
    """
    Extract standardized features for all faces in a CADModel.
    """
    if model is None:
        raise ValueError("CADModel cannot be None.")

    if not hasattr(model, "faces"):
        raise AttributeError("CADModel must contain a 'faces' attribute.")

    features: list[dict[str, Any]] = []

    for index, face in enumerate(model.faces):
        features.append(extract_face_features(face, face_id=index))

    return features


def attach_features_to_model(model: CADModel) -> CADModel:
    """
    Extract features and attach each feature dictionary back to its face.

    Populates:
        face.features
    """
    if model is None:
        raise ValueError("CADModel cannot be None.")

    if not hasattr(model, "faces"):
        raise AttributeError("CADModel must contain a 'faces' attribute.")

    for index, face in enumerate(model.faces):
        face.features = extract_face_features(face, face_id=index)

    return model


def get_feature_names() -> list[str]:
    """
    Return ordered feature names produced by this module.
    """
    return [
        "face_id",
        "surface_type",
        "surface_type_id",
        "area",
        "perimeter",
        "compactness",
        "edge_count",
        "neighbor_count",
        "is_analytical",
        "is_freeform",
        "is_curved",
        "is_high_complexity_surface",
    ]