"""
ML-ready feature-vector assembly for DeepMeshNet-v1.

This module converts extracted engineering face features into numerical
vectors suitable for classical ML models and later graph neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.geometry.cad_model import CADModel


FEATURE_VECTOR_NAMES: list[str] = [
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
    "complexity_score",
]


@dataclass(frozen=True)
class FaceFeatureVector:
    """
    ML-ready feature-vector container for one CAD face.
    """

    face_id: int | None
    x: list[float]
    y: int | None = None
    density_label: str | None = None


def safe_numeric(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to finite float.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default

    if numeric != numeric:  # NaN check
        return default

    if numeric == float("inf") or numeric == float("-inf"):
        return default

    return numeric


def bool_to_float(value: Any) -> float:
    """
    Convert boolean-like values to 0.0 or 1.0.
    """
    return 1.0 if bool(value) else 0.0


def build_face_feature_vector(face: Any) -> FaceFeatureVector:
    """
    Build an ML-ready numerical feature vector for one face.

    Requires:
        face.features
        face.complexity_score

    Optional:
        face.density_id
        face.density_label
    """
    if face is None:
        raise ValueError("Face cannot be None.")

    features = getattr(face, "features", None)

    if features is None:
        raise AttributeError(
            "Face must contain 'features'. Run attach_features_to_model() first."
        )

    if not hasattr(face, "complexity_score"):
        raise AttributeError(
            "Face must contain 'complexity_score'. "
            "Run attach_complexity_to_model() first."
        )

    face_id = features.get("face_id", getattr(face, "face_id", None))

    x = [
        safe_numeric(features.get("surface_type_id", 0)),
        safe_numeric(features.get("area", 0.0)),
        safe_numeric(features.get("perimeter", 0.0)),
        safe_numeric(features.get("compactness", 0.0)),
        safe_numeric(features.get("edge_count", 0)),
        safe_numeric(features.get("neighbor_count", 0)),
        bool_to_float(features.get("is_analytical", False)),
        bool_to_float(features.get("is_freeform", False)),
        bool_to_float(features.get("is_curved", False)),
        bool_to_float(features.get("is_high_complexity_surface", False)),
        safe_numeric(getattr(face, "complexity_score", 0.0)),
    ]

    y = getattr(face, "density_id", None)
    density_label = getattr(face, "density_label", None)

    return FaceFeatureVector(
        face_id=face_id,
        x=x,
        y=y,
        density_label=density_label,
    )


def build_model_feature_vectors(model: CADModel) -> list[FaceFeatureVector]:
    """
    Build feature vectors for all faces in CADModel.
    """
    if model is None:
        raise ValueError("CADModel cannot be None.")

    if not hasattr(model, "faces"):
        raise AttributeError("CADModel must contain a 'faces' attribute.")

    return [build_face_feature_vector(face) for face in model.faces]


def build_xy(model: CADModel) -> tuple[list[list[float]], list[int]]:
    """
    Build X and y arrays from a CADModel.

    Requires every face to contain density_id.
    """
    vectors = build_model_feature_vectors(model)

    x_values: list[list[float]] = []
    y_values: list[int] = []

    for vector in vectors:
        if vector.y is None:
            raise AttributeError(
                "Each face must contain 'density_id'. "
                "Run attach_density_labels_to_model() first."
            )

        x_values.append(vector.x)
        y_values.append(int(vector.y))

    return x_values, y_values


def attach_feature_vectors_to_model(model: CADModel) -> CADModel:
    """
    Attach ML-ready feature vectors to each face.

    Populates:
        face.feature_vector
    """
    if model is None:
        raise ValueError("CADModel cannot be None.")

    if not hasattr(model, "faces"):
        raise AttributeError("CADModel must contain a 'faces' attribute.")

    for face in model.faces:
        face.feature_vector = build_face_feature_vector(face)

    return model


def get_feature_vector_names() -> list[str]:
    """
    Return ordered numerical feature-vector names.
    """
    return list(FEATURE_VECTOR_NAMES)