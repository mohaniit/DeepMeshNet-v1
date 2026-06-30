"""
Geometry measurement utilities for DeepMeshNet-v1.

This module computes conservative geometric measurements for CAD faces
and stores them inside the CADModel face data.
"""

from __future__ import annotations

import math
from typing import Any

from OCP.BRep import BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from src.geometry.backend import require_cad_backend
from src.geometry.cad_model import CADModel


def compute_face_area(face_shape: Any) -> float:
    """
    Compute surface area of a CAD face.

    Parameters
    ----------
    face_shape:
        OpenCascade TopoDS_Face object.

    Returns
    -------
    float
        Face area. Returns 0.0 if unavailable.
    """
    require_cad_backend()

    if face_shape is None:
        return 0.0

    try:
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face_shape, props)
        area = float(props.Mass())

        if not math.isfinite(area) or area < 0.0:
            return 0.0

        return area

    except Exception:
        return 0.0


def compute_edge_length(edge_shape: Any) -> float:
    """
    Compute length of a CAD edge.

    Parameters
    ----------
    edge_shape:
        OpenCascade TopoDS_Edge object.

    Returns
    -------
    float
        Edge length. Returns 0.0 if unavailable.
    """
    require_cad_backend()

    if edge_shape is None:
        return 0.0

    try:
        props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge_shape, props)
        length = float(props.Mass())

        if not math.isfinite(length) or length < 0.0:
            return 0.0

        return length

    except Exception:
        return 0.0


def compute_face_perimeter(face: Any) -> float:
    """
    Compute face perimeter from its stored edge list.

    Parameters
    ----------
    face:
        FaceData object with an edges attribute.

    Returns
    -------
    float
        Sum of edge lengths.
    """
    if face is None:
        return 0.0

    edges = getattr(face, "edges", [])

    perimeter = 0.0

    for edge in edges:
        edge_shape = getattr(edge, "shape", None)
        perimeter += compute_edge_length(edge_shape)

    if not math.isfinite(perimeter) or perimeter < 0.0:
        return 0.0

    return perimeter


def compute_compactness(area: float, perimeter: float) -> float:
    """
    Compute dimensionless face compactness.

    Formula:
        compactness = 4*pi*area / perimeter^2

    A perfect circle gives compactness close to 1.
    Smaller or elongated faces give lower values.
    """
    if area <= 0.0 or perimeter <= 0.0:
        return 0.0

    value = (4.0 * math.pi * area) / (perimeter * perimeter)

    if not math.isfinite(value):
        return 0.0

    return max(0.0, value)


def populate_face_measurements(model: CADModel) -> CADModel:
    """
    Compute and populate basic measurements for all faces in CADModel.

    Populated fields:
        face.area
        face.perimeter
        face.compactness

    Parameters
    ----------
    model:
        CADModel already populated with topology.

    Returns
    -------
    CADModel
        Same model instance with measurement fields populated.
    """
    if model is None:
        raise ValueError("CADModel cannot be None.")

    if not hasattr(model, "faces"):
        raise AttributeError("CADModel must contain a 'faces' attribute.")

    for face in model.faces:
        face_shape = getattr(face, "shape", None)

        face.area = compute_face_area(face_shape)
        face.perimeter = compute_face_perimeter(face)
        face.compactness = compute_compactness(face.area, face.perimeter)

    return model


def get_measurement_fields() -> list[str]:
    """
    Return the measurement fields populated by this module.
    """
    return [
        "area",
        "perimeter",
        "compactness",
    ]