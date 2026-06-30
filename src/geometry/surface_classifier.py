"""
Surface classification utilities for DeepMeshNet-v1.

This module classifies OpenCascade CAD faces into standard analytical
surface-type labels and stores the result inside CADModel face data.
"""

from __future__ import annotations

from typing import Any

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_SurfaceType

from src.geometry.backend import require_cad_backend
from src.geometry.cad_model import CADModel


SURFACE_TYPE_MAP: dict[GeomAbs_SurfaceType, str] = {
    GeomAbs_SurfaceType.GeomAbs_Plane: "PLANE",
    GeomAbs_SurfaceType.GeomAbs_Cylinder: "CYLINDER",
    GeomAbs_SurfaceType.GeomAbs_Cone: "CONE",
    GeomAbs_SurfaceType.GeomAbs_Sphere: "SPHERE",
    GeomAbs_SurfaceType.GeomAbs_Torus: "TORUS",
    GeomAbs_SurfaceType.GeomAbs_BezierSurface: "BEZIER",
    GeomAbs_SurfaceType.GeomAbs_BSplineSurface: "BSPLINE",
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution: "REVOLUTION",
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion: "EXTRUSION",
    GeomAbs_SurfaceType.GeomAbs_OffsetSurface: "OFFSET",
    GeomAbs_SurfaceType.GeomAbs_OtherSurface: "OTHER",
}


def classify_surface_type(face_shape: Any) -> str:
    """
    Classify a single OpenCascade face into a standard surface type.

    Parameters
    ----------
    face_shape:
        OpenCascade TopoDS_Face object.

    Returns
    -------
    str
        Standardized surface type label.
    """
    require_cad_backend()

    if face_shape is None:
        return "UNKNOWN"

    try:
        adaptor = BRepAdaptor_Surface(face_shape, True)
        surface_type = adaptor.GetType()
        return SURFACE_TYPE_MAP.get(surface_type, "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def classify_model_surfaces(model: CADModel) -> CADModel:
    """
    Classify all faces in a CADModel and populate face.surface_type.

    Parameters
    ----------
    model:
        CADModel already populated with topology information.

    Returns
    -------
    CADModel
        Same CADModel instance with surface types populated.
    """
    if model is None:
        raise ValueError("CADModel cannot be None.")

    if not hasattr(model, "faces"):
        raise AttributeError("CADModel must contain a 'faces' attribute.")

    for face in model.faces:
        face_shape = getattr(face, "shape", None)
        face.surface_type = classify_surface_type(face_shape)

    return model


def get_supported_surface_types() -> list[str]:
    """
    Return the supported standardized surface type labels.
    """
    return [
        "PLANE",
        "CYLINDER",
        "CONE",
        "SPHERE",
        "TORUS",
        "BSPLINE",
        "BEZIER",
        "OFFSET",
        "REVOLUTION",
        "EXTRUSION",
        "OTHER",
        "UNKNOWN",
    ]