"""CAD backend detection utilities for DeepMeshNet-v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendInfo:
    """Information about the active CAD backend."""

    name: str
    available: bool
    import_error: str | None = None


try:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    BACKEND = BackendInfo(name="OCP", available=True)

except Exception as exc:  # pragma: no cover
    IFSelect_RetDone = None
    STEPControl_Reader = None
    TopAbs_EDGE = None
    TopAbs_FACE = None
    TopAbs_VERTEX = None
    TopExp_Explorer = None

    BACKEND = BackendInfo(
        name="unavailable",
        available=False,
        import_error=str(exc),
    )


def active_cad_backend() -> str:
    """Return the active CAD backend name."""
    return BACKEND.name


def cad_backend_available() -> bool:
    """Return True if a CAD backend is available."""
    return BACKEND.available


def require_cad_backend() -> None:
    """Raise ImportError if no CAD backend is available."""
    if not BACKEND.available:
        raise ImportError(
            "OpenCascade/OCP backend is unavailable. "
            f"Import error: {BACKEND.import_error}"
        )


def get_backend_info() -> BackendInfo:
    """Return backend information."""
    return BACKEND


__all__ = [
    "BackendInfo",
    "BACKEND",
    "IFSelect_RetDone",
    "STEPControl_Reader",
    "TopAbs_EDGE",
    "TopAbs_FACE",
    "TopAbs_VERTEX",
    "TopExp_Explorer",
    "active_cad_backend",
    "cad_backend_available",
    "require_cad_backend",
    "get_backend_info",
]