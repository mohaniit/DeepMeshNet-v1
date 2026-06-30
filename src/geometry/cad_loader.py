"""CAD loading utilities for DeepMeshNet-v1."""

from __future__ import annotations

from pathlib import Path

from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader

from src.geometry.cad_model import CADModel


SUPPORTED_STEP_EXTENSIONS = {".step", ".stp"}


def load_step_file(
    step_path: str | Path,
    model_id: str | None = None,
    dataset_name: str | None = None,
    dataset_split: str | None = None,
) -> CADModel:
    """Load a STEP/STP file and return a CADModel with the raw OCC shape attached."""
    path = Path(step_path)

    if not path.exists():
        raise FileNotFoundError(f"STEP file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_STEP_EXTENSIONS:
        raise ValueError(f"Unsupported CAD file extension: {path.suffix}")

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))

    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    if shape.IsNull():
        raise RuntimeError(f"Loaded STEP shape is null: {path}")

    return CADModel(
        model_id=model_id or path.stem,
        file_name=path.name,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        backend="OCP",
        source_path=str(path),
        _occ_shape=shape,
        metadata={
            "loader": "STEPControl_Reader",
            "extension": path.suffix.lower(),
        },
    )