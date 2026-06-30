"""Core CAD data structures for DeepMeshNet-v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Vector3D = tuple[float, float, float]
BoundingBox3D = tuple[Vector3D, Vector3D]


@dataclass
class VertexData:
    """CAD vertex data."""

    vertex_id: int
    coordinates: Vector3D


@dataclass
class EdgeData:
    """CAD edge data."""

    edge_id: int
    curve_type: str = "UNKNOWN"
    length: float = 0.0
    vertex_ids: list[int] = field(default_factory=list)
    adjacent_face_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FaceData:
    """CAD face data used across DeepMeshNet-v1 modules."""

    face_id: int
    surface_type: str = "UNKNOWN"

    area: float = 0.0
    perimeter: float = 0.0
    centroid: Vector3D = (0.0, 0.0, 0.0)
    normal: Vector3D | None = None
    bounding_box: BoundingBox3D | None = None

    edge_ids: list[int] = field(default_factory=list)
    neighbor_face_ids: list[int] = field(default_factory=list)

    feature_type: str | None = None
    density_label: str | None = None
    decision_reason: str | None = None

    graph_node_index: int | None = None
    color: str | None = None
    visible: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CADModel:
    """Standard CAD model object passed through DeepMeshNet-v1."""

    model_id: str
    file_name: str

    dataset_name: str | None = None
    dataset_split: str | None = None

    backend: str | None = None
    source_path: str | None = None
    _occ_shape: Any | None = None

    faces: list[FaceData] = field(default_factory=list)
    edges: list[EdgeData] = field(default_factory=list)
    vertices: list[VertexData] = field(default_factory=list)

    bounding_box: BoundingBox3D | None = None
    surface_area: float = 0.0
    volume: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    def get_face(self, face_id: int) -> FaceData:
        for face in self.faces:
            if face.face_id == face_id:
                return face

        raise KeyError(f"Face ID not found: {face_id}")