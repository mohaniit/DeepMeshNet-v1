"""CAD topology extraction for DeepMeshNet-v1."""

from __future__ import annotations

from typing import Any

from src.geometry.backend import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopExp_Explorer,
    require_cad_backend,
)
from src.geometry.cad_model import CADModel, EdgeData, FaceData


def iter_subshapes(shape: Any, shape_type: Any) -> list[Any]:
    """Return subshapes of a given OpenCascade shape type."""
    require_cad_backend()

    explorer = TopExp_Explorer(shape, shape_type)
    subshapes: list[Any] = []

    while explorer.More():
        subshapes.append(explorer.Current())
        explorer.Next()

    return subshapes


def iter_faces(shape: Any) -> list[Any]:
    """Return all faces from a CAD shape."""
    return iter_subshapes(shape, TopAbs_FACE)


def iter_edges(shape: Any) -> list[Any]:
    """Return all edges from a CAD shape."""
    return iter_subshapes(shape, TopAbs_EDGE)


def shape_hash(shape: Any, upper: int = 2_147_483_647) -> int:
    """Return OpenCascade shape hash."""
    if hasattr(shape, "HashCode"):
        return int(shape.HashCode(upper))

    return int(hash(str(shape)))


def extract_topology(model: CADModel) -> CADModel:
    """Extract faces, edges, and face adjacency into a CADModel."""

    if model._occ_shape is None:
        raise ValueError(
            "CADModel has no loaded OCC shape. Use load_step_file() first."
        )

    require_cad_backend()

    faces_raw = iter_faces(model._occ_shape)

    edge_hash_to_id: dict[int, int] = {}
    edge_to_faces: dict[int, list[int]] = {}
    faces: list[FaceData] = []

    for face_index, raw_face in enumerate(faces_raw):
        face_id = face_index
        raw_edges = iter_edges(raw_face)

        edge_ids: list[int] = []

        for raw_edge in raw_edges:
            edge_hash = shape_hash(raw_edge)

            if edge_hash not in edge_hash_to_id:
                edge_hash_to_id[edge_hash] = len(edge_hash_to_id)

            edge_id = edge_hash_to_id[edge_hash]
            edge_ids.append(edge_id)

            edge_to_faces.setdefault(edge_id, [])

            if face_id not in edge_to_faces[edge_id]:
                edge_to_faces[edge_id].append(face_id)

        faces.append(
            FaceData(
                face_id=face_id,
                edge_ids=edge_ids,
                metadata={
                    "num_edges": len(edge_ids),
                    "occ_hash": shape_hash(raw_face),
                },
            )
        )

    edges: list[EdgeData] = []

    for edge_hash, edge_id in sorted(edge_hash_to_id.items(), key=lambda item: item[1]):
        adjacent_faces = sorted(edge_to_faces.get(edge_id, []))

        edges.append(
            EdgeData(
                edge_id=edge_id,
                adjacent_face_ids=adjacent_faces,
                metadata={
                    "occ_hash": edge_hash,
                    "num_adjacent_faces": len(adjacent_faces),
                },
            )
        )

    for face in faces:
        neighbors: set[int] = set()

        for edge_id in face.edge_ids:
            for adjacent_face_id in edge_to_faces.get(edge_id, []):
                if adjacent_face_id != face.face_id:
                    neighbors.add(adjacent_face_id)

        face.neighbor_face_ids = sorted(neighbors)

    model.faces = faces
    model.edges = edges

    model.metadata["topology_extracted"] = True
    model.metadata["num_faces"] = model.face_count
    model.metadata["num_edges"] = model.edge_count

    return model