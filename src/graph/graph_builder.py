"""
CAD Graph Builder for DeepMeshNet-v1.

Converts an enriched CADModel into a neutral CADGraph.
"""

from __future__ import annotations

from typing import Any

from src.geometry.cad_model import CADModel
from src.graph.graph import CADGraph, GraphEdge, GraphNode


class GraphBuilder:
    """Build CADGraph objects from enriched CADModel objects."""

    def __init__(self, bidirectional: bool = True) -> None:
        self.bidirectional = bidirectional

    def build_graph(self, model: CADModel) -> CADGraph:
        """Build CADGraph from CADModel."""
        self._validate_model(model)

        graph = CADGraph(
            graph_id=self._resolve_graph_id(model),
            metadata=self._extract_model_metadata(model),
        )

        face_id_to_node_id: dict[int, int] = {}

        for node_id, face in enumerate(model.faces):
            face_id = self._resolve_face_id(face, node_id)
            face_id_to_node_id[face_id] = node_id

            feature_vector = getattr(face, "feature_vector")

            graph.add_node(
                GraphNode(
                    node_id=node_id,
                    face_id=face_id,
                    features=list(feature_vector.x),
                    label=getattr(feature_vector, "y", None),
                    metadata=self._extract_face_metadata(face),
                )
            )

        for source_node_id, face in enumerate(model.faces):
            source_face_id = self._resolve_face_id(face, source_node_id)
            neighbors = self._resolve_neighbors(face)

            for neighbor_face_id in neighbors:
                if neighbor_face_id not in face_id_to_node_id:
                    continue

                target_node_id = face_id_to_node_id[neighbor_face_id]

                if source_node_id == target_node_id:
                    continue

                self._add_edge_if_missing(
                    graph=graph,
                    source=source_node_id,
                    target=target_node_id,
                    metadata={
                        "source_face_id": source_face_id,
                        "target_face_id": neighbor_face_id,
                        "edge_type": "FACE_ADJACENCY",
                    },
                )

                if self.bidirectional:
                    self._add_edge_if_missing(
                        graph=graph,
                        source=target_node_id,
                        target=source_node_id,
                        metadata={
                            "source_face_id": neighbor_face_id,
                            "target_face_id": source_face_id,
                            "edge_type": "FACE_ADJACENCY",
                        },
                    )

        return graph

    def _validate_model(self, model: CADModel) -> None:
        if model is None:
            raise ValueError("CADModel cannot be None.")

        if not hasattr(model, "faces"):
            raise AttributeError("CADModel must contain a 'faces' attribute.")

        if len(model.faces) == 0:
            raise ValueError("CADModel must contain at least one face.")

        for index, face in enumerate(model.faces):
            if not hasattr(face, "feature_vector"):
                raise AttributeError(
                    f"Face {index} must contain 'feature_vector'. "
                    "Run attach_feature_vectors_to_model() first."
                )

            feature_vector = getattr(face, "feature_vector")

            if feature_vector is None:
                raise AttributeError(f"Face {index} has an empty 'feature_vector'.")

            if not hasattr(feature_vector, "x"):
                raise AttributeError(
                    f"Face {index} feature_vector must contain 'x'."
                )

    @staticmethod
    def _resolve_graph_id(model: CADModel) -> str:
        metadata = getattr(model, "metadata", None)

        if isinstance(metadata, dict):
            for key in ("model_name", "file_name", "name"):
                value = metadata.get(key)
                if value:
                    return str(value)

        for attr in ("model_name", "file_name", "name"):
            value = getattr(model, attr, None)
            if value:
                return str(value)

        return "cad_graph"

    @staticmethod
    def _extract_model_metadata(model: CADModel) -> dict[str, Any]:
        metadata = getattr(model, "metadata", None)

        if isinstance(metadata, dict):
            return dict(metadata)

        return {}

    @staticmethod
    def _resolve_face_id(face: Any, fallback: int) -> int:
        for attr in ("face_id", "id", "index"):
            value = getattr(face, attr, None)
            if value is not None:
                return int(value)

        return int(fallback)

    @staticmethod
    def _resolve_neighbors(face: Any) -> list[int]:
        for attr in (
            "neighbors",
            "neighbor_faces",
            "neighbor_face_ids",
            "adjacent_faces",
            "adjacent_face_ids",
        ):
            neighbors = getattr(face, attr, None)

            if neighbors is None:
                continue

            return sorted({int(neighbor) for neighbor in neighbors})

        return []

    @staticmethod
    def _extract_face_metadata(face: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        metadata_keys = [
            "surface_type",
            "area",
            "perimeter",
            "compactness",
            "complexity_score",
            "density_label",
            "density_id",
        ]

        for key in metadata_keys:
            if hasattr(face, key):
                metadata[key] = getattr(face, key)

        return metadata

    @staticmethod
    def _edge_exists(graph: CADGraph, source: int, target: int) -> bool:
        return any(
            edge.source == source and edge.target == target
            for edge in graph.edges
        )

    def _add_edge_if_missing(
        self,
        graph: CADGraph,
        source: int,
        target: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._edge_exists(graph, source, target):
            return

        graph.add_edge(
            GraphEdge(
                source=source,
                target=target,
                features=[1.0],
                metadata=metadata or {},
            )
        )


def build_cad_graph(
    model: CADModel,
    bidirectional: bool = True,
) -> CADGraph:
    """Convenience function for building a CADGraph."""
    return GraphBuilder(bidirectional=bidirectional).build_graph(model)


__all__ = [
    "GraphBuilder",
    "build_cad_graph",
]