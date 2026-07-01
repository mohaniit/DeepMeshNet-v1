"""
Graph edge feature extraction for DeepMeshNet-v1.

This module enriches CADGraph edges with numerical edge features.

Current version computes graph-level edge features only. Later versions
can extend this module with CAD geometry-aware features such as shared
edge length, dihedral angle, tangency, convexity, and continuity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from src.graph.graph import CADGraph, GraphEdge


EDGE_FEATURE_NAMES = [
    "source_degree",
    "target_degree",
    "degree_difference",
    "feature_distance",
    "edge_bias",
]


@dataclass(frozen=True)
class EdgeFeatureVector:
    """
    Numerical edge feature vector.
    """

    x: list[float]
    feature_names: list[str] = field(
        default_factory=lambda: list(EDGE_FEATURE_NAMES)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "feature_names": self.feature_names,
        }


class EdgeFeatureExtractor:
    """
    Extract graph-level features for CADGraph edges.
    """

    def extract_edge_features(
        self,
        graph: CADGraph,
        edge: GraphEdge,
    ) -> EdgeFeatureVector:
        """
        Extract edge feature vector for a single edge.
        """
        if graph is None:
            raise ValueError("CADGraph cannot be None.")

        if edge is None:
            raise ValueError("GraphEdge cannot be None.")

        source_node = graph.get_node(edge.source)
        target_node = graph.get_node(edge.target)

        source_degree = self._degree(graph, edge.source)
        target_degree = self._degree(graph, edge.target)
        degree_difference = abs(source_degree - target_degree)

        feature_distance = self._feature_distance(
            source_node.features,
            target_node.features,
        )

        return EdgeFeatureVector(
            x=[
                float(source_degree),
                float(target_degree),
                float(degree_difference),
                float(feature_distance),
                1.0,
            ]
        )

    def attach_edge_features(
        self,
        graph: CADGraph,
        overwrite: bool = True,
    ) -> CADGraph:
        """
        Attach edge feature vectors to all edges in graph.
        """
        if graph is None:
            raise ValueError("CADGraph cannot be None.")

        for edge in graph.edges:
            if edge.features and not overwrite:
                continue

            vector = self.extract_edge_features(
                graph=graph,
                edge=edge,
            )

            edge.features = vector.x
            edge.metadata["edge_feature_names"] = vector.feature_names

        return graph

    @staticmethod
    def _degree(
        graph: CADGraph,
        node_id: int,
    ) -> int:
        """
        Compute total directed degree of a node.
        """
        degree = 0

        for edge in graph.edges:
            if edge.source == node_id:
                degree += 1

            if edge.target == node_id:
                degree += 1

        return degree

    @staticmethod
    def _feature_distance(
        source_features: list[float],
        target_features: list[float],
    ) -> float:
        """
        Euclidean distance between two node feature vectors.
        """
        if len(source_features) != len(target_features):
            raise ValueError(
                "Source and target feature vectors must have same length."
            )

        squared_sum = sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(source_features, target_features)
        )

        return sqrt(squared_sum)


def extract_edge_feature_vector(
    graph: CADGraph,
    edge: GraphEdge,
) -> EdgeFeatureVector:
    """
    Convenience function for one edge.
    """
    return EdgeFeatureExtractor().extract_edge_features(
        graph=graph,
        edge=edge,
    )


def attach_edge_features(
    graph: CADGraph,
    overwrite: bool = True,
) -> CADGraph:
    """
    Convenience function for all edges.
    """
    return EdgeFeatureExtractor().attach_edge_features(
        graph=graph,
        overwrite=overwrite,
    )


__all__ = [
    "EDGE_FEATURE_NAMES",
    "EdgeFeatureVector",
    "EdgeFeatureExtractor",
    "extract_edge_feature_vector",
    "attach_edge_features",
]