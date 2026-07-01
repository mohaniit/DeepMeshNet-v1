"""
Graph analytics and statistics for DeepMeshNet-v1.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from src.graph.graph import CADGraph


@dataclass
class GraphStatistics:
    graph_id: str | None
    node_count: int
    edge_count: int
    average_degree: float
    min_degree: int
    max_degree: int
    graph_density: float
    connected_components: int
    isolated_nodes: list[int] = field(default_factory=list)
    feature_dimension: int = 0
    label_distribution: dict[int, int] = field(default_factory=dict)
    surface_distribution: dict[str, int] = field(default_factory=dict)
    density_distribution: dict[str, int] = field(default_factory=dict)
    duplicate_edges: int = 0
    self_loops: int = 0

    @property
    def is_connected(self) -> bool:
        return self.connected_components == 1 if self.node_count > 0 else False

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "average_degree": self.average_degree,
            "min_degree": self.min_degree,
            "max_degree": self.max_degree,
            "graph_density": self.graph_density,
            "connected_components": self.connected_components,
            "is_connected": self.is_connected,
            "isolated_nodes": self.isolated_nodes,
            "isolated_node_count": len(self.isolated_nodes),
            "feature_dimension": self.feature_dimension,
            "label_distribution": self.label_distribution,
            "surface_distribution": self.surface_distribution,
            "density_distribution": self.density_distribution,
            "duplicate_edges": self.duplicate_edges,
            "self_loops": self.self_loops,
        }


def compute_graph_statistics(graph: CADGraph) -> GraphStatistics:
    if graph is None:
        raise ValueError("CADGraph cannot be None.")

    node_ids = [node.node_id for node in graph.nodes]
    node_id_set = set(node_ids)

    degree = {node_id: 0 for node_id in node_ids}
    edge_pairs = []
    duplicate_edges = 0
    self_loops = 0
    seen_edges = set()

    for edge in graph.edges:
        pair = (edge.source, edge.target)

        if pair in seen_edges:
            duplicate_edges += 1
        seen_edges.add(pair)

        if edge.source == edge.target:
            self_loops += 1

        edge_pairs.append(pair)

        if edge.source in degree:
            degree[edge.source] += 1
        if edge.target in degree:
            degree[edge.target] += 1

    degree_values = list(degree.values())

    node_count = len(graph.nodes)
    edge_count = len(graph.edges)

    average_degree = (
        sum(degree_values) / node_count
        if node_count > 0
        else 0.0
    )

    min_degree = min(degree_values) if degree_values else 0
    max_degree = max(degree_values) if degree_values else 0

    isolated_nodes = [
        node_id for node_id, value in degree.items()
        if value == 0
    ]

    graph_density = _compute_graph_density(
        node_count=node_count,
        edge_count=edge_count,
    )

    connected_components = _count_connected_components(
        node_ids=node_ids,
        edge_pairs=edge_pairs,
    )

    feature_dimension = (
        len(graph.nodes[0].features)
        if graph.nodes
        else 0
    )

    label_distribution = _label_distribution(graph)
    surface_distribution = _metadata_distribution(graph, "surface_type")
    density_distribution = _metadata_distribution(graph, "density_label")

    return GraphStatistics(
        graph_id=graph.graph_id,
        node_count=node_count,
        edge_count=edge_count,
        average_degree=average_degree,
        min_degree=min_degree,
        max_degree=max_degree,
        graph_density=graph_density,
        connected_components=connected_components,
        isolated_nodes=isolated_nodes,
        feature_dimension=feature_dimension,
        label_distribution=label_distribution,
        surface_distribution=surface_distribution,
        density_distribution=density_distribution,
        duplicate_edges=duplicate_edges,
        self_loops=self_loops,
    )


def _compute_graph_density(node_count: int, edge_count: int) -> float:
    if node_count <= 1:
        return 0.0

    possible_directed_edges = node_count * (node_count - 1)

    return edge_count / possible_directed_edges


def _count_connected_components(
    node_ids: list[int],
    edge_pairs: list[tuple[int, int]],
) -> int:
    if not node_ids:
        return 0

    adjacency: dict[int, set[int]] = {node_id: set() for node_id in node_ids}

    for source, target in edge_pairs:
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)

    visited = set()
    components = 0

    for node_id in node_ids:
        if node_id in visited:
            continue

        components += 1
        queue = deque([node_id])
        visited.add(node_id)

        while queue:
            current = queue.popleft()

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return components


def _label_distribution(graph: CADGraph) -> dict[int, int]:
    labels = [
        int(node.label)
        for node in graph.nodes
        if node.label is not None
    ]

    return dict(sorted(Counter(labels).items()))


def _metadata_distribution(
    graph: CADGraph,
    key: str,
) -> dict[str, int]:
    values = []

    for node in graph.nodes:
        value = node.metadata.get(key)

        if value is not None:
            values.append(str(value))

    return dict(sorted(Counter(values).items()))


def graph_statistics_summary(graph: CADGraph) -> dict[str, Any]:
    return compute_graph_statistics(graph).to_dict()


__all__ = [
    "GraphStatistics",
    "compute_graph_statistics",
    "graph_statistics_summary",
]