import pytest

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.graph.graph_statistics import (
    GraphStatistics,
    compute_graph_statistics,
    graph_statistics_summary,
)


def test_graph_statistics_dataclass_to_dict():
    stats = GraphStatistics(
        graph_id="g1",
        node_count=2,
        edge_count=2,
        average_degree=2.0,
        min_degree=2,
        max_degree=2,
        graph_density=1.0,
        connected_components=1,
        isolated_nodes=[],
        feature_dimension=3,
        label_distribution={0: 1, 1: 1},
        surface_distribution={"PLANE": 1, "CYLINDER": 1},
        density_distribution={"LOW": 1, "HIGH": 1},
        duplicate_edges=0,
        self_loops=0,
    )

    result = stats.to_dict()

    assert result["graph_id"] == "g1"
    assert result["node_count"] == 2
    assert result["edge_count"] == 2
    assert result["average_degree"] == 2.0
    assert result["is_connected"] is True
    assert result["isolated_node_count"] == 0
    assert result["feature_dimension"] == 3
    assert result["label_distribution"] == {0: 1, 1: 1}
    assert result["surface_distribution"] == {"PLANE": 1, "CYLINDER": 1}
    assert result["density_distribution"] == {"LOW": 1, "HIGH": 1}


def test_graph_statistics_is_connected_false_for_empty():
    stats = GraphStatistics(
        graph_id="empty",
        node_count=0,
        edge_count=0,
        average_degree=0.0,
        min_degree=0,
        max_degree=0,
        graph_density=0.0,
        connected_components=0,
    )

    assert stats.is_connected is False


def test_compute_graph_statistics_valid_bidirectional_graph():
    graph = CADGraph(graph_id="valid_graph")

    graph.add_node(
        GraphNode(
            node_id=0,
            features=[1.0, 2.0],
            label=0,
            metadata={
                "surface_type": "PLANE",
                "density_label": "LOW",
            },
        )
    )
    graph.add_node(
        GraphNode(
            node_id=1,
            features=[3.0, 4.0],
            label=1,
            metadata={
                "surface_type": "CYLINDER",
                "density_label": "HIGH",
            },
        )
    )

    graph.add_edge(GraphEdge(source=0, target=1))
    graph.add_edge(GraphEdge(source=1, target=0))

    stats = compute_graph_statistics(graph)

    assert stats.graph_id == "valid_graph"
    assert stats.node_count == 2
    assert stats.edge_count == 2
    assert stats.average_degree == 2.0
    assert stats.min_degree == 2
    assert stats.max_degree == 2
    assert stats.graph_density == 1.0
    assert stats.connected_components == 1
    assert stats.is_connected is True
    assert stats.isolated_nodes == []
    assert stats.feature_dimension == 2
    assert stats.label_distribution == {0: 1, 1: 1}
    assert stats.surface_distribution == {"CYLINDER": 1, "PLANE": 1}
    assert stats.density_distribution == {"HIGH": 1, "LOW": 1}
    assert stats.duplicate_edges == 0
    assert stats.self_loops == 0


def test_compute_graph_statistics_empty_graph():
    graph = CADGraph(graph_id="empty")

    stats = compute_graph_statistics(graph)

    assert stats.node_count == 0
    assert stats.edge_count == 0
    assert stats.average_degree == 0.0
    assert stats.min_degree == 0
    assert stats.max_degree == 0
    assert stats.graph_density == 0.0
    assert stats.connected_components == 0
    assert stats.is_connected is False
    assert stats.isolated_nodes == []
    assert stats.feature_dimension == 0


def test_compute_graph_statistics_isolated_nodes():
    graph = CADGraph(graph_id="isolated")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=1))

    stats = compute_graph_statistics(graph)

    assert stats.node_count == 2
    assert stats.edge_count == 0
    assert stats.isolated_nodes == [0, 1]
    assert stats.connected_components == 2
    assert stats.is_connected is False


def test_compute_graph_statistics_connected_components():
    graph = CADGraph(graph_id="components")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=2, features=[1.0], label=1))
    graph.add_node(GraphNode(node_id=3, features=[1.0], label=1))

    graph.add_edge(GraphEdge(source=0, target=1))
    graph.add_edge(GraphEdge(source=2, target=3))

    stats = compute_graph_statistics(graph)

    assert stats.connected_components == 2
    assert stats.is_connected is False
    assert stats.isolated_nodes == []


def test_compute_graph_statistics_duplicate_edges_and_self_loops():
    graph = CADGraph(graph_id="quality")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=1))

    graph.edges.append(GraphEdge(source=0, target=1))
    graph.edges.append(GraphEdge(source=0, target=1))
    graph.edges.append(GraphEdge(source=1, target=1))

    stats = compute_graph_statistics(graph)

    assert stats.duplicate_edges == 1
    assert stats.self_loops == 1


def test_compute_graph_statistics_missing_labels_ignored():
    graph = CADGraph(graph_id="labels")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=None))

    stats = compute_graph_statistics(graph)

    assert stats.label_distribution == {0: 1}


def test_compute_graph_statistics_metadata_missing_ignored():
    graph = CADGraph(graph_id="metadata")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(
        GraphNode(
            node_id=1,
            features=[2.0],
            label=1,
            metadata={
                "surface_type": "TORUS",
                "density_label": "VERY_HIGH",
            },
        )
    )

    stats = compute_graph_statistics(graph)

    assert stats.surface_distribution == {"TORUS": 1}
    assert stats.density_distribution == {"VERY_HIGH": 1}


def test_compute_graph_statistics_rejects_none_graph():
    with pytest.raises(ValueError, match="CADGraph cannot be None"):
        compute_graph_statistics(None)


def test_graph_statistics_summary():
    graph = CADGraph(graph_id="summary")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=1))
    graph.add_edge(GraphEdge(source=0, target=1))

    summary = graph_statistics_summary(graph)

    assert summary["graph_id"] == "summary"
    assert summary["node_count"] == 2
    assert summary["edge_count"] == 1
    assert summary["connected_components"] == 1
    assert summary["label_distribution"] == {0: 1, 1: 1}