import pytest

from src.graph.graph import CADGraph, GraphEdge, GraphNode


def test_graph_node_defaults():
    node = GraphNode(node_id=1)

    assert node.node_id == 1
    assert node.face_id is None
    assert node.features == []
    assert node.label is None
    assert node.metadata == {}


def test_graph_edge_defaults_and_tuple():
    edge = GraphEdge(source=1, target=2)

    assert edge.source == 1
    assert edge.target == 2
    assert edge.features == []
    assert edge.metadata == {}
    assert edge.as_tuple() == (1, 2)


def test_cad_graph_defaults():
    graph = CADGraph()

    assert graph.graph_id is None
    assert graph.nodes == []
    assert graph.edges == []
    assert graph.metadata == {}
    assert graph.node_count() == 0
    assert graph.edge_count() == 0


def test_add_node():
    graph = CADGraph()
    node = GraphNode(node_id=0, face_id=10)

    graph.add_node(node)

    assert graph.node_count() == 1
    assert graph.has_node(0) is True
    assert graph.get_node(0) is node


def test_add_duplicate_node_raises():
    graph = CADGraph()

    graph.add_node(GraphNode(node_id=0))

    with pytest.raises(ValueError, match="Duplicate node ID"):
        graph.add_node(GraphNode(node_id=0))


def test_get_missing_node_raises():
    graph = CADGraph()

    with pytest.raises(KeyError, match="Node not found"):
        graph.get_node(99)


def test_add_edge_between_existing_nodes():
    graph = CADGraph()

    graph.add_node(GraphNode(node_id=0))
    graph.add_node(GraphNode(node_id=1))

    edge = GraphEdge(source=0, target=1)
    graph.add_edge(edge)

    assert graph.edge_count() == 1
    assert graph.edges[0] is edge
    assert graph.edge_index() == [(0, 1)]


def test_add_edge_with_missing_source_raises():
    graph = CADGraph()
    graph.add_node(GraphNode(node_id=1))

    with pytest.raises(ValueError, match="Source node does not exist"):
        graph.add_edge(GraphEdge(source=0, target=1))


def test_add_edge_with_missing_target_raises():
    graph = CADGraph()
    graph.add_node(GraphNode(node_id=0))

    with pytest.raises(ValueError, match="Target node does not exist"):
        graph.add_edge(GraphEdge(source=0, target=1))


def test_node_feature_matrix_and_labels():
    graph = CADGraph(graph_id="test_graph")

    graph.add_node(GraphNode(node_id=0, features=[1.0, 2.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[3.0, 4.0], label=1))

    assert graph.node_feature_matrix() == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]

    assert graph.labels() == [0, 1]


def test_graph_summary_empty_graph():
    graph = CADGraph(graph_id="empty")

    summary = graph.summary()

    assert summary == {
        "graph_id": "empty",
        "node_count": 0,
        "edge_count": 0,
        "has_node_features": True,
        "has_labels": True,
    }


def test_graph_summary_complete_graph():
    graph = CADGraph(graph_id="complete")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=1))
    graph.add_edge(GraphEdge(source=0, target=1))

    summary = graph.summary()

    assert summary["graph_id"] == "complete"
    assert summary["node_count"] == 2
    assert summary["edge_count"] == 1
    assert summary["has_node_features"] is True
    assert summary["has_labels"] is True


def test_graph_summary_detects_missing_features_and_labels():
    graph = CADGraph(graph_id="partial")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1))

    summary = graph.summary()

    assert summary["has_node_features"] is False
    assert summary["has_labels"] is False