from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.graph.graph_validator import (
    GraphValidationResult,
    GraphValidator,
    validate_graph,
)


def test_graph_validation_result_defaults():
    result = GraphValidationResult()

    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []


def test_graph_validation_result_add_error():
    result = GraphValidationResult()

    result.add_error("error message")

    assert result.is_valid is False
    assert result.errors == ["error message"]


def test_graph_validation_result_add_warning():
    result = GraphValidationResult()

    result.add_warning("warning message")

    assert result.is_valid is True
    assert result.warnings == ["warning message"]


def test_validate_none_graph():
    result = GraphValidator().validate(None)

    assert result.is_valid is False
    assert "Graph cannot be None." in result.errors


def test_validate_empty_graph():
    graph = CADGraph(graph_id="empty")

    result = GraphValidator().validate(graph)

    assert result.is_valid is False
    assert "Graph contains no nodes." in result.errors


def test_validate_valid_graph():
    graph = CADGraph(graph_id="valid")

    graph.add_node(GraphNode(node_id=0, features=[1.0, 2.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[3.0, 4.0], label=1))
    graph.add_edge(GraphEdge(source=0, target=1))
    graph.add_edge(GraphEdge(source=1, target=0))

    result = GraphValidator().validate(graph)

    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []


def test_validate_duplicate_node_ids():
    graph = CADGraph(graph_id="duplicate_nodes")

    graph.nodes.append(GraphNode(node_id=0, features=[1.0], label=0))
    graph.nodes.append(GraphNode(node_id=0, features=[2.0], label=1))

    result = GraphValidator().validate(graph)

    assert result.is_valid is False
    assert any("Duplicate node ID detected: 0" in error for error in result.errors)


def test_validate_edge_missing_source():
    graph = CADGraph(graph_id="missing_source")

    graph.add_node(GraphNode(node_id=1, features=[1.0], label=0))
    graph.edges.append(GraphEdge(source=0, target=1))

    result = GraphValidator().validate(graph)

    assert result.is_valid is False
    assert any("Edge source node does not exist: 0" in error for error in result.errors)


def test_validate_edge_missing_target():
    graph = CADGraph(graph_id="missing_target")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.edges.append(GraphEdge(source=0, target=1))

    result = GraphValidator().validate(graph)

    assert result.is_valid is False
    assert any("Edge target node does not exist: 1" in error for error in result.errors)


def test_validate_self_loop():
    graph = CADGraph(graph_id="self_loop")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.edges.append(GraphEdge(source=0, target=0))

    result = GraphValidator().validate(graph)

    assert result.is_valid is False
    assert any("Self-loop detected at node 0" in error for error in result.errors)


def test_validate_duplicate_directed_edges():
    graph = CADGraph(graph_id="duplicate_edges")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=1))
    graph.edges.append(GraphEdge(source=0, target=1))
    graph.edges.append(GraphEdge(source=0, target=1))

    result = GraphValidator().validate(graph)

    assert result.is_valid is False
    assert any("Duplicate directed edge" in error for error in result.errors)


def test_validate_inconsistent_feature_dimensions():
    graph = CADGraph(graph_id="bad_features")

    graph.add_node(GraphNode(node_id=0, features=[1.0, 2.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[3.0], label=1))
    graph.add_edge(GraphEdge(source=0, target=1))
    graph.add_edge(GraphEdge(source=1, target=0))

    result = GraphValidator().validate(graph)

    assert result.is_valid is False
    assert any("feature dimension" in error for error in result.errors)


def test_validate_missing_label_warning():
    graph = CADGraph(graph_id="missing_label")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=None))

    result = GraphValidator().validate(graph)

    assert result.is_valid is True
    assert any("Node 0 has no label." in warning for warning in result.warnings)


def test_validate_isolated_node_warning():
    graph = CADGraph(graph_id="isolated")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[2.0], label=1))

    result = GraphValidator().validate(graph)

    assert result.is_valid is True
    assert any("Node 0 is isolated." in warning for warning in result.warnings)
    assert any("Node 1 is isolated." in warning for warning in result.warnings)


def test_validate_graph_convenience_function():
    graph = CADGraph(graph_id="convenience")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))

    result = validate_graph(graph)

    assert result.is_valid is True
    assert any("Node 0 is isolated." in warning for warning in result.warnings)