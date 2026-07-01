import pytest

from src.graph.edge_features import (
    EDGE_FEATURE_NAMES,
    EdgeFeatureExtractor,
    EdgeFeatureVector,
    attach_edge_features,
    extract_edge_feature_vector,
)
from src.graph.graph import CADGraph, GraphEdge, GraphNode


def make_graph():
    graph = CADGraph(graph_id="edge_feature_graph")

    graph.add_node(GraphNode(node_id=0, features=[1.0, 0.0], label=0))
    graph.add_node(GraphNode(node_id=1, features=[0.0, 1.0], label=1))
    graph.add_node(GraphNode(node_id=2, features=[1.0, 1.0], label=2))

    graph.add_edge(GraphEdge(source=0, target=1))
    graph.add_edge(GraphEdge(source=1, target=0))
    graph.add_edge(GraphEdge(source=1, target=2))

    return graph


def test_edge_feature_vector_to_dict():
    vector = EdgeFeatureVector(x=[1.0, 2.0, 1.0, 0.5, 1.0])

    assert vector.to_dict() == {
        "x": [1.0, 2.0, 1.0, 0.5, 1.0],
        "feature_names": EDGE_FEATURE_NAMES,
    }


def test_extract_edge_features():
    graph = make_graph()
    edge = graph.edges[0]

    vector = EdgeFeatureExtractor().extract_edge_features(graph, edge)

    assert isinstance(vector, EdgeFeatureVector)
    assert vector.feature_names == EDGE_FEATURE_NAMES
    assert len(vector.x) == len(EDGE_FEATURE_NAMES)

    assert vector.x[0] == 2.0
    assert vector.x[1] == 3.0
    assert vector.x[2] == 1.0
    assert vector.x[3] == pytest.approx(2 ** 0.5)
    assert vector.x[4] == 1.0


def test_attach_edge_features():
    graph = make_graph()

    EdgeFeatureExtractor().attach_edge_features(graph)

    for edge in graph.edges:
        assert len(edge.features) == len(EDGE_FEATURE_NAMES)
        assert edge.metadata["edge_feature_names"] == EDGE_FEATURE_NAMES


def test_attach_edge_features_overwrite_false_keeps_existing_features():
    graph = make_graph()

    graph.edges[0].features = [99.0]

    EdgeFeatureExtractor().attach_edge_features(
        graph,
        overwrite=False,
    )

    assert graph.edges[0].features == [99.0]

    for edge in graph.edges[1:]:
        assert len(edge.features) == len(EDGE_FEATURE_NAMES)


def test_attach_edge_features_overwrite_true_replaces_existing_features():
    graph = make_graph()

    graph.edges[0].features = [99.0]

    EdgeFeatureExtractor().attach_edge_features(
        graph,
        overwrite=True,
    )

    assert graph.edges[0].features != [99.0]
    assert len(graph.edges[0].features) == len(EDGE_FEATURE_NAMES)


def test_extract_edge_feature_vector_convenience_function():
    graph = make_graph()
    edge = graph.edges[0]

    vector = extract_edge_feature_vector(graph, edge)

    assert isinstance(vector, EdgeFeatureVector)
    assert len(vector.x) == len(EDGE_FEATURE_NAMES)


def test_attach_edge_features_convenience_function():
    graph = make_graph()

    returned = attach_edge_features(graph)

    assert returned is graph

    for edge in graph.edges:
        assert len(edge.features) == len(EDGE_FEATURE_NAMES)


def test_extract_rejects_none_graph():
    edge = GraphEdge(source=0, target=1)

    with pytest.raises(ValueError, match="CADGraph cannot be None"):
        EdgeFeatureExtractor().extract_edge_features(None, edge)


def test_extract_rejects_none_edge():
    graph = make_graph()

    with pytest.raises(ValueError, match="GraphEdge cannot be None"):
        EdgeFeatureExtractor().extract_edge_features(graph, None)


def test_attach_rejects_none_graph():
    with pytest.raises(ValueError, match="CADGraph cannot be None"):
        EdgeFeatureExtractor().attach_edge_features(None)


def test_feature_distance_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        EdgeFeatureExtractor._feature_distance(
            [1.0, 2.0],
            [1.0],
        )


def test_degree_for_isolated_node():
    graph = CADGraph(graph_id="isolated")

    graph.add_node(GraphNode(node_id=0, features=[1.0], label=0))

    degree = EdgeFeatureExtractor._degree(graph, 0)

    assert degree == 0