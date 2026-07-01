from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from src.graph.graph import CADGraph, GraphEdge, GraphNode
from src.graph.graph_visualizer import (
    circular_layout,
    draw_cad_graph,
    graph_to_adjacency_dict,
    graph_visualization_summary,
    save_cad_graph_figure,
)


def make_sample_graph():
    graph = CADGraph(graph_id="sample_graph")

    graph.add_node(GraphNode(node_id=0, face_id=10, features=[1.0], label=0))
    graph.add_node(GraphNode(node_id=1, face_id=20, features=[2.0], label=1))
    graph.add_node(GraphNode(node_id=2, face_id=30, features=[3.0], label=2))

    graph.add_edge(GraphEdge(source=0, target=1))
    graph.add_edge(GraphEdge(source=1, target=2))

    return graph


def test_circular_layout_returns_positions():
    graph = make_sample_graph()

    positions = circular_layout(graph)

    assert isinstance(positions, dict)
    assert set(positions.keys()) == {0, 1, 2}

    for value in positions.values():
        assert isinstance(value, tuple)
        assert len(value) == 2
        assert isinstance(value[0], float)
        assert isinstance(value[1], float)


def test_circular_layout_empty_graph():
    graph = CADGraph(graph_id="empty")

    positions = circular_layout(graph)

    assert positions == {}


def test_circular_layout_rejects_none_graph():
    with pytest.raises(ValueError, match="CADGraph cannot be None"):
        circular_layout(None)


def test_draw_cad_graph_returns_figure():
    graph = make_sample_graph()

    fig = draw_cad_graph(graph)

    assert fig is not None
    assert hasattr(fig, "savefig")

    plt.close(fig)


def test_draw_cad_graph_without_labels():
    graph = make_sample_graph()

    fig = draw_cad_graph(
        graph,
        title="No Labels",
        show_labels=False,
    )

    assert fig is not None

    plt.close(fig)


def test_draw_cad_graph_empty_graph():
    graph = CADGraph(graph_id="empty")

    fig = draw_cad_graph(graph)

    assert fig is not None

    plt.close(fig)


def test_draw_cad_graph_rejects_none_graph():
    with pytest.raises(ValueError, match="CADGraph cannot be None"):
        draw_cad_graph(None)


def test_save_cad_graph_figure(tmp_path):
    graph = make_sample_graph()

    output_path = tmp_path / "figures" / "graph.png"

    saved_path = save_cad_graph_figure(
        graph=graph,
        output_path=output_path,
        title="Saved Graph",
    )

    assert isinstance(saved_path, Path)
    assert saved_path.exists()
    assert saved_path.suffix == ".png"


def test_graph_to_adjacency_dict():
    graph = make_sample_graph()

    adjacency = graph_to_adjacency_dict(graph)

    assert adjacency == {
        0: [1],
        1: [2],
        2: [],
    }


def test_graph_to_adjacency_dict_empty_graph():
    graph = CADGraph(graph_id="empty")

    adjacency = graph_to_adjacency_dict(graph)

    assert adjacency == {}


def test_graph_to_adjacency_dict_rejects_none_graph():
    with pytest.raises(ValueError, match="CADGraph cannot be None"):
        graph_to_adjacency_dict(None)


def test_graph_visualization_summary():
    graph = make_sample_graph()

    summary = graph_visualization_summary(graph)

    assert summary == {
        "graph_id": "sample_graph",
        "node_count": 3,
        "edge_count": 2,
        "adjacency": {
            0: [1],
            1: [2],
            2: [],
        },
    }


def test_graph_visualization_summary_rejects_none_graph():
    with pytest.raises(ValueError, match="CADGraph cannot be None"):
        graph_visualization_summary(None)