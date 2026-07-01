"""
CAD graph visualization utilities for DeepMeshNet-v1.

This module visualizes the neutral CADGraph representation.
It does not depend on PyTorch Geometric.

The visualizations are intended for debugging, reports, and publication
figures of CAD face-adjacency graphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.graph.graph import CADGraph


def circular_layout(graph: CADGraph) -> dict[int, tuple[float, float]]:
    """
    Compute a simple circular layout for CADGraph nodes.

    This avoids requiring NetworkX while keeping the visualization
    lightweight and deterministic.
    """
    if graph is None:
        raise ValueError("CADGraph cannot be None.")

    import math

    node_count = graph.node_count()

    if node_count == 0:
        return {}

    positions: dict[int, tuple[float, float]] = {}

    for index, node in enumerate(graph.nodes):
        angle = 2.0 * math.pi * index / node_count
        positions[node.node_id] = (
            math.cos(angle),
            math.sin(angle),
        )

    return positions


def draw_cad_graph(
    graph: CADGraph,
    title: str | None = None,
    show_labels: bool = True,
    node_size: int = 600,
) -> plt.Figure:
    """
    Draw a CADGraph using matplotlib.

    Parameters
    ----------
    graph:
        CADGraph instance.
    title:
        Optional plot title.
    show_labels:
        If True, show node IDs.
    node_size:
        Node marker size.

    Returns
    -------
    matplotlib.figure.Figure
        Created figure.
    """
    if graph is None:
        raise ValueError("CADGraph cannot be None.")

    positions = circular_layout(graph)

    fig, ax = plt.subplots(figsize=(6, 6))

    for edge in graph.edges:
        if edge.source not in positions or edge.target not in positions:
            continue

        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=1.0,
            alpha=0.7,
        )

    x_values = []
    y_values = []

    for node in graph.nodes:
        x, y = positions[node.node_id]
        x_values.append(x)
        y_values.append(y)

    if x_values and y_values:
        ax.scatter(
            x_values,
            y_values,
            s=node_size,
            zorder=3,
        )

    if show_labels:
        for node in graph.nodes:
            x, y = positions[node.node_id]
            label = str(node.face_id if node.face_id is not None else node.node_id)

            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=9,
                zorder=4,
            )

    ax.set_aspect("equal")
    ax.axis("off")

    if title is None:
        title = graph.graph_id or "CAD Graph"

    ax.set_title(title)

    fig.tight_layout()

    return fig


def save_cad_graph_figure(
    graph: CADGraph,
    output_path: str | Path,
    title: str | None = None,
    show_labels: bool = True,
    dpi: int = 300,
) -> Path:
    """
    Save CADGraph figure to disk.

    Parameters
    ----------
    graph:
        CADGraph instance.
    output_path:
        Output image path.
    title:
        Optional figure title.
    show_labels:
        If True, show node labels.
    dpi:
        Figure DPI.

    Returns
    -------
    Path
        Saved file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = draw_cad_graph(
        graph=graph,
        title=title,
        show_labels=show_labels,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def graph_to_adjacency_dict(graph: CADGraph) -> dict[int, list[int]]:
    """
    Convert graph to adjacency dictionary.
    """
    if graph is None:
        raise ValueError("CADGraph cannot be None.")

    adjacency: dict[int, set[int]] = {
        node.node_id: set()
        for node in graph.nodes
    }

    for edge in graph.edges:
        if edge.source in adjacency and edge.target in adjacency:
            adjacency[edge.source].add(edge.target)

    return {
        node_id: sorted(neighbors)
        for node_id, neighbors in adjacency.items()
    }


def graph_visualization_summary(graph: CADGraph) -> dict[str, Any]:
    """
    Return summary useful for visualization/reporting.
    """
    if graph is None:
        raise ValueError("CADGraph cannot be None.")

    adjacency = graph_to_adjacency_dict(graph)

    return {
        "graph_id": graph.graph_id,
        "node_count": graph.node_count(),
        "edge_count": graph.edge_count(),
        "adjacency": adjacency,
    }


__all__ = [
    "circular_layout",
    "draw_cad_graph",
    "save_cad_graph_figure",
    "graph_to_adjacency_dict",
    "graph_visualization_summary",
]