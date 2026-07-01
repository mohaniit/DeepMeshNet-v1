"""
CAD Graph Validator for DeepMeshNet-v1.

This module validates CADGraph integrity before conversion to
PyTorch Geometric or other graph-learning frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.graph.graph import CADGraph


@dataclass
class GraphValidationResult:
    """
    Result of graph validation.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


class GraphValidator:
    """
    Validate CADGraph integrity.
    """

    def validate(self, graph: CADGraph) -> GraphValidationResult:
        """
        Validate a CADGraph.

        Parameters
        ----------
        graph
            Graph to validate.

        Returns
        -------
        GraphValidationResult
        """
        result = GraphValidationResult()

        if graph is None:
            result.add_error("Graph cannot be None.")
            return result

        self._validate_nodes(graph, result)
        self._validate_edges(graph, result)
        self._validate_feature_dimensions(graph, result)
        self._validate_labels(graph, result)
        self._validate_isolated_nodes(graph, result)

        return result

    def _validate_nodes(
        self,
        graph: CADGraph,
        result: GraphValidationResult,
    ) -> None:
        """
        Validate graph nodes.
        """
        if graph.node_count() == 0:
            result.add_error("Graph contains no nodes.")
            return

        node_ids = set()

        for node in graph.nodes:
            if node.node_id in node_ids:
                result.add_error(
                    f"Duplicate node ID detected: {node.node_id}"
                )

            node_ids.add(node.node_id)

    def _validate_edges(
        self,
        graph: CADGraph,
        result: GraphValidationResult,
    ) -> None:
        """
        Validate graph edges.
        """
        node_ids = {node.node_id for node in graph.nodes}

        directed_edges = set()

        for edge in graph.edges:

            if edge.source not in node_ids:
                result.add_error(
                    f"Edge source node does not exist: {edge.source}"
                )

            if edge.target not in node_ids:
                result.add_error(
                    f"Edge target node does not exist: {edge.target}"
                )

            if edge.source == edge.target:
                result.add_error(
                    f"Self-loop detected at node {edge.source}"
                )

            pair = (edge.source, edge.target)

            if pair in directed_edges:
                result.add_error(
                    f"Duplicate directed edge: {pair}"
                )

            directed_edges.add(pair)

    def _validate_feature_dimensions(
        self,
        graph: CADGraph,
        result: GraphValidationResult,
    ) -> None:
        """
        Validate node feature dimensions.
        """
        if graph.node_count() == 0:
            return

        expected_dimension = len(graph.nodes[0].features)

        for node in graph.nodes:

            if len(node.features) != expected_dimension:
                result.add_error(
                    f"Node {node.node_id} has feature dimension "
                    f"{len(node.features)} "
                    f"(expected {expected_dimension})."
                )

    def _validate_labels(
        self,
        graph: CADGraph,
        result: GraphValidationResult,
    ) -> None:
        """
        Validate node labels.
        """
        for node in graph.nodes:

            if node.label is None:
                result.add_warning(
                    f"Node {node.node_id} has no label."
                )

    def _validate_isolated_nodes(
        self,
        graph: CADGraph,
        result: GraphValidationResult,
    ) -> None:
        """
        Detect isolated nodes.
        """
        connected = set()

        for edge in graph.edges:
            connected.add(edge.source)
            connected.add(edge.target)

        for node in graph.nodes:
            if node.node_id not in connected:
                result.add_warning(
                    f"Node {node.node_id} is isolated."
                )


def validate_graph(graph: CADGraph) -> GraphValidationResult:
    """
    Convenience function.
    """
    return GraphValidator().validate(graph)


__all__ = [
    "GraphValidationResult",
    "GraphValidator",
    "validate_graph",
]