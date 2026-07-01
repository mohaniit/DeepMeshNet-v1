"""
GraphSAGE model for DeepMeshNet-v1.

This module implements a two-layer GraphSAGE model for node-level
CAD face mesh-density classification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.models.base_model import BaseGraphModel, ModelConfig


class GraphSAGE(BaseGraphModel):
    """
    GraphSAGE model.
    """

    model_name = "graphsage"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        self.conv1 = SAGEConv(
            in_channels=config.in_channels,
            out_channels=config.hidden_channels,
        )

        self.conv2 = SAGEConv(
            in_channels=config.hidden_channels,
            out_channels=config.out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.config.dropout,
            training=self.training,
        )
        x = self.conv2(x, edge_index)

        return x


def create_graphsage(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float = 0.0,
) -> GraphSAGE:
    """
    Factory function for GraphSAGE.
    """
    config = ModelConfig(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    )

    return GraphSAGE(config)


__all__ = [
    "GraphSAGE",
    "create_graphsage",
]