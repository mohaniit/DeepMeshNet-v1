"""
GCN model for DeepMeshNet-v1.

This module implements a two-layer Graph Convolutional Network for
node-level CAD face mesh-density classification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.models.base_model import BaseGraphModel, ModelConfig


class GCN(BaseGraphModel):
    """
    Graph Convolutional Network model.
    """

    model_name = "gcn"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        self.conv1 = GCNConv(
            in_channels=config.in_channels,
            out_channels=config.hidden_channels,
        )

        self.conv2 = GCNConv(
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


def create_gcn(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float = 0.0,
) -> GCN:
    """
    Factory function for GCN.
    """
    config = ModelConfig(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    )

    return GCN(config)


__all__ = [
    "GCN",
    "create_gcn",
]