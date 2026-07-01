"""
GIN model for DeepMeshNet-v1.

This module implements a two-layer Graph Isomorphism Network for
node-level CAD face mesh-density classification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv

from src.models.base_model import BaseGraphModel, ModelConfig


class GIN(BaseGraphModel):
    """
    Graph Isomorphism Network model.
    """

    model_name = "gin"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        self.mlp1 = nn.Sequential(
            nn.Linear(config.in_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, config.hidden_channels),
        )

        self.conv1 = GINConv(self.mlp1)

        self.mlp2 = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, config.out_channels),
        )

        self.conv2 = GINConv(self.mlp2)

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


def create_gin(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float = 0.0,
) -> GIN:
    """
    Factory function for GIN.
    """
    config = ModelConfig(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    )

    return GIN(config)


__all__ = [
    "GIN",
    "create_gin",
]