"""
GAT model for DeepMeshNet-v1.

This module implements a two-layer Graph Attention Network for
node-level CAD face mesh-density classification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.models.base_model import BaseGraphModel, ModelConfig


class GAT(BaseGraphModel):
    """
    Graph Attention Network model.
    """

    model_name = "gat"

    def __init__(
        self,
        config: ModelConfig,
        heads: int = 4,
    ) -> None:
        super().__init__(config)

        if heads <= 0:
            raise ValueError("heads must be positive.")

        self.heads = heads

        self.conv1 = GATConv(
            in_channels=config.in_channels,
            out_channels=config.hidden_channels,
            heads=heads,
            dropout=config.dropout,
        )

        self.conv2 = GATConv(
            in_channels=config.hidden_channels * heads,
            out_channels=config.out_channels,
            heads=1,
            concat=False,
            dropout=config.dropout,
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
        x = F.elu(x)
        x = F.dropout(
            x,
            p=self.config.dropout,
            training=self.training,
        )
        x = self.conv2(x, edge_index)

        return x

    def model_summary(self) -> dict:
        """
        Return model summary.
        """
        summary = super().model_summary()
        summary["heads"] = self.heads
        return summary


def create_gat(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float = 0.0,
    heads: int = 4,
) -> GAT:
    """
    Factory function for GAT.
    """
    config = ModelConfig(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    )

    return GAT(
        config=config,
        heads=heads,
    )


__all__ = [
    "GAT",
    "create_gat",
]