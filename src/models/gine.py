"""
GINE model for DeepMeshNet-v1.

This module implements a two-layer edge-aware Graph Isomorphism Network
for node-level CAD face mesh-density classification.

Unlike GIN, GINE uses edge attributes through GINEConv.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv

from src.models.base_model import BaseGraphModel, ModelConfig


class GINE(BaseGraphModel):
    """
    Edge-aware Graph Isomorphism Network model.
    """

    model_name = "gine"

    def __init__(
        self,
        config: ModelConfig,
        edge_dim: int,
    ) -> None:
        super().__init__(config)

        if edge_dim <= 0:
            raise ValueError("edge_dim must be positive.")

        self.edge_dim = edge_dim

        self.mlp1 = nn.Sequential(
            nn.Linear(config.in_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, config.hidden_channels),
        )

        self.conv1 = GINEConv(
            nn=self.mlp1,
            edge_dim=edge_dim,
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, config.out_channels),
        )

        self.conv2 = GINEConv(
            nn=self.mlp2,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        edge_attr is required for GINE.
        """
        if edge_attr is None:
            raise ValueError("GINE requires edge_attr.")

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.config.dropout,
            training=self.training,
        )
        x = self.conv2(x, edge_index, edge_attr)

        return x

    def predict_logits(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return raw logits.
        """
        self.eval()

        with torch.no_grad():
            return self.forward(x, edge_index, edge_attr)

    def predict_classes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return predicted class IDs.
        """
        logits = self.predict_logits(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return torch.argmax(logits, dim=1)

    def model_summary(self) -> dict:
        """
        Return model summary.
        """
        summary = super().model_summary()
        summary["edge_dim"] = self.edge_dim
        return summary


def create_gine(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    edge_dim: int,
    dropout: float = 0.0,
) -> GINE:
    """
    Factory function for GINE.
    """
    config = ModelConfig(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    )

    return GINE(
        config=config,
        edge_dim=edge_dim,
    )


__all__ = [
    "GINE",
    "create_gine",
]