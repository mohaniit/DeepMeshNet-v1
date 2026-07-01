"""
Base model interface for DeepMeshNet-v1 Model Zoo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """
    Common model configuration.
    """

    in_channels: int
    hidden_channels: int
    out_channels: int
    dropout: float = 0.0

    def validate(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")

        if self.out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.out_channels,
            "dropout": self.dropout,
        }


class BaseGraphModel(nn.Module, ABC):
    """
    Abstract base class for DeepMeshNet graph models.
    """

    model_name: str = "base_graph_model"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if config is None:
            raise ValueError("ModelConfig cannot be None.")

        config.validate()

        self.config = config

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Expected output shape:
            [num_nodes, out_channels]
        """

    def predict_logits(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return raw logits.
        """
        self.eval()

        with torch.no_grad():
            return self.forward(x, edge_index)

    def predict_classes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return predicted class IDs.
        """
        logits = self.predict_logits(x, edge_index)
        return torch.argmax(logits, dim=1)

    def num_parameters(self) -> int:
        """
        Return number of trainable parameters.
        """
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def model_summary(self) -> dict[str, Any]:
        """
        Return model summary.
        """
        return {
            "model_name": self.model_name,
            "config": self.config.to_dict(),
            "num_parameters": self.num_parameters(),
        }


__all__ = [
    "ModelConfig",
    "BaseGraphModel",
]