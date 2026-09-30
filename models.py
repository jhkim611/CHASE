"""GCN-GRU backbone used by CHASE."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelDefaults:
    num_epochs: int
    lr: float
    hidden_channels: int
    dropout: float
    eval_every: int
    patience: int
    weight_decay: float


class BaseTemporalModel(nn.Module):
    def reset_state(self) -> None:
        return

    def detach_state(self) -> None:
        return


class GCNGRU(BaseTemporalModel):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 1):
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv
        except ImportError as e:
            raise ImportError(
                "GCN-GRU requires torch_geometric (+ torch_sparse). "
                "Install torch_geometric dependencies before running."
            ) from e
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.h: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self.h = None

    def detach_state(self) -> None:
        if self.h is not None:
            self.h = self.h.detach()

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        z = self.gcn(x, edge_index, edge_weight=edge_attr)
        z = F.relu(z)
        if self.h is None:
            self.h = torch.zeros_like(z)
        self.h = self.gru(z, self.h)
        return self.h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.linear(self.encode(x, edge_index, edge_attr))


def model_defaults() -> ModelDefaults:
    return ModelDefaults(
        num_epochs=500, lr=1e-2, hidden_channels=128, dropout=0.0,
        eval_every=5, patience=10, weight_decay=0.0,
    )


def build_model(in_channels: int, hidden_channels: int, num_classes: int = 2) -> BaseTemporalModel:
    out_channels = 1 if num_classes <= 2 else num_classes
    return GCNGRU(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )
