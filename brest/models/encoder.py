"""Spatial encoder: local GAT message passing + global Transformer self-attention.

This is the locked two-stage backbone shared by bulk / Visium / Xenium models:

    stage A (local) : GATLayer x N  -- attention restricted to Delaunay edges,
                      so each cell aggregates from its tissue neighbourhood.
    stage B (global): a Transformer encoder over all cells of each graph (padded
                      + masked per graph), giving whole-graph context.

Coordinates are min-max normalised per graph before they enter the model.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter, to_dense_batch


def normalize_positions(pos: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
    """Per-graph min-max normalise coordinates to [0, 1] (pixel scale varies)."""
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    lo = scatter(pos, batch, dim=0, reduce="min")[batch]
    hi = scatter(pos, batch, dim=0, reduce="max")[batch]
    return (pos - lo) / (hi - lo).clamp_min(1e-6)


class GATLayer(nn.Module):
    """Pre-norm GAT block: local graph attention + residual, then FFN + residual."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by heads {heads}")
        self.conv = GATConv(dim, dim // heads, heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.conv(self.norm1(x), edge_index))
        x = x + self.ffn(self.norm2(x))
        return x


class SpatialEncoder(nn.Module):
    """Two-stage per-cell encoder: local GAT then a global Transformer encoder."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        use_local: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GATLayer(dim, heads, dropout) for _ in range(num_layers)] if use_local else []
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:                       # stage A: local Delaunay GAT
            h = layer(h, edge_index)
        dense, mask = to_dense_batch(h, batch)          # stage B: per-graph global attention
        dense = self.global_encoder(dense, src_key_padding_mask=~mask)
        return dense[mask]
