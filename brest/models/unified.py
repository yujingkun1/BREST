"""Unified bulk / Visium / Xenium expression model (architecture-locked).

ONE architecture across all three modalities, so a bulk backbone transfers 1:1
into the spatial models (shared module names + shapes -> load_state_dict works):

    H0-mini features
      -> Delaunay graph (edges precomputed in the data; hyperparameter-free)
      -> SpatialEncoder = local GAT + global Transformer
      -> gene_queries head

Granularity is the ONLY thing that differs between modalities:

    "cell"  (Xenium): per-node prediction, no pooling      -> [N_cells, G]
    "spot"  (Visium): mean-pool each spot's cells          -> [N_spots, G]
    "slide" (bulk):   mean-pool all cells of a slide        -> [N_slides, G]

Why a *modulation* gene-query head instead of an attention-pooling readout: the
attention pool collapses at per-cell granularity (a single token gives a trivial
softmax, so genes lose specificity). Modulation
``expr[m, g] = gene_readout(U[m] * q_g)`` keeps each gene's learnable query q_g,
stays per-gene addressable/transferable, and is identical across granularities.
"""
from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import nn
from torch_geometric.utils import scatter

from .encoder import SpatialEncoder, normalize_positions


class GeneQueryHead(nn.Module):
    """Modulation-style gene-query readout.

    Each gene g owns a learnable query vector ``gene_queries[g]`` of size E.
    ``expr[m, g] = gene_readout(U[m] * gene_queries[g])`` for unit embeddings U
    (per cell, per spot, or per slide -- the head does not care which). Transfer
    across panels copies ``gene_queries`` rows by gene symbol; ``gene_readout``
    loads wholesale.
    """

    def __init__(self, embed_dim: int, num_genes: int, dropout: float = 0.1,
                 gene_chunk: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.gene_chunk = int(gene_chunk)  # bound the [M, g, E] interaction tensor
        self.gene_queries = nn.Parameter(torch.empty(num_genes, embed_dim))
        nn.init.xavier_uniform_(self.gene_queries)
        self.gene_readout = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def _chunk(self, U: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        inter = U.unsqueeze(1) * q.unsqueeze(0)               # [M, g, E]
        return self.gene_readout(inter).squeeze(-1)            # [M, g]

    def forward(self, U: torch.Tensor) -> torch.Tensor:  # U: [M, E] -> [M, G]
        num_genes = self.gene_queries.size(0)
        chunk = self.gene_chunk or num_genes
        outs = []
        for start in range(0, num_genes, chunk):
            q = self.gene_queries[start:start + chunk]         # [g, E]
            # Gradient-checkpoint each gene chunk: backward recomputes it, so the
            # large [M, g, E] activations are not retained -> peak memory ~ one
            # chunk (lets the per-cell head run over a whole bulk slide's cells).
            if self.training and U.requires_grad:
                outs.append(torch.utils.checkpoint.checkpoint(self._chunk, U, q, use_reentrant=False))
            else:
                outs.append(self._chunk(U, q))
        return torch.cat(outs, dim=1)                          # [M, G]


class LinearGeneHead(nn.Module):
    """Direct multi-output readout used for the no-gene-query ablation."""

    def __init__(self, embed_dim: int, num_genes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.readout = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_genes),
        )

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        return self.readout(U)


class UnifiedExpressionModel(nn.Module):
    """The locked architecture. ``granularity`` selects cell / spot / slide readout."""

    def __init__(
        self,
        *,
        in_dim: int = 1536,
        num_genes: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        use_pos: bool = True,
        granularity: str = "cell",
        gene_chunk: int = 64,
        head_type: str = "gene_query",
        use_local: bool = True,
    ) -> None:
        super().__init__()
        if granularity not in {"cell", "spot", "slide"}:
            raise ValueError(f"granularity must be cell|spot|slide, got {granularity}")
        if head_type not in {"gene_query", "linear"}:
            raise ValueError(f"head_type must be gene_query|linear, got {head_type}")
        self.granularity = granularity
        self.head_type = head_type
        self.use_pos = bool(use_pos)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        if self.use_pos:
            self.pos_proj = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
        self.spatial = SpatialEncoder(
            hidden_dim,
            n_layers,
            heads,
            dropout,
            use_local=use_local,
        )
        if head_type == "linear":
            self.head = LinearGeneHead(hidden_dim, num_genes, dropout=dropout)
        else:
            self.head = GeneQueryHead(hidden_dim, num_genes, dropout=dropout, gene_chunk=gene_chunk)

    def encode(self, x, edge_index, pos=None, batch=None) -> torch.Tensor:
        """Per-cell embeddings H [N, hidden] -- the unit the head consumes."""
        h = self.input_proj(x)
        normalized_pos = None
        if self.use_pos and pos is not None:
            normalized_pos = normalize_positions(pos, batch)
            h = h + self.pos_proj(normalized_pos)
        return self.spatial(h, edge_index, batch)

    def forward(self, data) -> dict[str, torch.Tensor]:
        # Gene-query MODULATION + additive aggregation: gate each CELL by the gene
        # queries (cell_pred), then mean-aggregate over the spot/slide. Aggregating
        # AFTER the (nonlinear) head -- mean(head(h)) not head(mean(h)) -- gives the
        # faithful per-cell cellular ST `cell_pred` and a better spot/bulk metric.
        batch = getattr(data, "batch", None)
        h_node = self.encode(data.x, data.edge_index, getattr(data, "pos", None), batch)
        if self.granularity == "cell":
            cell_pred = self.head(h_node)
            return {"h_node": h_node, "prediction": cell_pred, "cell_pred": cell_pred}
        if batch is None:
            batch = torch.zeros(h_node.size(0), dtype=torch.long, device=h_node.device)
        cell_pred = self.head(h_node)
        prediction = scatter(cell_pred, batch, dim=0, reduce="mean")
        return {"h_node": h_node, "prediction": prediction, "cell_pred": cell_pred}
