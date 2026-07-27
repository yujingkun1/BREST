"""Backbone transfer: load a bulk-trained UnifiedExpressionModel into a spatial
model. The spatial encoder + projections load wholesale by name; the per-gene
``gene_queries`` rows are copied by gene SYMBOL overlap (so the bulk and spatial
panels must share genes -- train the bulk backbone on the union of the target
Visium and Xenium panels for full overlap).

Three modes:
  frozen - backbone frozen, only the gene head (gene_queries + gene_readout) trains.
  full   - everything trainable (full fine-tune).
  lora   - backbone frozen + LoRA adapters on the local GAT; the gene head
           trains. The global Transformer stays frozen because its fused path
           reads linear weights directly.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen nn.Linear: y = W0 x + (alpha/r) B A x."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.scaling = alpha / r
        dev = base.weight.device  # keep adapters on the base layer's device
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # B stays 0 -> adapter starts as identity
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (self.drop(x) @ self.A.t()) @ self.B.t()


def apply_lora(module: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> int:
    """Recursively replace nn.Linear under ``module`` with LoRALinear. Returns count.

    Skips nn.TransformerEncoderLayer / nn.MultiheadAttention subtrees: their forward
    reads ``linear1.weight`` / ``out_proj.weight`` directly (fused path), which a
    LoRALinear wrapper does not expose. LoRA therefore adapts the local GAT layers,
    the input/pos projections and the gene head; the global Transformer stays frozen.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.TransformerEncoderLayer, nn.MultiheadAttention)):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            n += 1
        else:
            n += apply_lora(child, r=r, alpha=alpha, dropout=dropout)
    return n


def load_backbone_by_symbol(model, ckpt_path: str, tgt_genes: list[str],
                            src_genes: list[str] | None = None,
                            *, freeze: bool = True, map_location="cpu") -> dict:
    """Load a unified backbone into ``model``: spatial/head wholesale, gene_queries by symbol.

    ``ckpt`` is a state_dict (or {"model": state_dict, "genes": [...]}) from a
    backbone trained with this same UnifiedExpressionModel. Spatial + gene_readout
    load directly; ``gene_queries`` rows are copied only for genes present in both
    panels (matched by symbol). Returns a small report dict.
    """
    obj = torch.load(ckpt_path, map_location=map_location)
    src_state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if isinstance(obj, dict) and "genes" in obj and src_genes is None:
        src_genes = [str(g) for g in obj["genes"]]

    tgt_state = model.state_dict()
    loaded, skipped = [], []
    for k, v in src_state.items():
        if k == "head.gene_queries":
            continue  # handled per-gene below
        if k in tgt_state and tgt_state[k].shape == v.shape:
            tgt_state[k] = v
            loaded.append(k)
        else:
            skipped.append(k)

    n_copied = 0
    if (
        "head.gene_queries" in src_state
        and "head.gene_queries" in tgt_state
        and src_genes is not None
    ):
        src_q = src_state["head.gene_queries"]
        src_idx = {g: i for i, g in enumerate(src_genes)}
        tgt_q = tgt_state["head.gene_queries"].clone()
        for ti, g in enumerate(tgt_genes):
            si = src_idx.get(g)
            if si is not None and src_q[si].shape == tgt_q[ti].shape:
                tgt_q[ti] = src_q[si]
                n_copied += 1
        tgt_state["head.gene_queries"] = tgt_q
    elif "head.gene_queries" in src_state:
        skipped.append("head.gene_queries")

    model.load_state_dict(tgt_state)
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():  # keep the gene head trainable for unseen genes
            p.requires_grad_(True)
    return {"loaded": len(loaded), "skipped": skipped,
            "gene_queries_copied": n_copied, "n_tgt_genes": len(tgt_genes)}


def setup_transfer(model, ckpt_path: str, tgt_genes: list[str],
                   src_genes: list[str] | None = None, *, mode: str = "frozen",
                   lora_rank: int = 8, lora_alpha: int = 16, map_location="cpu") -> dict:
    """Load a unified backbone and configure the transfer arm (frozen|full|lora)."""
    if mode not in {"frozen", "full", "lora"}:
        raise ValueError(f"transfer mode must be frozen|full|lora, got {mode}")
    rep = load_backbone_by_symbol(model, ckpt_path, tgt_genes, src_genes,
                                  freeze=(mode != "full"), map_location=map_location)
    rep["mode"] = mode
    if mode == "lora":
        rep["lora_wrapped"] = apply_lora(model.spatial, r=lora_rank, alpha=lora_alpha)
        for p in model.head.parameters():  # keep the gene head trainable alongside LoRA
            p.requires_grad_(True)
    rep["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return rep
