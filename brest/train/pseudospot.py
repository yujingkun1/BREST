#!/usr/bin/env python3
"""Evaluate cellular deconvolution learned from pseudo-spot supervision.

Xenium gives true single-cell expression. We bin its cells into Visium-like
pseudo-spots, then train the gene-query head (per-cell modulation followed by
additive mean) with only the pseudo-spot mean as the label.
At eval we read off the per-cell prediction ``cell_pred`` and score it against the
TRUE single-cell expression. We compare against a model trained with direct CELL
supervision (the upper bound).

The gene-query head supports this because its spot prediction is
``mean(head(cell))``, leaving each cell output as an explicit
supervised-by-aggregation estimate.

Metric: per-cell mean_gene_pearson of cell_pred vs true single-cell expression.

Usage:
    python -m brest.train.pseudospot --feature-cache <sample>_all.npz --gene-list <panel>.txt \
        --spot-size 50 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.utils import to_undirected

from brest.data import build_delaunay_edges, spatial_tiles
from brest.models import UnifiedExpressionModel
from brest.utils import mean_gene_pearson, overall_pearson, gene_pearson_stats


def make_pseudospots(feats, coords, expr, spot_size):
    """Bin cells into spatial pseudo-spots (~spot_size cells each). Each pseudo-spot
    is one intra-spot Delaunay graph; its label is the mean cell expression."""
    graphs, cell_expr_groups = [], []
    for tile in spatial_tiles(coords, spot_size):
        cc = coords[tile]
        edges = build_delaunay_edges(cc)
        ei = to_undirected(torch.from_numpy(edges.T).long()) if edges.shape[0] else torch.empty((2, 0), dtype=torch.long)
        spot_label = expr[tile].mean(0)                          # pseudo-spot mean (the supervision)
        graphs.append(Data(x=torch.from_numpy(feats[tile]).float(),
                           edge_index=ei,
                           pos=torch.from_numpy(cc.astype(np.float32)),
                           y=torch.from_numpy(spot_label).unsqueeze(0)))
        cell_expr_groups.append(expr[tile])                     # true per-cell expression (held for eval)
    return graphs, cell_expr_groups


def make_cellgraphs(feats, coords, expr, tile_size):
    """Per-cell graphs (tiled) for direct cell supervision, y = per-cell expression."""
    graphs = []
    for tile in spatial_tiles(coords, tile_size):
        cc = coords[tile]
        edges = build_delaunay_edges(cc)
        ei = to_undirected(torch.from_numpy(edges.T).long()) if edges.shape[0] else torch.empty((2, 0), dtype=torch.long)
        graphs.append(Data(x=torch.from_numpy(feats[tile]).float(), edge_index=ei,
                           pos=torch.from_numpy(cc.astype(np.float32)),
                           y=torch.from_numpy(expr[tile]).float()))
    return graphs


def train_eval_spot(train_graphs, test_graphs, test_cell_expr, n_genes, args, device):
    """Train on pseudo-spot means and evaluate per-cell predictions."""
    tr = np.stack([g.y.numpy().ravel() for g in train_graphs]).astype(np.float32)
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    mu_t, sd_t = torch.from_numpy(mu).to(device), torch.from_numpy(sd).to(device)
    model = UnifiedExpressionModel(in_dim=train_graphs[0].x.shape[1], num_genes=n_genes,
                                   granularity="spot", gene_chunk=args.gene_chunk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = GeoDataLoader(train_graphs, batch_size=args.batch, shuffle=True)
    for ep in range(args.epochs):
        model.train()
        for b in loader:
            b = b.to(device); opt.zero_grad()
            loss = F.mse_loss(model(b)["prediction"], (b.y - mu_t) / sd_t)
            loss.backward(); opt.step()
    # eval: per-cell cell_pred (de-normalised) vs TRUE single-cell expression
    model.eval(); P, T = [], []
    with torch.no_grad():
        for g, ce in zip(test_graphs, test_cell_expr):
            cp = model(Batch.from_data_list([g]).to(device))["cell_pred"]   # [n_cells, G]
            P.append((cp * sd_t + mu_t).cpu().numpy()); T.append(ce)
    P, T = np.concatenate(P, 0), np.concatenate(T, 0)
    gm, gs = gene_pearson_stats(P, T)
    return overall_pearson(P, T), gm, gs


def train_eval_cell(train_graphs, test_graphs, n_genes, args, device):
    """Direct cell supervision (upper bound): y = per-cell expression."""
    tr = np.concatenate([g.y.numpy() for g in train_graphs], 0).astype(np.float32)
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    mu_t, sd_t = torch.from_numpy(mu).to(device), torch.from_numpy(sd).to(device)
    model = UnifiedExpressionModel(in_dim=train_graphs[0].x.shape[1], num_genes=n_genes,
                                   granularity="cell", gene_chunk=args.gene_chunk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = GeoDataLoader(train_graphs, batch_size=args.batch, shuffle=True)
    for ep in range(args.epochs):
        model.train()
        for b in loader:
            b = b.to(device); opt.zero_grad()
            loss = F.mse_loss(model(b)["prediction"], (b.y - mu_t) / sd_t)
            loss.backward(); opt.step()
    model.eval(); P, T = [], []
    with torch.no_grad():
        for g in test_graphs:
            pr = model(Batch.from_data_list([g]).to(device))["prediction"]
            P.append((pr * sd_t + mu_t).cpu().numpy()); T.append(g.y.numpy())
    P, T = np.concatenate(P, 0), np.concatenate(T, 0)
    gm, gs = gene_pearson_stats(P, T)
    return overall_pearson(P, T), gm, gs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-cache", required=True, help="Xenium per-cell npz (features/coords/expression)")
    ap.add_argument("--gene-list", required=True)
    ap.add_argument("--sample", default="?", help="label for logging")
    ap.add_argument("--spot-size", type=int, default=50, help="cells per pseudo-spot")
    ap.add_argument("--tile-size", type=int, default=512, help="cells per tile for the cell-supervised graphs")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gene-chunk", type=int, default=16)
    ap.add_argument("--max-cells", type=int, default=80000,
                    help="subsample to this many cells (graph building over all cells is slow; 0 = all)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    genes = [l.strip() for l in open(args.gene_list) if l.strip()]
    d = np.load(args.feature_cache)
    feats, coords, expr = d["features"].astype(np.float32), d["coords"].astype(np.float64), np.log1p(d["expression"].astype(np.float32))
    if args.max_cells and len(feats) > args.max_cells:
        # contiguous spatial crop keeps neighbourhood structure (vs random subsample)
        order = np.lexsort((coords[:, 1], coords[:, 0]))[:args.max_cells]
        feats, coords, expr = feats[order], coords[order], expr[order]
    n_genes = expr.shape[1]
    print(f"[pseudospot {args.sample}] {len(feats)} cells (max {args.max_cells}), {n_genes} genes, spot_size={args.spot_size}", flush=True)

    # spatial train/test split: split the x-axis (left = train, right = test)
    rng = np.random.default_rng(args.seed)
    thr = np.quantile(coords[:, 0], 1 - args.test_frac)
    test_mask = coords[:, 0] >= thr
    tr_idx, te_idx = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)

    # build pseudo-spots (for spot-supervised) and cell graphs (for cell-supervised)
    sp_tr, _ = make_pseudospots(feats[tr_idx], coords[tr_idx], expr[tr_idx], args.spot_size)
    sp_te, ce_te = make_pseudospots(feats[te_idx], coords[te_idx], expr[te_idx], args.spot_size)
    cg_tr = make_cellgraphs(feats[tr_idx], coords[tr_idx], expr[tr_idx], args.tile_size)
    cg_te = make_cellgraphs(feats[te_idx], coords[te_idx], expr[te_idx], args.tile_size)

    o_spot, g_spot, gs_spot = train_eval_spot(sp_tr, sp_te, ce_te, n_genes, args, device)
    o_cell, g_cell, gs_cell = train_eval_cell(cg_tr, cg_te, n_genes, args, device)
    print(f">>> [{args.sample}] spot-supervised per-cell: overall={o_spot:.3f} "
          f"gene={g_spot:.3f}+/-{gs_spot:.3f} | CELL-supervised per-cell: overall={o_cell:.3f} "
          f"gene={g_cell:.3f}+/-{gs_cell:.3f} | recovery={g_spot / g_cell:.2%}", flush=True)


if __name__ == "__main__":
    main()
