#!/usr/bin/env python3
"""Xenium per-cell training on the unified architecture (granularity="cell").

Xenium is already single-cell. A fold's cells form one large Delaunay graph, so we
tile it into spatial sub-graphs (Delaunay rebuilt per tile) for mini-batching.
5-fold cross-validation by the precomputed ``position_fold_5`` split.

Consumes a precomputed per-cell H0 feature cache (npz with features / coords /
expression, aligned to the metadata row order). The required schema is
documented in ``docs/preprocessing.md``. Optionally loads a bulk backbone.

Usage:
    python -m brest.train.xenium --feature-cache <sample>_all.npz \
        --metadata <sample>_cell_metadata.parquet --gene-list <panel>.txt \
        [--backbone bb.pt --transfer full] --output-dir runs/xen
"""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

from brest.data import build_graph_arrays, spatial_tiles
from brest.models import UnifiedExpressionModel, setup_transfer
from brest.utils import mean_gene_pearson, overall_pearson


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_subgraphs(feats, coords, expr, idx, tile_size, device):
    graphs = []
    for tile in spatial_tiles(coords[idx], tile_size):
        rows = idx[tile]
        graphs.append(build_graph_arrays(feats[rows], coords[rows], expr[rows]).to(device))
    return graphs


def train_one_fold(args, genes, feats, coords, expr, fold_ids, fold, parent_dir, device) -> None:
    seed_everything(args.seed + fold)
    tr_idx = np.flatnonzero(fold_ids != fold)
    te_idx = np.flatnonzero(fold_ids == fold)
    train_graphs = make_subgraphs(feats, coords, expr, tr_idx, args.tile_size, device)
    test_graphs = make_subgraphs(feats, coords, expr, te_idx, args.tile_size, device)
    print(f"  fold{fold}: train {len(tr_idx)} cells / {len(train_graphs)} tiles, "
          f"test {len(te_idx)} cells / {len(test_graphs)} tiles", flush=True)

    model = UnifiedExpressionModel(
        in_dim=feats.shape[1], num_genes=len(genes), hidden_dim=args.hidden_dim,
        n_layers=args.num_layers, heads=args.heads, dropout=args.dropout,
        use_pos=not args.no_pos, granularity="cell", head_type=args.head_type,
        use_local=not args.no_gat,
    ).to(device)
    if args.backbone:
        rep = setup_transfer(model, args.backbone, [str(g) for g in genes], mode=args.transfer,
                             lora_rank=args.lora_rank, map_location=device)
        print(f"  [fold{fold}] backbone[{args.transfer}]: {rep['loaded']} tensors, gene_queries "
              f"{rep['gene_queries_copied']}/{rep['n_tgt_genes']} by symbol, "
              f"trainable={rep['trainable_params'] / 1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    loader = GeoDataLoader(train_graphs, batch_size=args.graph_batch, shuffle=True)

    run_dir = os.path.join(parent_dir, f"fold{fold}")
    os.makedirs(run_dir, exist_ok=True)
    best_metric, best_payload, history = -1e9, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch)["prediction"], batch.y)
            loss.backward(); opt.step()
            total += float(loss.detach().cpu()) * batch.num_graphs; count += batch.num_graphs
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for data in test_graphs:
                preds.append(model(data)["prediction"].float().cpu().numpy())
                targets.append(data.y.float().cpu().numpy())
        P = np.concatenate(preds, 0); T = np.concatenate(targets, 0)
        metrics = {"overall_pearson": overall_pearson(P, T),
                   "mean_gene_pearson": mean_gene_pearson(P, T),
                   "num_genes": len(genes), "num_cells": int(T.shape[0])}
        row = {"epoch": epoch, "train": {"loss": total / max(count, 1)}, "test": metrics}
        history.append(row)
        print(f"[fold{fold}] epoch={epoch:03d} loss={total / max(count, 1):.5f} "
              f"test_mean_gene={metrics['mean_gene_pearson']:.5f} "
              f"test_overall={metrics['overall_pearson']:.5f}", flush=True)
        if metrics["mean_gene_pearson"] > best_metric:   # model selection by gene Pearson
            best_metric = metrics["mean_gene_pearson"]; best_payload = row
        with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
            json.dump({"history": history, "best": best_payload, "last": history[-1]}, fh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-cache", required=True, help="npz with features/coords/expression (per-cell, metadata order)")
    ap.add_argument("--metadata", required=True, help="parquet with a position_fold_5 column")
    ap.add_argument("--gene-list", required=True)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--output-dir", default="runs/xenium")
    ap.add_argument("--batch-name", default=None)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-pos", action="store_true")
    ap.add_argument("--head-type", default="gene_query", choices=["gene_query", "linear"],
                    help="gene_query = full BREST head; linear = direct no-query ablation")
    ap.add_argument("--no-gat", action="store_true", help="disable local GAT for ablation")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--transfer", default="frozen", choices=["frozen", "full", "lora"])
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--graph-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    genes = [l.strip() for l in open(args.gene_list) if l.strip()]
    cache = np.load(args.feature_cache)
    feats = cache["features"].astype(np.float32)
    coords = cache["coords"].astype(np.float32)
    expr = cache["expression"].astype(np.float32)
    fold_ids = pd.read_parquet(args.metadata)["position_fold_5"].to_numpy(dtype=np.int64)
    assert len(fold_ids) == len(feats), f"meta {len(fold_ids)} != features {len(feats)}"

    folds = args.folds if args.folds is not None else list(range(5))
    parent_dir = os.path.join(args.output_dir, args.batch_name or datetime.now().strftime("graph_%Y%m%d_%H%M%S")) \
        if (len(folds) > 1 or args.batch_name) else args.output_dir
    print(f"[xenium] device={device} folds={folds} tile={args.tile_size} genes={len(genes)} "
          f"head={args.head_type} gat={'off' if args.no_gat else 'on'} "
          f"backbone={'yes' if args.backbone else 'no'} out={parent_dir}", flush=True)
    for fold in folds:
        print(f"=== Xenium fold {fold} ===", flush=True)
        train_one_fold(args, genes, feats, coords, expr, fold_ids, int(fold), parent_dir, device)


if __name__ == "__main__":
    main()
