#!/usr/bin/env python3
"""Visium per-spot training on the unified architecture (granularity="spot").

Leave-one-sample-out. Each spot becomes one intra-spot Delaunay graph. The gene
head predicts every cell and the cell predictions are averaged into the spot
output. Optionally load a bulk backbone by gene symbol.

Eval (METRICS.md): target = per-gene z-scored log1p expression (train mu/sd);
predictions de-normalised to log1p before scoring; per fold report the
overall/gene at the epoch with the highest gene Pearson, then mean ± std.

Usage:
    python -m brest.train.visium --cancer COAD --sp-dir <cache> [--backbone bb.pt --transfer full]
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeoDataLoader

from brest.data import build_bag_graphs
from brest.models import UnifiedExpressionModel, setup_transfer
from brest.utils import mean_gene_pearson, overall_pearson

# Per-cancer Visium sample lists + default cache dirs. Override the dir with --sp-dir.
CANCERS = {
    "COAD": dict(sp="cache/coad_h0", samples="TENX152 TENX92 TENX91 TENX90 TENX89 TENX49 TENX29 ZEN47 ZEN46 ZEN45 ZEN44 ZEN43 ZEN42 ZEN39 ZEN38".split()),
    "BRCA": dict(sp="cache/brca_h0_785", samples=[f"SPA{i}" for i in range(119, 155)]),
    "LIHC": dict(sp="cache/lihc_h0", samples=["NCBI642", "NCBI643"]),
    "PRAD": dict(sp="cache/prad_h0", samples=[f"MEND{i}" for i in list(range(139, 155)) + list(range(156, 163))]),
    "CSCC": dict(sp="cache/cscc_h0", samples=[f"NCBI{i}" for i in range(759, 771)]),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sample(sp: str, sample: str):
    npz = np.load(f"{sp}/visium_{sample}.npz")
    coords = np.load(f"{sp}/coords_{sample}.npy")
    expr = npz["expr"].astype(np.float32)
    graphs = build_bag_graphs({
        "cell_features": npz["cell_features"],
        "cell_coords": coords,
        "bag_ptr": npz["bag_ptr"],
        "spot_expression": expr,
    })
    genes = [str(g) for g in npz["genes"]]
    return graphs, expr, genes


def run_fold(train_graphs, test_graphs, test_expr, genes, args, device, fold_seed,
             backbone=None, src_genes=None):
    seed_everything(fold_seed)
    tr_expr = np.stack([g.y.numpy().ravel() for g in train_graphs]).astype(np.float32)
    mu, sd = tr_expr.mean(0), tr_expr.std(0) + 1e-6
    mu_t = torch.from_numpy(mu).to(device)
    sd_t = torch.from_numpy(sd).to(device)

    model = UnifiedExpressionModel(
        in_dim=train_graphs[0].x.shape[1], num_genes=len(genes), hidden_dim=args.hidden_dim,
        n_layers=args.num_layers, heads=args.heads, dropout=args.dropout, granularity="spot",
        gene_chunk=args.gene_chunk, head_type=args.head_type, use_local=not args.no_gat,
    ).to(device)
    if backbone:
        rep = setup_transfer(model, backbone, genes, src_genes, mode=args.transfer,
                             lora_rank=args.lora_rank, map_location=device)
        print(f"    backbone[{args.transfer}]: {rep['loaded']} tensors, gene_queries "
              f"{rep['gene_queries_copied']}/{rep['n_tgt_genes']} by symbol, "
              f"trainable={rep['trainable_params'] / 1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    loader = GeoDataLoader(train_graphs, batch_size=args.batch, shuffle=True)
    best_g, best_o_at_g, best_ep = -9.0, -9.0, -1
    for ep in range(args.epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)["prediction"]
            loss = F.mse_loss(pred, (batch.y - mu_t) / sd_t)
            loss.backward()
            opt.step()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            preds = []
            with torch.no_grad():
                for batch in GeoDataLoader(test_graphs, batch_size=args.batch):
                    batch = batch.to(device)
                    preds.append((model(batch)["prediction"] * sd_t + mu_t).cpu().numpy())
            P = np.concatenate(preds, 0)
            o, g = overall_pearson(P, test_expr), mean_gene_pearson(P, test_expr)
            if g > best_g:                      # model selection by gene Pearson
                best_g, best_o_at_g, best_ep = g, o, ep + 1
    torch.cuda.empty_cache()
    return best_o_at_g, best_g, best_ep


def _write_results(path, args, genes, fold_rows, holdout, complete):
    outs = np.array([row["overall_pearson"] for row in fold_rows], dtype=float)
    genes_p = np.array([row["mean_gene_pearson"] for row in fold_rows], dtype=float)
    output = {
        "dataset": args.cancer, "platform": "Visium", "head_type": args.head_type,
        "transfer": args.transfer if args.backbone else None,
        "seed": args.seed, "genes": len(genes), "folds": fold_rows,
        "complete": bool(complete and len(fold_rows) == len(holdout)),
        "summary": {
            "overall_mean": float(outs.mean()), "overall_std": float(outs.std()),
            "gene_mean": float(genes_p.mean()), "gene_std": float(genes_p.std()),
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(output, fh, indent=2)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True)
    ap.add_argument("--sp-dir", default=None, help="override the cached spatial dir")
    ap.add_argument("--folds", type=int, default=999)
    ap.add_argument("--fold-samples", default=None, help="comma-list of holdout samples (LOO over just these)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", default=None, help="unified bulk backbone .pt (transfer arm)")
    ap.add_argument("--transfer", default="frozen", choices=["frozen", "full", "lora"])
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--head-type", default="gene_query", choices=["gene_query", "linear"],
                    help="gene_query = full BREST head; linear = direct no-query ablation")
    ap.add_argument("--no-gat", action="store_true", help="disable local GAT for ablation")
    ap.add_argument("--gene-chunk", type=int, default=64,
                    help="genes per chunk in the modulation head; lower = less GPU memory (no effect on results)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-json", default=None, help="optional fold-level metrics artifact")
    ap.add_argument("--resume", action="store_true", help="skip folds already present in output-json")
    args = ap.parse_args()

    c = CANCERS[args.cancer]
    if args.sp_dir:
        c = {**c, "sp": args.sp_dir}
    dev = torch.device(args.device)
    have = [s for s in c["samples"] if __import__("os").path.exists(f"{c['sp']}/coords_{s}.npy")]
    holdout = [s.strip() for s in args.fold_samples.split(",")] if args.fold_samples else have[:min(args.folds, len(have))]
    if not have:
        raise FileNotFoundError(f"No Visium caches found in {c['sp']!r} for {args.cancer}")
    cache = {s: load_sample(c["sp"], s) for s in have}
    genes = cache[have[0]][2]
    src_genes = None
    if args.backbone:
        obj = torch.load(args.backbone, map_location="cpu")
        if isinstance(obj, dict) and "genes" in obj:
            src_genes = [str(g) for g in obj["genes"]]
    print(f"[{args.cancer} Visium] {len(have)} samples, {len(genes)} genes, granularity=spot, "
          f"head={args.head_type}, gat={'off' if args.no_gat else 'on'}, "
          f"backbone={'yes' if args.backbone else 'no'}, holdout={holdout}", flush=True)

    fold_rows = []
    if args.resume and args.output_json and os.path.exists(args.output_json):
        previous = json.load(open(args.output_json))
        fold_rows = previous.get("folds", [])
    completed = {row["sample"] for row in fold_rows}
    for test in holdout:
        if test in completed:
            print(f"  {test}: already complete, skipping", flush=True)
            continue
        train_graphs = [g for s in have if s != test for g in cache[s][0]]
        test_graphs, test_expr, _ = cache[test]
        o, gp, best_ep = run_fold(train_graphs, test_graphs, test_expr, genes, args, dev,
                                  args.seed + have.index(test),
                                  backbone=args.backbone, src_genes=src_genes)
        fold_rows.append({"sample": test, "overall_pearson": o,
                          "mean_gene_pearson": gp, "best_epoch": best_ep})
        print(f"  {test}: overall={o:.3f} gene={gp:.3f} best_epoch={best_ep}", flush=True)
        if args.output_json:
            _write_results(args.output_json, args, genes, fold_rows, holdout, complete=False)
    ordered = [row for sample in holdout for row in fold_rows if row["sample"] == sample]
    outs = np.array([row["overall_pearson"] for row in ordered])
    genes_p = np.array([row["mean_gene_pearson"] for row in ordered])
    print(f">>> [{args.cancer} Visium] overall={outs.mean():.3f}+/-{outs.std():.3f} "
          f"gene={genes_p.mean():.3f}+/-{genes_p.std():.3f}", flush=True)
    if args.output_json:
        _write_results(args.output_json, args, genes, ordered, holdout, complete=True)


if __name__ == "__main__":
    main()
