#!/usr/bin/env python3
"""Bulk-RNA backbone training on the unified architecture (granularity="slide").

A slide's WSI cells are tiled into Delaunay sub-graphs (built by
``scripts/preprocess/build_bulk_subgraphs.py``); the model predicts per-subgraph gene vectors
that are SUMMED over a slide, L1-normalised over genes and scaled to 1e6, then
matched (MSE) to the slide's TCGA bulk TPM(1e6). Saves
``{"model": state_dict, "genes": GENE_SYMBOLS}`` so the backbone transfers into
the Visium/Xenium models by gene symbol.

IMPORTANT: train the bulk backbone on the UNION of the target Visium ∪ Xenium gene
panels (intersected with the TCGA transcriptome) so the gene head transfers 100 %.

Usage:
    python -m brest.train.bulk --graphs bulk_COAD_subgraphs --gene-list union.txt \
        --tpm tpm-union-million.csv --out bb_COAD_union.pt
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from brest.data import BulkGraphDataset, collate_bulk
from brest.models import UnifiedExpressionModel

EPS = 1e-8


def read_symbols_file_order(path: str) -> list[str]:
    out = []
    for line in open(path):
        g = line.strip()
        if g and not g.startswith(("Efficiently", "Total", "Detection", "Samples")):
            out.append(g)
    return out


def normalize_pred(p: torch.Tensor) -> torch.Tensor:  # [G] -> CPM 1e6, clamp
    return torch.clamp(p / (p.sum() + EPS) * 1e6, min=0.0, max=1e6)


def pearson(a, b, eps=1e-12):
    if a.std() < eps or b.std() < eps:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", required=True, help="bulk subgraph dir (build_bulk_subgraphs.py output)")
    ap.add_argument("--gene-list", required=True, help="union gene symbols, file order")
    ap.add_argument("--tpm", required=True, help="panel-filtered TPM CSV in gene-list order")
    ap.add_argument("--out", required=True, help="output backbone .pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    gene_symbols = read_symbols_file_order(args.gene_list)
    num_genes = len(gene_symbols)
    print(f"[genes] {num_genes} file-order symbols (first 3 {gene_symbols[:3]})", flush=True)

    train_ds = BulkGraphDataset(args.graphs, args.tpm, gene_symbols, split="train")
    test_ds = BulkGraphDataset(args.graphs, args.tpm, gene_symbols, split="test")
    print(f"[data] train {len(train_ds)} test {len(test_ds)} num_genes {num_genes}", flush=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_bulk)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_bulk)

    model = UnifiedExpressionModel(in_dim=1536, num_genes=num_genes, hidden_dim=args.hidden_dim,
                                   n_layers=args.num_layers, heads=args.heads, granularity="slide").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = torch.nn.MSELoss()

    def slide_pred(spot_graphs):
        graphs = [g for g in spot_graphs if getattr(g, "x", None) is not None and g.x.shape[0] > 0]
        if not graphs:
            return None
        for g in graphs:
            g.x = g.x.float()
            if getattr(g, "pos", None) is None:
                g.pos = torch.zeros((g.x.shape[0], 2))
        out = model(Batch.from_data_list(graphs).to(device))["prediction"]  # [n_sub, G]
        return out.sum(dim=0)                                               # [G]

    @torch.no_grad()
    def evaluate():
        model.eval(); preds, tgts = [], []
        for batch in test_dl:
            for i in range(len(batch["slide_ids"])):
                sp = slide_pred(batch["spot_graphs_list"][i])
                if sp is None:
                    continue
                preds.append(normalize_pred(sp).cpu().numpy()); tgts.append(batch["expressions"][i].numpy())
        if not preds:
            return float("nan"), float("nan"), 0
        P, T = np.array(preds), np.array(tgts)
        overall = float(np.nanmean([pearson(T[i], P[i]) for i in range(len(T))]))
        gene = float(np.nanmean([pearson(T[:, g], P[:, g]) for g in range(T.shape[1])]))
        return overall, gene, len(T)

    best, best_state, best_ep, best_gene = -1.0, None, -1, float("nan")
    for ep in range(args.epochs):
        model.train(); tot, nb = 0.0, 0
        for batch in train_dl:
            exprs = batch["expressions"].to(device)
            opt.zero_grad(); loss, cnt = 0.0, 0
            for i in range(len(batch["slide_ids"])):
                sp = slide_pred(batch["spot_graphs_list"][i])
                if sp is None:
                    continue
                loss = loss + crit(normalize_pred(sp), exprs[i]); cnt += 1
            if cnt == 0:
                continue
            (loss / cnt).backward(); opt.step()
            tot += float(loss / cnt); nb += 1
        ov, ge, n = evaluate()
        print(f"epoch {ep:02d} train_mse {tot/max(nb,1):.2f} test_overall {ov:.4f} "
              f"test_gene {ge:.4f} n {n} t {time.time()-t0:.0f}s", flush=True)
        if not np.isnan(ov) and ov > best:
            best, best_ep, best_gene = ov, ep, ge
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"model": best_state, "genes": gene_symbols}, args.out)
    print(f"SAVED {args.out}  best_ep {best_ep}  best_overall {best:.4f}  best_gene {best_gene:.4f}", flush=True)


if __name__ == "__main__":
    main()
