#!/usr/bin/env python3
"""Build tiled bulk WSI subgraphs from per-slide H0-mini feature caches.

Each slide is divided into spatial tiles and a local Delaunay graph is built
within every retained tile. All cells are kept unless ``--max-total-cells`` is
set, in which case the densest complete tiles are retained first.

Input NPZ files are produced by ``extract_bulk_cells.py`` and contain
``cell_features``, ``positions``, ``patient``, ``slide``, ``n_total``, and
``mpp``.

Outputs per split:
  ``bulk_<split>_intra_patch_graphs.pkl`` and
  ``bulk_<split>_slide_to_patient_mapping.pkl``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data

from brest.data import build_delaunay_edges, split_slides_by_patient


def local_delaunay_edges(pos: np.ndarray) -> torch.Tensor:
    """Return a bidirectional ``edge_index`` for one tile."""
    edges = build_delaunay_edges(pos)
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    bidirectional = np.concatenate([edges, edges[:, ::-1]], axis=0)
    return torch.from_numpy(bidirectional.T.copy()).long()


def build_slide_subgraphs(feat, pos, tile_px, min_cells, max_total_cells):
    """Tile one slide and return ``{tile_id: Data}`` plus retained cell count."""
    tx = np.floor(pos[:, 0] / tile_px).astype(np.int64)
    ty = np.floor(pos[:, 1] / tile_px).astype(np.int64)
    keys = tx * 100000 + ty
    _, inv = np.unique(keys, return_inverse=True)
    groups = {}
    for ci, t in enumerate(inv):
        groups.setdefault(int(t), []).append(ci)
    tiles = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    subgraphs = {}
    kept = 0
    tid = 0
    for _, idxs in tiles:
        if len(idxs) < min_cells:
            continue
        if max_total_cells and kept + len(idxs) > max_total_cells and kept > 0:
            continue
        idxs = np.array(idxs)
        p = pos[idxs]
        x = torch.from_numpy(feat[idxs].astype(np.float32))
        ei = local_delaunay_edges(p)
        subgraphs[tid] = Data(x=x, edge_index=ei, pos=torch.from_numpy(p))
        kept += len(idxs)
        tid += 1
    return subgraphs, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--tile-um",
        type=float,
        default=256.0,
        help="physical tile size in microns (converted to pixels using slide MPP)",
    )
    ap.add_argument(
        "--min-cells", type=int, default=3, help="minimum cells in a retained tile"
    )
    ap.add_argument(
        "--max-total-cells",
        type=int,
        default=40000,
        help="per-slide cap using complete dense tiles (0 keeps all cells)",
    )
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feature-dim", type=int, default=1536)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    max_total = None if a.max_total_cells <= 0 else a.max_total_cells

    files = sorted(glob.glob(os.path.join(a.npz_dir, "TCGA-*.npz")))
    print(f"found {len(files)} npz files", flush=True)
    if not files:
        raise FileNotFoundError(f"No TCGA-*.npz files found in {a.npz_dir}")
    recs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        feat = d["cell_features"]
        if feat.shape[0] == 0:
            print("  skip empty", os.path.basename(f))
            continue
        pos = d["positions"].astype(np.float32)
        pid = str(d["patient"])
        mpp = float(d["mpp"]) if "mpp" in d.files else 0.5
        slide_id = str(d["slide"]) if "slide" in d.files else os.path.basename(f).removesuffix(".npz")
        recs.append((slide_id, pid, feat, pos, int(d["n_total"]), mpp))
    print(
        f"usable slides: {len(recs)} from {len({record[1] for record in recs})} patients",
        flush=True,
    )
    if not recs:
        raise RuntimeError("No non-empty slide feature caches were found")

    records_by_slide = {record[0]: record for record in recs}
    slide_to_patient = {record[0]: record[1] for record in recs}
    split_ids = split_slides_by_patient(
        slide_to_patient,
        test_fraction=a.test_frac,
        seed=a.seed,
    )
    splits = {"train": [], "test": []}
    for split, slide_ids in split_ids.items():
        splits[split] = [records_by_slide[slide_id] for slide_id in slide_ids]
    print(f"split: train {len(splits['train'])}  test {len(splits['test'])}", flush=True)

    n_sub_all = []; n_cells_kept = []; sub_sizes = []
    for split, items in splits.items():
        intra = {}
        s2p = {}
        meta = {}
        for slide_id, pid, feat, pos, n_total, mpp in items:
            tile_px = max(1.0, a.tile_um / max(mpp, 1e-6))
            subg, kept = build_slide_subgraphs(feat, pos, tile_px, a.min_cells, max_total)
            if not subg:
                # fallback: whole slide one graph (rare tiny slides)
                x = torch.from_numpy(feat.astype(np.float32))
                subg = {0: Data(x=x, edge_index=local_delaunay_edges(pos), pos=torch.from_numpy(pos))}
                kept = feat.shape[0]
            intra[slide_id] = subg
            s2p[slide_id] = pid
            meta[slide_id] = {
                "slide_id": slide_id,
                "patient_id": pid,
                "n_total": n_total,
                "num_subgraphs": len(subg),
                "cells_kept": int(kept),
                "has_graphs": True,
                "feature_dim": a.feature_dim,
            }
            n_sub_all.append(len(subg))
            n_cells_kept.append(int(kept))
            sub_sizes.extend([g.x.shape[0] for g in subg.values()])

        def dump(name, obj):
            with open(os.path.join(a.out_dir, f"bulk_{split}_{name}.pkl"), "wb") as fh:
                pickle.dump(obj, fh)
        dump("intra_patch_graphs", intra)
        dump("slide_to_patient_mapping", s2p)
        with open(os.path.join(a.out_dir, f"bulk_{split}_metadata.json"), "w") as fh:
            json.dump({"feature_dim": a.feature_dim, "num_slides": len(items), "slides": meta}, fh)
        print(f"[{split}] wrote {len(items)} slides", flush=True)

    ns = np.array(n_sub_all)
    nc = np.array(n_cells_kept)
    ss = np.array(sub_sizes)
    print(f"subgraphs/slide: min {ns.min()} med {int(np.median(ns))} mean {ns.mean():.0f} max {ns.max()}", flush=True)
    print(f"cells kept/slide: min {nc.min()} med {int(np.median(nc))} mean {int(nc.mean())} max {nc.max()}", flush=True)
    print(f"cells/subgraph: min {ss.min()} med {int(np.median(ss))} mean {ss.mean():.1f} max {ss.max()} p95 {int(np.percentile(ss,95))}", flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
