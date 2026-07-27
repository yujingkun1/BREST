#!/usr/bin/env python3
"""Extract per-cell H0-mini features from bulk TCGA whole-slide images.

Difference from the per-patient version: NO dedup to 1 slide/patient. Every segmented
slide (cells.json) whose patient is in TPM and which has an svs becomes its OWN sample.
Multiple slides of the same patient => multiple samples, all sharing that patient's TPM row
(TPM pairing is done downstream by the `patient` field). Output one npz per SLIDE, keyed by
the full slide barcode.

For each selected slide:
  - load ALL cell centroids from CellViT cells.json (NO subsampling)
  - crop a fixed physical window (PHYS_UM) around each centroid, resize 224, H0 -> 1536-d
  - save per-slide npz: cell_features[N,1536] fp16, positions[N,2] fp32, patient, slide, n_total, mpp

Paths are passed via args so one script serves BRCA/LIHC/PRAD.
"""
import os, re, json, glob, argparse
import numpy as np
import openslide
import torch

from h0mini import H0MiniEncoder, H0MINI_MEAN, H0MINI_STD

MEAN = torch.tensor(H0MINI_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(H0MINI_STD).view(1, 3, 1, 1)
PHYS_UM = 56.0
H0_BATCH = int(os.environ.get("H0_BATCH", "256"))


def patient(s):
    match = re.match(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", s)
    return match.group(1) if match else None


def build_h0(dev, checkpoint):
    return H0MiniEncoder(
        checkpoint_path=checkpoint,
        trainable=False,
    ).eval().to(dev)


def slide_list(wsi_dir, seg_dir, tpm):
    """Return ALL segmented slides (no patient dedup) as (slide_barcode, svs_path, cells.json, patient)."""
    import pandas as pd
    tpm_pat = {patient(c) for c in pd.read_csv(tpm, nrows=0).columns[1:]}
    svs = {}
    for f in glob.glob(f"{wsi_dir}/*.svs"):
        bc = os.path.basename(f).split(".")[0]
        svs.setdefault(bc, f)
    out = []
    for d in glob.glob(f"{seg_dir}/TCGA-*"):
        cj = os.path.join(d, "cells.json")
        if not os.path.exists(cj):
            continue
        bc = os.path.basename(d).split(".")[0]   # full slide barcode, e.g. TCGA-A6-2671-01A-01-BS1
        p = patient(bc)
        if p in tpm_pat and bc in svs:
            out.append((bc, svs[bc], cj, p))      # NO dedup -> slice level
    out.sort()
    return out


@torch.no_grad()
def extract_slide(svs_path, cj_path, model, dev, max_cells=0):
    slide = openslide.OpenSlide(svs_path)
    mpp = float(slide.properties.get("openslide.mpp-x", 0.5))
    crop = max(16, int(round(PHYS_UM / mpp)))
    half = crop // 2
    cells = json.load(open(cj_path))["cells"]
    cen = np.array([c["centroid"] for c in cells], dtype=np.float64)  # [x,y] level-0
    n_total = len(cen)
    # Cap giant slides: random subsample of centroids before H0 (representative estimator of the
    # population manifold for a bulk WSI->patient-TPM backbone; bounds per-slide cost).
    if max_cells and n_total > max_cells:
        idx = np.random.RandomState(0).choice(n_total, max_cells, replace=False)
        cen = cen[idx]
    n_used = len(cen)
    # HDD read locality: sort kept cells top->bottom, left->right so read_region accesses svs
    # tiles near-sequentially (big win on a spinning disk vs scattered/random order). Features
    # and positions stay aligned (downstream pooling is order-invariant).
    if len(cen) > 1:
        cen = cen[np.lexsort((cen[:, 0], cen[:, 1]))]
    W, H = slide.level_dimensions[0]
    feats, buf = [], []

    def flush():
        if not buf:
            return
        arr = np.stack(buf).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(0, 3, 1, 2)
        x = (x - MEAN) / STD
        x = x.to(dev)
        with torch.autocast(
            device_type=dev.type,
            dtype=torch.float16,
            enabled=dev.type == "cuda",
        ):
            encoded = model(x)
            out = torch.cat([encoded["cls"], encoded["patch"].mean(1)], 1)
        feats.append(out.float().cpu().numpy())
        buf.clear()

    for cx, cy in cen:
        left, top = int(round(cx - half)), int(round(cy - half))
        left = min(max(left, 0), max(W - crop, 0)); top = min(max(top, 0), max(H - crop, 0))
        reg = slide.read_region((left, top), 0, (crop, crop)).convert("RGB")
        if crop != 224:
            reg = reg.resize((224, 224))
        buf.append(np.asarray(reg, dtype=np.uint8))
        if len(buf) >= H0_BATCH:
            flush()
    flush()
    slide.close()
    f = np.concatenate(feats, 0) if feats else np.zeros((0, 1536), np.float32)
    return f, cen.astype(np.float32), n_total, n_used, mpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi-dir", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--tpm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h0-checkpoint", required=True)
    ap.add_argument("--max-slides", type=int, default=0, help="0 = ALL segmented slides")
    ap.add_argument("--max-cells", type=int, default=0, help="0 = ALL cells; else random subsample per slide")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    model = build_h0(dev, a.h0_checkpoint)
    slides = slide_list(a.wsi_dir, a.seg_dir, a.tpm)
    if a.max_slides > 0:
        slides = slides[:a.max_slides]
    npat = len({p for _, _, _, p in slides})
    i, n = map(int, a.shard.split("/"))
    mine = slides[i::n]
    print(f"[shard {a.shard}] {len(mine)}/{len(slides)} slides ({npat} patients) on {a.device}, H0_BATCH={H0_BATCH}", flush=True)
    failures = []
    for k, (bc, svs_path, cj, p) in enumerate(mine):
        out = f"{a.out}/{bc}.npz"
        if os.path.exists(out):
            print(f"  [{a.shard}] skip {bc} (exists)", flush=True)
            continue
        try:
            f, pos, n_total, n_used, mpp = extract_slide(svs_path, cj, model, dev, a.max_cells)
            np.savez(out, cell_features=f.astype(np.float16), positions=pos,
                     patient=p, slide=bc, n_total=n_total, n_used=n_used, mpp=mpp)
            print(f"  [{a.shard}] {k+1}/{len(mine)} {bc} (pat {p}): {f.shape[0]}/{n_total} cells "
                  f"(cap={a.max_cells or 0}), mpp={mpp:.3f}", flush=True)
        except Exception as e:
            import traceback
            failures.append(bc)
            print(f"  [{a.shard}] FAIL {bc}: {e}", flush=True)
            traceback.print_exc()
    if failures:
        raise RuntimeError(f"Feature extraction failed for {len(failures)} slides: {failures}")
    print(f"SHARD_{a.shard.replace('/','_')}_DONE", flush=True)


if __name__ == "__main__":
    main()
