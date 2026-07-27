#!/usr/bin/env python3
"""Extract H0-mini cell features for Visium samples.

Each output cache contains a CSR-style bag of 1,536-dimensional cell features
and the aligned log1p spot-expression panel consumed by ``brest.train.visium``.
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

from _helpers import _SimpleWSI, _load_cell_centroids, decode_string
from h0mini import H0MINI_MEAN, H0MINI_STD, H0MiniEncoder

HEST = os.environ.get("BREST_HEST_DIR", "")
PANEL = []
OUT = os.environ.get("BREST_SPATIAL_CACHE_DIR", "cache/spatial_h0")
PHYS_UM = 56.0
MAX_SPOTS = 0
MEAN = torch.tensor(H0MINI_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(H0MINI_STD).view(1, 3, 1, 1)


def open_slide(p):
    try:
        import openslide

        return openslide.OpenSlide(p)
    except Exception:
        return _SimpleWSI(p)


@torch.no_grad()
def extract_visium(s, enc, dev):
    with h5py.File(f"{HEST}/patches/{s}.h5","r") as h:
        coords = np.asarray(h["coords"][:], np.float64)
        barcodes = [decode_string(b) for b in h["barcode"][:]]
        att = dict(h["img"].attrs); mpp = float(att["pixel_size"]); box = int(att["patch_size_src"])
    crop = max(16, int(round(PHYS_UM/mpp))); half = crop//2
    if MAX_SPOTS and len(coords) > MAX_SPOTS:  # deterministic spot subsample (same across scales)
        rng = np.random.RandomState(0); sel = np.sort(rng.choice(len(coords), MAX_SPOTS, replace=False))
        coords = coords[sel]; barcodes = [barcodes[i] for i in sel]
    cen = _load_cell_centroids(HEST, s)
    a = sc.read_h5ad(f"{HEST}/st/{s}.h5ad")
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    vn = {g:i for i,g in enumerate(map(str,a.var_names))}
    PANEL_OK = [g for g in PANEL if g in vn]; gidx = [vn[g] for g in PANEL_OK]; bc2row = {b:i for i,b in enumerate(map(str,a.obs_names))}
    slide = open_slide(f"{HEST}/wsis/{s}.tif"); W,H = slide.level_dimensions[0]
    from PIL import Image
    feats, ptr, exprs, buf, cell_xy = [], [0], [], [], []
    def flush():
        if not buf: return
        x = torch.from_numpy(np.stack(buf).astype(np.float32)/255.).permute(0,3,1,2)
        x = ((x-MEAN)/STD).to(dev)
        e = enc(x); feats.append(torch.cat([e["cls"], e["patch"].mean(1)],1).float().cpu().numpy()); buf.clear()
    # Read ONE region per spot (box + crop margin), crop cells in-memory -> ~34x fewer WSI reads.
    margin = half + 1
    for i,(cx,cy) in enumerate(coords):
        if barcodes[i] not in bc2row: continue
        inbox = cen[(cen[:,0]>=cx)&(cen[:,0]<cx+box)&(cen[:,1]>=cy)&(cen[:,1]<cy+box)]
        if len(inbox)==0: continue
        rx, ry = int(cx)-margin, int(cy)-margin
        rw = box + 2*margin
        rx=min(max(rx,0),max(W-rw,0)); ry=min(max(ry,0),max(H-rw,0))
        region = np.asarray(slide.read_region((rx,ry),0,(rw,rw)).convert("RGB"), np.uint8)
        for gx,gy in inbox:
            lx,ly = int(round(gx))-rx-half, int(round(gy))-ry-half
            lx=min(max(lx,0),rw-crop); ly=min(max(ly,0),rw-crop)
            tile = region[ly:ly+crop, lx:lx+crop]
            if crop!=224:
                tile = np.asarray(Image.fromarray(tile).resize((224,224)), np.uint8)
            buf.append(tile); cell_xy.append((float(gx),float(gy)))
            if len(buf)>=256: flush()
        ptr.append(ptr[-1]+len(inbox))
        exprs.append(np.log1p(X[bc2row[barcodes[i]], gidx].astype(np.float32)))
    flush()
    cf = np.concatenate(feats,0) if feats else np.zeros((0,1536),np.float32)
    np.savez(f"{OUT}/visium_{s}.npz", cell_features=cf.astype(np.float32),
             bag_ptr=np.asarray(ptr,np.int64), expr=np.stack(exprs), genes=np.asarray(PANEL_OK),
             mpp=mpp, crop=crop, box=box)
    np.save(f"{OUT}/coords_{s}.npy", np.asarray(cell_xy, np.float32))
    print(f"[{s}] {len(exprs)} spots, {cf.shape[0]} cells, mpp={mpp:.3f} crop={crop}px box={box}", flush=True)


def main():
    global HEST, PANEL, OUT, PHYS_UM, MAX_SPOTS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--hest-dir", default=HEST, required=not bool(HEST))
    ap.add_argument("--panel-file", required=True)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--h0-checkpoint", default=os.environ.get("H0MINI_CHECKPOINT", "H0-mini/pytorch_model.bin"))
    ap.add_argument("--phys-um", type=float, default=PHYS_UM, help="physical crop window (um) before resize to 224")
    ap.add_argument("--max-spots", type=int, default=0, help="deterministic spot subsample per slide (0 = all)")
    ap.add_argument("--device", default="cuda:1"); a = ap.parse_args()
    HEST = a.hest_dir; OUT = a.out; PHYS_UM = a.phys_um; MAX_SPOTS = a.max_spots
    print(f"PHYS_UM = {PHYS_UM} um, MAX_SPOTS = {MAX_SPOTS}", flush=True)
    PANEL = [x.strip() for x in open(a.panel_file) if x.strip() and not x.startswith("#")]
    print(f"panel: {len(PANEL)} genes from {a.panel_file}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(a.device)
    enc = H0MiniEncoder(checkpoint_path=a.h0_checkpoint, trainable=False).to(dev).eval()
    failures = []
    for sample in a.samples:
        if os.path.exists(f"{OUT}/visium_{sample}.npz"):
            print(f"[{sample}] cached")
            continue
        try:
            extract_visium(sample, enc, dev)
        except Exception as exc:
            failures.append(sample)
            print(f"[{sample}] FAIL {exc}", flush=True)
    if failures:
        raise RuntimeError(
            f"Feature extraction failed for {len(failures)} samples: {failures}"
        )
    print("VISIUM_H0_DONE", flush=True)


if __name__ == "__main__":
    main()
