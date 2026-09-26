#!/usr/bin/env python3
"""Cache aligned uint8 Visium cell crops for end-to-end H0 fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import h5py
import numpy as np
import anndata as ad
import scipy.sparse as sp
from PIL import Image

from _helpers import _SimpleWSI, _load_cell_centroids, decode_string


def open_slide(path: str):
    try:
        import openslide

        return openslide.OpenSlide(path)
    except Exception:
        return _SimpleWSI(path)


def _partial(path: Path) -> Path:
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def cache_sample(args, sample: str, panel: list[str]) -> dict:
    paths = {
        "crops": Path(args.out) / f"crops_{sample}.npy",
        "coords": Path(args.out) / f"coords_{sample}.npy",
        "bagptr": Path(args.out) / f"bagptr_{sample}.npy",
        "expr": Path(args.out) / f"expr_{sample}.npy",
    }
    if all(path.exists() for path in paths.values()):
        bag_ptr = np.load(paths["bagptr"], mmap_mode="r")
        crops = np.load(paths["crops"], mmap_mode="r")
        print(f"[{sample}] cached", flush=True)
        return {
            "sample": sample,
            "cached": True,
            "spots": int(len(bag_ptr) - 1),
            "cells": int(len(crops)),
        }

    with h5py.File(f"{args.hest_dir}/patches/{sample}.h5", "r") as handle:
        spot_coords = np.asarray(handle["coords"][:], dtype=np.float64)
        barcodes = [decode_string(value) for value in handle["barcode"][:]]
        attributes = dict(handle["img"].attrs)
        mpp = float(attributes["pixel_size"])
        spot_box = int(attributes["patch_size_src"])
    crop_pixels = max(16, int(round(args.phys_um / mpp)))
    half = crop_pixels // 2
    centroids = _load_cell_centroids(args.hest_dir, sample)
    adata = ad.read_h5ad(f"{args.hest_dir}/st/{sample}.h5ad")
    expression = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    gene_index = {str(gene): index for index, gene in enumerate(adata.var_names)}
    missing = [gene for gene in panel if gene not in gene_index]
    if missing:
        raise ValueError(f"{sample}: {len(missing)} panel genes missing; first={missing[:5]}")
    panel_index = [gene_index[gene] for gene in panel]
    barcode_index = {str(barcode): index for index, barcode in enumerate(adata.obs_names)}

    rng = np.random.default_rng([args.seed, zlib.crc32(sample.encode("utf-8"))])
    keep_spots = None
    if args.max_spots and len(spot_coords) > args.max_spots:
        keep_spots = set(rng.choice(len(spot_coords), args.max_spots, replace=False).tolist())
    selected = []
    for index, (cx, cy) in enumerate(spot_coords):
        if barcodes[index] not in barcode_index:
            continue
        if keep_spots is not None and index not in keep_spots:
            continue
        cells = centroids[
            (centroids[:, 0] >= cx) & (centroids[:, 0] < cx + spot_box)
            & (centroids[:, 1] >= cy) & (centroids[:, 1] < cy + spot_box)
        ]
        if not len(cells):
            continue
        if args.max_cells and len(cells) > args.max_cells:
            cells = cells[rng.choice(len(cells), args.max_cells, replace=False)]
        selected.append((index, float(cx), float(cy), cells.astype(np.float32)))

    total_cells = sum(len(item[3]) for item in selected)
    for path in paths.values():
        _partial(path).unlink(missing_ok=True)
    crops = np.lib.format.open_memmap(
        _partial(paths["crops"]), mode="w+", dtype=np.uint8,
        shape=(total_cells, 224, 224, 3),
    )
    coords = np.lib.format.open_memmap(
        _partial(paths["coords"]), mode="w+", dtype=np.float32,
        shape=(total_cells, 2),
    )
    targets = np.empty((len(selected), len(panel)), dtype=np.float32)
    bag_ptr = np.zeros(len(selected) + 1, dtype=np.int64)
    slide = open_slide(f"{args.hest_dir}/wsis/{sample}.tif")
    width, height = slide.level_dimensions[0]
    margin = half + 1
    offset = 0
    try:
        for spot, (index, cx, cy, cells) in enumerate(selected):
            region_width = spot_box + 2 * margin
            rx = min(max(int(cx) - margin, 0), max(width - region_width, 0))
            ry = min(max(int(cy) - margin, 0), max(height - region_width, 0))
            region = np.asarray(
                slide.read_region((rx, ry), 0, (region_width, region_width)).convert("RGB"),
                dtype=np.uint8,
            )
            for cell in cells:
                lx = min(max(int(round(cell[0])) - rx - half, 0), region_width - crop_pixels)
                ly = min(max(int(round(cell[1])) - ry - half, 0), region_width - crop_pixels)
                tile = region[ly : ly + crop_pixels, lx : lx + crop_pixels]
                if crop_pixels != 224:
                    tile = np.asarray(Image.fromarray(tile).resize((224, 224)), dtype=np.uint8)
                crops[offset] = tile
                coords[offset] = cell
                offset += 1
            bag_ptr[spot + 1] = offset
            targets[spot] = np.log1p(
                expression[barcode_index[barcodes[index]], panel_index].astype(np.float32)
            )
    finally:
        close = getattr(slide, "close", None)
        if close is not None:
            close()
    crops.flush()
    coords.flush()
    del crops, coords
    np.save(_partial(paths["bagptr"]), bag_ptr)
    np.save(_partial(paths["expr"]), targets)
    for path in paths.values():
        os.replace(_partial(path), path)
    print(
        f"[{sample}] {len(selected)} spots, {total_cells} crops, "
        f"mpp={mpp:.3f} crop={crop_pixels}px",
        flush=True,
    )
    return {"sample": sample, "spots": len(selected), "cells": total_cells}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--hest-dir", required=True)
    parser.add_argument("--panel-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--phys-um", type=float, default=56.0)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--max-spots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    panel = [
        line.strip() for line in open(args.panel_file)
        if line.strip() and not line.startswith("#")
    ]
    gene_path = Path(args.out) / "genes.npy"
    if gene_path.exists() and not np.array_equal(
        np.load(gene_path).astype(str), np.asarray(panel)
    ):
        raise ValueError(f"existing gene panel differs in {gene_path}")
    np.save(gene_path, np.asarray(panel))
    records = [cache_sample(args, sample, panel) for sample in args.samples]
    config = {
        "samples": args.samples,
        "panel_file": os.path.abspath(args.panel_file),
        "hest_dir": os.path.abspath(args.hest_dir),
        "physical_crop_um": args.phys_um,
        "max_cells_per_spot": args.max_cells,
        "max_spots_per_sample": args.max_spots,
        "seed": args.seed,
        "records": records,
    }
    with (Path(args.out) / "cache_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    print("VISIUM_CROPS_DONE", flush=True)


if __name__ == "__main__":
    main()
