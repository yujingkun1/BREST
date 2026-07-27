#!/usr/bin/env python3
"""Build the small, redistributable BREST demonstration cohort.

The artifact is synthetic by design: it exercises the real BREST graph model
and bulk-prior completion code without redistributing HEST images, H0-mini
weights, or patient expression. Gene labels are drawn from the standard COAD
panel, but every numerical value is generated from the fixed seed below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEED = 2027
SLIDE_NAMES = np.array(["train_A", "train_B", "validation", "test"])
SLIDE_SPLITS = np.array(["train", "train", "validation", "test"])
GENES = np.array(
    [
        "ACTB", "ACTN1", "ACTN4", "ANXA2", "AQP3", "ARPC1B", "ATP2A2", "B2M",
        "BSG", "CALML5", "CAPG", "CD24", "CD44", "CD63", "CD81", "CDH1",
        "CDH3", "CLCA2", "CLIC1", "COL17A1", "CST3", "CTNNA1", "CTNNB1", "CTSB",
        "CTSD", "DSC3", "DSG1", "DSG3", "DSP", "ENO1", "EPAS1", "EZR",
        "FABP5", "FGFR3", "FLNA", "FSCN1", "GJA1", "GJB2", "GPNMB", "GSTP1",
        "HIF1A", "IFI27", "IFITM3", "ITGB4", "IVL", "JUP", "KRT10", "KRT14",
        "KRT16", "KRT5", "KRT6B", "LGALS3", "LGALS7", "NDRG1", "PKP1", "PKP3",
        "S100A2", "S100A8", "S100A9", "SDC1", "TGM1", "TP63", "VIM", "VMP1",
    ]
)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    latent_dim = 8
    input_dim = 1536
    spots_per_slide = 64
    cells_per_spot = 6
    bulk_samples = 256

    feature_basis = rng.normal(0, 0.38, size=(latent_dim, input_dim)).astype(np.float32)
    gene_basis = rng.normal(0, 0.75, size=(latent_dim, len(GENES))).astype(np.float32)
    gene_basis[0, :16] += 0.8
    gene_basis[1, 16:32] -= 0.7
    gene_basis[2, 32:48] += 0.9
    gene_basis[3, 48:] -= 0.8
    gene_bias = rng.normal(0.1, 0.25, size=len(GENES)).astype(np.float32)

    features: list[np.ndarray] = []
    cell_coords: list[np.ndarray] = []
    expressions: list[np.ndarray] = []
    spot_coords: list[np.ndarray] = []
    slide_index: list[int] = []
    bag_ptr = [0]

    grid = np.stack(np.meshgrid(np.linspace(0.05, 0.95, 8), np.linspace(0.05, 0.95, 8)), -1)
    grid = grid.reshape(-1, 2)
    for slide in range(len(SLIDE_NAMES)):
        theta = (slide - 1.5) * 0.045
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float32,
        )
        xy = (grid - 0.5) @ rotation.T + 0.5
        slide_shift = rng.normal(0, 0.08, size=latent_dim)
        for x, y in xy:
            radial = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
            spot_latent = np.array(
                [
                    2 * x - 1,
                    2 * y - 1,
                    np.sin(np.pi * x),
                    np.cos(np.pi * y),
                    np.sin(2 * np.pi * (x + y)),
                    2 * radial - 0.7,
                    (2 * x - 1) * (2 * y - 1),
                    1.0,
                ],
                dtype=np.float32,
            )
            spot_latent += slide_shift
            cell_latent = spot_latent + rng.normal(
                0, 0.10, size=(cells_per_spot, latent_dim)
            ).astype(np.float32)
            jitter = rng.normal(0, 8.0, size=(cells_per_spot, 2)).astype(np.float32)
            coords = np.array([x * 800, y * 800], dtype=np.float32) + jitter
            feats = cell_latent @ feature_basis
            feats += rng.normal(0, 0.035, size=feats.shape).astype(np.float32)
            raw_expression = cell_latent.mean(0) @ gene_basis + gene_bias
            raw_expression += rng.normal(0, 0.055, size=len(GENES)).astype(np.float32)

            features.append(feats.astype(np.float16))
            cell_coords.append(coords)
            expressions.append(softplus(raw_expression).astype(np.float32))
            spot_coords.append(coords.mean(0))
            slide_index.append(slide)
            bag_ptr.append(bag_ptr[-1] + cells_per_spot)

    bulk_latent = rng.normal(0, 0.85, size=(bulk_samples, latent_dim)).astype(np.float32)
    bulk_latent[:, 2] += 0.7 * bulk_latent[:, 0]
    bulk_latent[:, 3] -= 0.6 * bulk_latent[:, 1]
    bulk_latent[:, 7] = 1.0
    bulk_raw = bulk_latent @ gene_basis + gene_bias
    bulk_raw += rng.normal(0, 0.08, size=bulk_raw.shape).astype(np.float32)
    bulk_expression = softplus(bulk_raw).astype(np.float32)

    heldout = np.sort(np.random.RandomState(0).choice(len(GENES), 12, replace=False))
    return {
        "cell_features": np.concatenate(features),
        "cell_coords": np.concatenate(cell_coords),
        "bag_ptr": np.asarray(bag_ptr, dtype=np.int64),
        "expression": np.stack(expressions),
        "spot_coords": np.stack(spot_coords).astype(np.float32),
        "slide_index": np.asarray(slide_index, dtype=np.int64),
        "slide_names": SLIDE_NAMES,
        "slide_splits": SLIDE_SPLITS,
        "genes": GENES,
        "bulk_expression": bulk_expression,
        "heldout_indices": heldout.astype(np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "demo_data",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output = args.output_dir / "brest_toy_coad.npz"
    payload = build_payload()
    np.savez_compressed(output, **payload)
    (args.output_dir / "genes.txt").write_text("\n".join(payload["genes"].tolist()) + "\n")

    manifest = {
        "name": "BREST synthetic COAD-panel toy cohort",
        "version": 1,
        "seed": SEED,
        "artifact": output.name,
        "sha256": sha256(output),
        "license": "CC0-1.0",
        "provenance": (
            "All numerical values are generated locally by demo/build_demo_data.py. "
            "Gene symbols are labels selected from the BREST standard COAD panel."
        ),
        "scope": "Functional demonstration only; not a benchmark or biological dataset.",
        "shapes": {key: list(value.shape) for key, value in payload.items()},
        "splits": dict(zip(SLIDE_NAMES.tolist(), SLIDE_SPLITS.tolist())),
        "heldout_gene_count": int(len(payload["heldout_indices"])),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    size_mb = output.stat().st_size / 1024**2
    print(f"Wrote {output} ({size_mb:.2f} MiB)")
    print(f"SHA256 {manifest['sha256']}")


if __name__ == "__main__":
    main()
