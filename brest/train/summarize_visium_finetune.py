#!/usr/bin/env python3
"""Create cancer-level and fold-level tables from Visium full-FT results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from brest.utils import regression_metrics

METRICS = (
    ("overall_pearson", "Overall PCC"),
    ("mean_gene_pearson", "Gene PCC"),
    ("mae", "MAE"),
    ("mse", "MSE"),
    ("rvd", "RVD"),
)


def _enrich(row: dict) -> dict:
    row = dict(row)
    if not all(name in row for name in ("mae", "mse", "rvd")):
        artifact = np.load(row["prediction"])
        row.update(regression_metrics(artifact["prediction"], artifact["target"]))
    return row


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    folds = []
    for path in args.inputs:
        payload = json.load(open(path))
        rows = [_enrich(row) for row in payload["folds"]]
        cancer = payload["dataset"]
        cache = payload.get("crop_cache") or {}
        summary = {
            "cancer": cancer,
            "folds": len(rows),
            "genes": payload["genes"],
            "max_cells_per_spot": cache.get("max_cells_per_spot", "all"),
            "max_spots_per_sample": cache.get("max_spots_per_sample", "all"),
        }
        for name, _ in METRICS:
            values = np.asarray([row[name] for row in rows], dtype=float)
            summary[f"{name}_mean"] = float(values.mean())
            summary[f"{name}_std"] = float(values.std())
        summaries.append(summary)
        for row in rows:
            folds.append({
                "cancer": cancer,
                "sample": row["sample"],
                "best_epoch": row["best_epoch"],
                **{name: row[name] for name, _ in METRICS},
            })

    summary_fields = list(summaries[0])
    with (output_dir / "visium_fullft_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    fold_fields = list(folds[0])
    with (output_dir / "visium_fullft_folds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fold_fields)
        writer.writeheader()
        writer.writerows(folds)

    lines = [
        "# Visium H0-mini full-fine-tuning results",
        "",
        "Metrics are computed in de-normalised log1p expression space. Lower is better for MAE, MSE, and RVD.",
        "",
        "| Cancer | Folds | Genes | Cells/spot | Spots/sample | "
        + " | ".join(label for _, label in METRICS) + " |",
        "|---|---:|---:|---:|---:|" + "---:|" * len(METRICS),
    ]
    for row in summaries:
        metrics = [
            _fmt(row[f"{name}_mean"], row[f"{name}_std"])
            for name, _ in METRICS
        ]
        lines.append(
            f"| {row['cancer']} | {row['folds']} | {row['genes']} | "
            f"{row['max_cells_per_spot']} | {row['max_spots_per_sample']} | "
            + " | ".join(metrics) + " |"
        )
    lines.extend([
        "",
        "## Fold-level results",
        "",
        "| Cancer | Sample | Best epoch | " + " | ".join(label for _, label in METRICS) + " |",
        "|---|---|---:|" + "---:|" * len(METRICS),
    ])
    for row in folds:
        lines.append(
            f"| {row['cancer']} | {row['sample']} | {row['best_epoch']} | "
            + " | ".join(f"{row[name]:.4f}" for name, _ in METRICS) + " |"
        )
    (output_dir / "visium_fullft_table.md").write_text("\n".join(lines) + "\n")
    print(output_dir / "visium_fullft_table.md", flush=True)


if __name__ == "__main__":
    main()
