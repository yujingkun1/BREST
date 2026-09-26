#!/usr/bin/env python3
"""Merge independently scheduled Visium fold-result JSON artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np

from brest.utils import regression_metrics


def _sample_key(row: dict) -> tuple[str, int]:
    match = re.match(r"^(.*?)(\d+)$", row["sample"])
    return (match.group(1), int(match.group(2))) if match else (row["sample"], -1)


def merge(paths: list[str]) -> dict:
    payloads = [json.load(open(path)) for path in paths]
    rows = [row for payload in payloads for row in payload["folds"]]
    for row in rows:
        if not all(name in row for name in ("mae", "mse", "rvd")):
            artifact = np.load(row["prediction"])
            row.update(regression_metrics(artifact["prediction"], artifact["target"]))
    names = [row["sample"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate fold samples across input artifacts")
    rows.sort(key=_sample_key)
    overall = np.asarray([row["overall_pearson"] for row in rows], dtype=float)
    gene = np.asarray([row["mean_gene_pearson"] for row in rows], dtype=float)
    mae = np.asarray([row["mae"] for row in rows], dtype=float)
    mse = np.asarray([row["mse"] for row in rows], dtype=float)
    rvd = np.asarray([row["rvd"] for row in rows], dtype=float)
    first = payloads[0]
    return {
        "dataset": first["dataset"],
        "platform": first["platform"],
        "image_encoder": first.get("image_encoder"),
        "encoder_lr": first.get("encoder_lr"),
        "head_lr": first.get("head_lr"),
        "seed": first.get("seed"),
        "genes": first["genes"],
        "folds": rows,
        "complete": all(payload.get("complete", False) for payload in payloads),
        "summary": {
            "overall_mean": float(overall.mean()),
            "overall_std": float(overall.std()),
            "gene_mean": float(gene.mean()),
            "gene_std": float(gene.std()),
            "mae_mean": float(mae.mean()),
            "mae_std": float(mae.std()),
            "mse_mean": float(mse.mean()),
            "mse_std": float(mse.std()),
            "rvd_mean": float(rvd.mean()),
            "rvd_std": float(rvd.std()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    while args.wait:
        if all(os.path.exists(path) for path in args.inputs):
            payloads = [json.load(open(path)) for path in args.inputs]
            if all(payload.get("complete", False) for payload in payloads):
                break
        time.sleep(args.poll_seconds)
    result = merge(args.inputs)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    temporary = f"{args.output}.tmp"
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2)
    os.replace(temporary, args.output)
    summary = result["summary"]
    print(
        f"overall={summary['overall_mean']:.4f}+/-{summary['overall_std']:.4f} "
        f"gene={summary['gene_mean']:.4f}+/-{summary['gene_std']:.4f} "
        f"mae={summary['mae_mean']:.4f} mse={summary['mse_mean']:.4f} "
        f"rvd={summary['rvd_mean']:.4f} folds={len(result['folds'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
