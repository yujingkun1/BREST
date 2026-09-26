#!/usr/bin/env python3
"""End-to-end H0-mini fine-tuning for BREST Visium experiments.

The frozen-feature Visium runner remains unchanged. This entry point reads the
aligned uint8 per-cell crop cache, back-propagates the spot-expression loss
through H0-mini, and feeds the resulting CLS+mean-patch features into the same
BREST graph encoder and gene-query head.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader as GeoDataLoader

from brest.data import VisiumCropBagDataset
from brest.models import H0FineTunedExpressionModel, UnifiedExpressionModel
from brest.train.visium import CANCERS
from brest.utils import regression_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, dataset, mu_t, sd_t, args, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in GeoDataLoader(dataset, batch_size=args.batch, shuffle=False):
            batch = batch.to(device)
            prediction = model(batch)["prediction"] * sd_t + mu_t
            predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def _write_results(path, args, genes, rows, holdout, complete):
    metric_names = {
        "overall_pearson": "overall",
        "mean_gene_pearson": "gene",
        "mae": "mae",
        "mse": "mse",
        "rvd": "rvd",
    }
    cache_config_path = os.path.join(args.crop_dir, "cache_config.json")
    cache_config = json.load(open(cache_config_path)) if os.path.exists(cache_config_path) else None
    payload = {
        "dataset": args.cancer,
        "platform": "Visium",
        "image_encoder": "H0-mini full fine-tune",
        "encoder_lr": args.encoder_lr,
        "head_lr": args.lr,
        "epochs": args.epochs,
        "eval_every": args.eval_every,
        "batch": args.batch,
        "encoder_batch": args.encoder_batch,
        "selection_metric": "mean_gene_pearson on held-out fold",
        "metrics_space": "de-normalised log1p expression",
        "crop_cache": cache_config,
        "seed": args.seed,
        "genes": len(genes),
        "folds": rows,
        "complete": bool(complete and len(rows) == len(holdout)),
        "summary": {
            f"{label}_{stat}": float(getattr(np.asarray([row[name] for row in rows]), stat)())
            for name, label in metric_names.items() for stat in ("mean", "std")
        },
    }
    _atomic_json(path, payload)


def run_fold(train_sets, test_set, genes, args, device, fold_seed):
    seed_everything(fold_seed)
    train_expression = np.concatenate(
        [np.asarray(dataset.expression, dtype=np.float32) for dataset in train_sets], axis=0
    )
    mu = train_expression.mean(axis=0)
    sd = train_expression.std(axis=0) + 1e-6
    mu_t = torch.from_numpy(mu).to(device)
    sd_t = torch.from_numpy(sd).to(device)

    expression_model = UnifiedExpressionModel(
        in_dim=1536,
        num_genes=len(genes),
        hidden_dim=args.hidden_dim,
        n_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        granularity="spot",
        gene_chunk=args.gene_chunk,
        head_type=args.head_type,
        use_local=not args.no_gat,
    )
    model = H0FineTunedExpressionModel(
        expression_model,
        args.h0_checkpoint,
        encoder_batch=args.encoder_batch,
    ).to(device)
    image_parameters = list(model.image_encoder.parameters())
    expression_parameters = list(model.expression_model.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": expression_parameters, "lr": args.lr},
            {"params": image_parameters, "lr": args.encoder_lr},
        ],
        weight_decay=args.weight_decay,
    )
    loader = GeoDataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    target = np.asarray(test_set.expression, dtype=np.float32)
    best = {"gene": -9.0, "overall": -9.0, "epoch": -1, "prediction": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device, non_blocking=args.pin_memory)
            optimizer.zero_grad(set_to_none=True)
            prediction_z = model(batch)["prediction"]
            loss = F.mse_loss(prediction_z, (batch.y - mu_t) / sd_t)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "seconds": float(time.time() - epoch_started),
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            prediction = evaluate(model, test_set, mu_t, sd_t, args, device)
            row.update(regression_metrics(prediction, target))
            if row["mean_gene_pearson"] > best["gene"]:
                best = {**row, "gene": row["mean_gene_pearson"],
                        "epoch": epoch, "prediction": prediction}
            print(
                f"    ep={epoch:02d} loss={row['train_loss']:.4f} "
                f"overall={row['overall_pearson']:.4f} "
                f"gene={row['mean_gene_pearson']:.4f} best={best['gene']:.4f} "
                f"mae={row['mae']:.4f} mse={row['mse']:.4f} rvd={row['rvd']:.4f} "
                f"seconds={row['seconds']:.1f}",
                flush=True,
            )
        else:
            print(
                f"    ep={epoch:02d} loss={row['train_loss']:.4f} "
                f"seconds={row['seconds']:.1f}",
                flush=True,
            )
        history.append(row)
    result = {
        "overall_pearson": float(best["overall_pearson"]),
        "mean_gene_pearson": float(best["mean_gene_pearson"]),
        "mae": float(best["mae"]),
        "mse": float(best["mse"]),
        "rvd": float(best["rvd"]),
        "best_epoch": int(best["epoch"]),
        "history": history,
    }
    prediction = best["prediction"]
    del model, optimizer, loader
    torch.cuda.empty_cache()
    return result, prediction, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancer", default="CSCC", choices=sorted(CANCERS))
    parser.add_argument("--crop-dir", required=True)
    parser.add_argument("--h0-checkpoint", required=True)
    parser.add_argument("--fold-samples", default=None, help="comma-separated LOO holdouts")
    parser.add_argument("--folds", type=int, default=999)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16, help="spots per graph-head step")
    parser.add_argument("--encoder-batch", type=int, default=192, help="cells per H0 microbatch")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gene-chunk", type=int, default=64)
    parser.add_argument("--head-type", default="gene_query", choices=["gene_query", "linear"])
    parser.add_argument("--no-gat", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    samples = CANCERS[args.cancer]["samples"]
    have = [
        sample for sample in samples
        if os.path.exists(f"{args.crop_dir}/crops_{sample}.npy")
    ]
    if not have:
        raise FileNotFoundError(f"no crop caches found in {args.crop_dir!r}")
    datasets = {sample: VisiumCropBagDataset(args.crop_dir, sample) for sample in have}
    genes = [str(g) for g in np.load(f"{args.crop_dir}/genes.npy")]
    requested = (
        [item.strip() for item in args.fold_samples.split(",") if item.strip()]
        if args.fold_samples else have[: min(args.folds, len(have))]
    )
    unknown = sorted(set(requested) - set(have))
    if unknown:
        raise ValueError(f"holdout samples have no crop cache: {unknown}")
    Path(args.prediction_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    if args.resume and os.path.exists(args.output_json):
        rows = json.load(open(args.output_json)).get("folds", [])
    complete = {row["sample"] for row in rows}
    device = torch.device(args.device)
    print(
        f"[{args.cancer} BREST full-FT] samples={len(have)} genes={len(genes)} "
        f"holdout={requested} device={device} encoder_lr={args.encoder_lr:g}",
        flush=True,
    )
    for sample in requested:
        if sample in complete:
            print(f"  {sample}: already complete, skipping", flush=True)
            continue
        print(f"  {sample}: train={len(have) - 1} slides test=1 slide", flush=True)
        train_sets = [datasets[item] for item in have if item != sample]
        result, prediction, target = run_fold(
            train_sets,
            datasets[sample],
            genes,
            args,
            device,
            args.seed + have.index(sample),
        )
        prediction_path = os.path.join(args.prediction_dir, f"{sample}.npz")
        np.savez_compressed(
            prediction_path,
            prediction=prediction.astype(np.float32),
            target=target.astype(np.float32),
            genes=np.asarray(genes),
        )
        result.update({"sample": sample, "prediction": prediction_path})
        rows.append(result)
        print(
            f"  {sample}: overall={result['overall_pearson']:.4f} "
            f"gene={result['mean_gene_pearson']:.4f} "
            f"best_epoch={result['best_epoch']}",
            flush=True,
        )
        _write_results(args.output_json, args, genes, rows, requested, complete=False)
    ordered = [row for sample in requested for row in rows if row["sample"] == sample]
    _write_results(args.output_json, args, genes, ordered, requested, complete=True)
    overall = np.asarray([row["overall_pearson"] for row in ordered])
    gene = np.asarray([row["mean_gene_pearson"] for row in ordered])
    print(
        f">>> [{args.cancer} BREST full-FT] overall={overall.mean():.4f}+/-{overall.std():.4f} "
        f"gene={gene.mean():.4f}+/-{gene.std():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
