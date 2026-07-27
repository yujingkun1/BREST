"""Small reusable helpers for the executed BREST demonstration notebook."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from brest.data import build_bag_graphs
from brest.downstream import bulk_covariance, impute_blup
from brest.models import UnifiedExpressionModel
from brest.utils import completion_pearsons, mean_gene_pearson, overall_pearson, safe_pearson


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_demo_data(path: str | Path) -> dict[str, np.ndarray]:
    """Load the self-contained demo NPZ and return ordinary arrays."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def build_demo_graphs(data: dict[str, np.ndarray]):
    return build_bag_graphs(
        {
            "cell_features": data["cell_features"],
            "cell_coords": data["cell_coords"],
            "bag_ptr": data["bag_ptr"],
            "spot_expression": data["expression"],
        }
    )


def split_indices(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    spot_splits = data["slide_splits"][data["slide_index"]]
    return {
        split: np.flatnonzero(spot_splits == split)
        for split in ("train", "validation", "test")
    }


@torch.no_grad()
def predict(model, graphs, device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    chunks = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        chunks.append(model(batch.to(device))["prediction"].cpu().numpy())
    return np.concatenate(chunks)


def fit_demo_model(
    data: dict[str, np.ndarray],
    *,
    device: str | torch.device,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
) -> dict:
    """Train the reduced demo configuration and evaluate the untouched test slide."""
    seed_everything(seed)
    device = torch.device(device)
    graphs = build_demo_graphs(data)
    indices = split_indices(data)
    by_split = {name: [graphs[i] for i in rows] for name, rows in indices.items()}

    train_expression = data["expression"][indices["train"]].astype(np.float32)
    mean = train_expression.mean(0)
    std = train_expression.std(0) + 1e-6
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    model = UnifiedExpressionModel(
        in_dim=data["cell_features"].shape[1],
        num_genes=len(data["genes"]),
        hidden_dim=64,
        n_layers=1,
        heads=4,
        dropout=0.1,
        granularity="spot",
        gene_chunk=32,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(by_split["train"], batch_size=batch_size, shuffle=True)

    history = []
    best_state = None
    best_gene = -np.inf
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction_z = model(batch)["prediction"]
            target_z = (batch.y - mean_t) / std_t
            loss = F.mse_loss(prediction_z, target_z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * batch.num_graphs
            seen += batch.num_graphs

        validation_z = predict(model, by_split["validation"], device, batch_size)
        validation = validation_z * std + mean
        validation_target = data["expression"][indices["validation"]]
        val_overall = overall_pearson(validation, validation_target)
        val_gene = mean_gene_pearson(validation, validation_target)
        history.append(
            {
                "epoch": epoch,
                "train_mse": total_loss / max(seen, 1),
                "validation_overall_pcc": val_overall,
                "validation_gene_pcc": val_gene,
            }
        )
        if val_gene > best_gene:
            best_gene = val_gene
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")
    model.load_state_dict(best_state)
    test_z = predict(model, by_split["test"], device, batch_size)
    test_prediction = test_z * std + mean
    test_target = data["expression"][indices["test"]]
    test_metrics = {
        "Overall PCC": overall_pearson(test_prediction, test_target),
        "Gene PCC": mean_gene_pearson(test_prediction, test_target),
    }
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "train_mean": mean,
        "train_std": std,
        "indices": indices,
        "test_prediction": test_prediction,
        "test_target": test_target,
        "test_metrics": test_metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def evaluate_bulk_completion(
    data: dict[str, np.ndarray],
    run: dict,
    *,
    measured_ridge: float = 10.0,
    prediction_ridge: float = 1.0,
) -> dict:
    """Evaluate Direct, BLUP-true, and BLUP-pred on the fixed held-out genes."""
    heldout = data["heldout_indices"].astype(np.int64)
    observed = np.setdiff1d(np.arange(len(data["genes"])), heldout)
    covariance = bulk_covariance(data["bulk_expression"])

    mean = run["train_mean"]
    std = run["train_std"]
    target = run["test_target"]
    direct = run["test_prediction"]
    target_z = (target - mean) / std
    direct_z = (direct - mean) / std

    completion_z = {
        "Direct": direct_z[:, heldout],
        "BLUP-true": impute_blup(
            target_z[:, observed], covariance, observed, heldout, measured_ridge
        ),
        "BLUP-pred": impute_blup(
            direct_z[:, observed], covariance, observed, heldout, prediction_ridge
        ),
    }
    rows = []
    restored = {}
    per_gene = {}
    for method, prediction_z in completion_z.items():
        overall, gene = completion_pearsons(
            prediction_z, target[:, heldout], mean[heldout], std[heldout]
        )
        prediction = prediction_z * std[heldout] + mean[heldout]
        restored[method] = prediction
        per_gene[method] = np.array(
            [safe_pearson(prediction[:, j], target[:, heldout[j]]) for j in range(len(heldout))]
        )
        rows.append({"Method": method, "Overall PCC": overall, "Gene PCC": gene})
    return {
        "rows": rows,
        "heldout": heldout,
        "observed": observed,
        "target": target[:, heldout],
        "predictions": restored,
        "per_gene": per_gene,
        "measured_ridge": measured_ridge,
        "prediction_ridge": prediction_ridge,
    }
