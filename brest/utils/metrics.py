"""Regression metrics used by BREST.

overall_pearson    -- correlation over the flattened [N, G] prediction/target.
mean_gene_pearson  -- mean of the per-gene (column-wise) correlations.
regression_metrics -- PCC, MAE, MSE, and relative variation distance (RVD).

Spatial models are scored in log1p space after de-normalising z-scored predictions
(P = pred_z * sd + mu); model selection takes the epoch with the highest
mean_gene_pearson.
"""
from __future__ import annotations

import numpy as np


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with deterministic handling of constant arrays."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or x.size != y.size:
        raise ValueError(f"Pearson inputs must have the same non-zero size, got {x.size} and {y.size}")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denom < 1e-12 else float(np.dot(x, y) / denom)


def overall_pearson(P: np.ndarray, T: np.ndarray) -> float:
    """Pearson correlation after flattening prediction and target matrices."""
    P = np.asarray(P)
    T = np.asarray(T)
    if P.shape != T.shape:
        raise ValueError(f"Prediction/target shape mismatch: {P.shape} vs {T.shape}")
    return safe_pearson(P, T)


def _per_gene_pearson(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    Pc = P - P.mean(0, keepdims=True)
    Tc = T - T.mean(0, keepdims=True)
    return (Pc * Tc).sum(0) / (np.sqrt((Pc ** 2).sum(0)) * np.sqrt((Tc ** 2).sum(0)) + 1e-8)


def mean_gene_pearson(P: np.ndarray, T: np.ndarray) -> float:
    return float(np.nanmean(_per_gene_pearson(P, T)))


def gene_pearson_stats(P: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """Mean and std of the per-gene correlation distribution (across genes)."""
    per = _per_gene_pearson(P, T)
    return float(np.nanmean(per)), float(np.nanstd(per))


def regression_metrics(P: np.ndarray, T: np.ndarray) -> dict[str, float]:
    """Return expression-regression metrics in de-normalised log1p space.

    RVD follows the Relative Variation Distance definition from Stem: for each
    non-constant gene, compare predicted and true variance across spots using
    ``((var_pred - var_true) / var_true) ** 2``, then average across genes.
    Genes with effectively zero true variance are excluded because their
    relative variation is undefined.
    """
    prediction = np.asarray(P, dtype=np.float64)
    target = np.asarray(T, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )
    if prediction.ndim != 2 or prediction.size == 0:
        raise ValueError(f"Expected non-empty [spots, genes] matrices, got {prediction.shape}")
    error = prediction - target
    true_variance = target.var(axis=0)
    predicted_variance = prediction.var(axis=0)
    varying = true_variance > 1e-12
    rvd = (
        float(np.mean(((predicted_variance[varying] - true_variance[varying])
                       / true_variance[varying]) ** 2))
        if np.any(varying) else float("nan")
    )
    return {
        "overall_pearson": overall_pearson(prediction, target),
        "mean_gene_pearson": mean_gene_pearson(prediction, target),
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error ** 2)),
        "rvd": rvd,
    }


def completion_pearsons(
    prediction_z: np.ndarray,
    target_log1p: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
) -> tuple[float, float]:
    """Score held-out genes after restoring predictions to log1p expression.

    ``prediction_z`` is in the training-set standardized space. Overall PCC is
    computed on the raw flattened log1p matrices; Gene PCC is the mean
    column-wise correlation. This matches the fixed mask-100 evaluation used in
    the BREST experiments.
    """
    prediction_z = np.asarray(prediction_z, dtype=np.float64)
    target_log1p = np.asarray(target_log1p, dtype=np.float64)
    train_mean = np.asarray(train_mean, dtype=np.float64).reshape(1, -1)
    train_std = np.asarray(train_std, dtype=np.float64).reshape(1, -1)
    if prediction_z.shape != target_log1p.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction_z.shape} vs {target_log1p.shape}"
        )
    if prediction_z.shape[1] != train_mean.shape[1] or train_mean.shape != train_std.shape:
        raise ValueError("Held-out normalization statistics must match the gene dimension")
    prediction_log1p = prediction_z * train_std + train_mean
    return overall_pearson(prediction_log1p, target_log1p), mean_gene_pearson(
        prediction_log1p, target_log1p
    )
