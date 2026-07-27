"""Bulk-covariance completion of off-panel genes."""
from __future__ import annotations

import numpy as np


def bulk_covariance(bulk_expr: np.ndarray) -> np.ndarray:
    """Estimate gene covariance from a ``[samples, genes]`` bulk matrix."""
    bz = (bulk_expr - bulk_expr.mean(0)) / (bulk_expr.std(0) + 1e-6)
    bz = bz - bz.mean(0)
    return np.cov(bz.T).astype(np.float32)


def blup_map(
    Sigma: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Precompute the ridge-regularized bulk completion map from ``O`` to ``H``."""
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    O = np.asarray(O, dtype=np.int64)
    H = np.asarray(H, dtype=np.int64)
    Soo = Sigma[np.ix_(O, O)]
    Soh = Sigma[np.ix_(O, H)]
    return np.linalg.solve(Soo + lam * np.eye(len(O), dtype=Soo.dtype), Soh)


def impute_blup(
    X_O: np.ndarray,
    Sigma: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Complete held-out genes with a precomputed bulk covariance prior."""
    return np.asarray(X_O) @ blup_map(Sigma, O, H, lam)


def estimate_prediction_ridge(
    prediction_z: np.ndarray,
    target_z: np.ndarray,
    observed: np.ndarray,
    *,
    minimum: float = 0.05,
    maximum: float = 30.0,
) -> float:
    """Estimate prediction noise using observed training genes only.

    Inputs must already use the same gene-wise standardized expression space as
    the bulk covariance. No held-out-gene values are consulted.
    """
    prediction_z = np.asarray(prediction_z, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.int64)
    if prediction_z.shape != target_z.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction_z.shape} vs {target_z.shape}"
        )
    if observed.size == 0:
        raise ValueError("observed must contain at least one gene index")
    error = prediction_z[:, observed] - target_z[:, observed]
    return float(np.clip(np.mean(np.square(error)), minimum, maximum))

