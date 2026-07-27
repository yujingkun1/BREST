"""Delaunay spatial graphs -- a hyperparameter-free neighbour structure.

A 2-D Delaunay triangulation gives a parameter-free neighbour graph (no ``k`` for
kNN, no radius) over cells / spots. Degenerate inputs (fewer than three points or
a collinear cloud QHull rejects) fall back to a simple chain so callers never have
to special-case tiny components.
"""
from __future__ import annotations

import numpy as np


def build_delaunay_edges(coords: np.ndarray) -> np.ndarray:
    """Unique undirected edges ``(i < j)`` of a 2-D Delaunay triangulation. ``coords`` is (N, 2)."""
    coords = np.asarray(coords, dtype=np.float64)
    n = int(coords.shape[0])
    if n <= 1:
        return np.empty((0, 2), dtype=np.int64)
    if n == 2:
        return np.array([[0, 1]], dtype=np.int64)
    try:
        from scipy.spatial import Delaunay

        tri = Delaunay(coords)
        edges: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = int(simplex[a]), int(simplex[b])
                    if i > j:
                        i, j = j, i
                    edges.add((i, j))
        if edges:
            return np.array(sorted(edges), dtype=np.int64)
    except Exception:
        pass
    # Collinear / degenerate: order along the dominant axis and chain.
    axis = 0 if np.ptp(coords[:, 0]) >= np.ptp(coords[:, 1]) else 1
    order = np.argsort(coords[:, axis])
    edges = np.stack([order[:-1], order[1:]], axis=1)
    edges.sort(axis=1)
    return edges.astype(np.int64)
