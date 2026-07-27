"""Assemble PyG graphs from cached H0 features + Delaunay edges.

Two readers feed the unified model:

* ``build_bag_graphs`` -- Visium: a CSR cell-bag cache (cell_features / cell_coords
  / bag_ptr / spot_expression) becomes one intra-spot Delaunay graph per spot, with
  a graph-level expression label (granularity="spot").
* ``spatial_tiles`` + ``build_graph`` -- Xenium: one fold's ~80k cells are tiled into
  spatial sub-graphs of ~``target_size`` cells (so the model gets many steps/epoch);
  Delaunay edges are rebuilt within each tile (granularity="cell").
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from .graph import build_delaunay_edges


def spatial_tiles(coords: np.ndarray, target_size: int) -> list[np.ndarray]:
    """Partition points into ~equal-count spatial tiles (for sub-graph batching)."""
    coords = np.asarray(coords, dtype=np.float64)
    n = int(coords.shape[0])
    if n <= target_size:
        return [np.arange(n, dtype=np.int64)]
    grid = max(1, int(round(np.sqrt(n / float(target_size)))))
    x_edges = np.quantile(coords[:, 0], np.linspace(0.0, 1.0, grid + 1))
    x_bin = np.clip(np.digitize(coords[:, 0], x_edges[1:-1]), 0, grid - 1)
    tiles: list[np.ndarray] = []
    for xi in range(grid):
        xs = np.flatnonzero(x_bin == xi)
        if xs.size == 0:
            continue
        y_edges = np.quantile(coords[xs, 1], np.linspace(0.0, 1.0, grid + 1))
        y_bin = np.clip(np.digitize(coords[xs, 1], y_edges[1:-1]), 0, grid - 1)
        for yi in range(grid):
            sub = xs[y_bin == yi]
            if sub.size:
                tiles.append(sub.astype(np.int64))
    return tiles


def build_graph(payload: dict[str, np.ndarray]) -> Data:
    """Cache payload (features / coords / expression) -> a Delaunay graph Data."""
    coords = np.asarray(payload["coords"], dtype=np.float64)
    edges = build_delaunay_edges(coords)  # (E, 2) unique i<j
    if edges.shape[0] == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = to_undirected(torch.from_numpy(edges.T).long())
    return Data(
        x=torch.from_numpy(np.asarray(payload["features"], dtype=np.float32)),
        edge_index=edge_index,
        pos=torch.from_numpy(coords.astype(np.float32)),
        y=torch.from_numpy(np.asarray(payload["expression"], dtype=np.float32)),
    )


def build_graph_arrays(features: np.ndarray, coords: np.ndarray, expression: np.ndarray) -> Data:
    return build_graph({"features": features, "coords": coords, "expression": expression})


def build_bag_graphs(payload: dict[str, np.ndarray]) -> list[Data]:
    """CSR cell-bag cache -> one intra-spot Delaunay graph per spot (graph-level label)."""
    feats = np.asarray(payload["cell_features"], dtype=np.float32)
    coords = np.asarray(payload["cell_coords"], dtype=np.float64)
    ptr = np.asarray(payload["bag_ptr"], dtype=np.int64)
    expr = np.asarray(payload["spot_expression"], dtype=np.float32)
    graphs: list[Data] = []
    for s in range(len(ptr) - 1):
        lo, hi = int(ptr[s]), int(ptr[s + 1])
        cell_coords = coords[lo:hi]
        edges = build_delaunay_edges(cell_coords)
        if edges.shape[0] == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = to_undirected(torch.from_numpy(edges.T).long())
        graphs.append(
            Data(
                x=torch.from_numpy(feats[lo:hi]),
                edge_index=edge_index,
                pos=torch.from_numpy(cell_coords.astype(np.float32)),
                y=torch.from_numpy(expr[s]).unsqueeze(0),  # [1, G] graph-level label
            )
        )
    return graphs
