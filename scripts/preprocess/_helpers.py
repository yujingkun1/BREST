"""Self-contained I/O helpers for the extraction scripts (vendored; no external
package imports). CellViT-segmentation centroid loading, HEST h5 string decoding,
and a PIL WSI fallback for formats openslide cannot read.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

# Optional on-disk cache for parsed CellViT centroids (override via env var).
CENTROID_CACHE_DIR = os.environ.get("CENTROID_CACHE_DIR", "cache/centroids")


def decode_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_string(value.item())
        if value.size == 1:
            return decode_string(value.reshape(-1)[0])
    return str(value)


def _load_cell_centroids(hest_dir: str, sample_id: str) -> np.ndarray:
    """Per-cell centroids (N,2) from CellViT segmentation (parquet, geojson fallback)."""
    cache_path = os.path.join(CENTROID_CACHE_DIR, f"{sample_id}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    from shapely import wkb
    import pandas as pd

    seg_path = os.path.join(hest_dir, "cellvit_seg", f"{sample_id}_cellvit_seg.parquet")
    if os.path.exists(seg_path):
        geoms = pd.read_parquet(seg_path, columns=["geometry"])["geometry"].tolist()
        centroids = np.empty((len(geoms), 2), dtype=np.float64)
        for idx, geometry in enumerate(geoms):
            polygon = wkb.loads(geometry) if isinstance(geometry, (bytes, bytearray)) else geometry
            centroids[idx] = (polygon.centroid.x, polygon.centroid.y)
    else:  # geojson fallback (Xenium seg)
        import json
        from shapely.geometry import shape
        gj = json.load(open(os.path.join(hest_dir, "cellvit_seg", f"{sample_id}_cellvit_seg.geojson")))
        feats = gj["features"] if isinstance(gj, dict) else gj
        centroids = np.empty((len(feats), 2), dtype=np.float64)
        for idx, ft in enumerate(feats):
            c = shape(ft["geometry"]).centroid
            centroids[idx] = (c.x, c.y)
    Path(CENTROID_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    np.save(cache_path, centroids)
    return centroids


class _SimpleWSI:
    """PIL-backed WSI with an openslide-like read_region (for non-pyramidal images)."""

    def __init__(self, image_path: str) -> None:
        self._image = Image.open(image_path).convert("RGB")
        self.dimensions = self._image.size
        self.level_dimensions = [self.dimensions]
        self.level_downsamples = [1.0]

    def read_region(self, location, level, size) -> Image.Image:
        del level
        x, y = location
        width, height = size
        canvas = Image.new("RGB", (int(width), int(height)), (0, 0, 0))
        left, top = max(0, int(x)), max(0, int(y))
        right = min(self._image.size[0], int(x + width))
        bottom = min(self._image.size[1], int(y + height))
        if right > left and bottom > top:
            canvas.paste(self._image.crop((left, top, right, bottom)), (left - int(x), top - int(y)))
        return canvas

    def close(self) -> None:
        self._image.close()
