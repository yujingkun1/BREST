# Preprocessing scripts

Only cache-building utilities live in this directory:

| Script | Purpose |
|---|---|
| `preprocess/h0mini.py` | local H0-mini encoder wrapper |
| `preprocess/extract_visium_h0.py` | Visium cell-feature and spot-label caches |
| `preprocess/extract_bulk_cells.py` | bulk WSI per-cell feature caches |
| `preprocess/build_bulk_subgraphs.py` | tiled bulk Delaunay graph caches |

Training, transfer, ablation, and pseudo-spot experiments use the installable
entry points under `brest.train`; there are no machine-specific shell
launchers in the public repository.

See [`../docs/preprocessing.md`](../docs/preprocessing.md) for commands and
cache schemas.
