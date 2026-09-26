# Preprocessing and cache formats

BREST training uses frozen, per-cell H0-mini features. Raw images, patient
expression matrices, CellViT outputs, and H0-mini weights are not included.

## External components

- **CellViT** supplies cell centroids for each WSI.
- **H0-mini** (`bioptimus/H0-mini`) encodes each cell crop. BREST concatenates
  the 768-dimensional CLS token and the mean of the patch tokens to obtain a
  1,536-dimensional feature.
- **HEST** supplies the full-scale spatial data. Its data are not included.

The H0-mini normalization is
`mean=(0.707223, 0.578729, 0.703617)` and
`std=(0.211883, 0.230117, 0.177517)`.

## Visium

Run:

```bash
python scripts/preprocess/extract_visium_h0.py \
  --samples TENX152 TENX92 \
  --hest-dir /path/to/hest \
  --panel-file /path/to/genes.txt \
  --h0-checkpoint /path/to/H0-mini/pytorch_model.bin \
  --out cache/visium \
  --phys-um 112 \
  --device cuda:0
```

Each sample produces:

- `visium_<sample>.npz`
  - `cell_features`: `[N_cells, 1536]`
  - `bag_ptr`: CSR pointers delimiting the cells in every spot
  - `expr`: `[N_spots, G]` log1p expression
  - `genes`: `[G]` gene symbols
- `coords_<sample>.npy`: `[N_cells, 2]` cell coordinates

For end-to-end H0-mini fine-tuning, retain the aligned pre-encoding arrays:

- `crops_<sample>.npy`: `[N_cells, 224, 224, 3]` uint8 RGB crops
- `coords_<sample>.npy`: `[N_cells, 2]` cell coordinates
- `bagptr_<sample>.npy`: CSR spot boundaries
- `expr_<sample>.npy`: `[N_spots, G]` log1p expression
- `genes.npy`: `[G]` gene symbols

The fine-tuning runner memory-maps the crop arrays and loads one spot at a
time; crop order must match `coords` and `bagptr` exactly.

Build that cache directly from HEST and CellViT segmentations with:

```bash
python scripts/preprocess/cache_visium_crops.py \
  --samples NCBI642 NCBI643 \
  --hest-dir /path/to/hest_data \
  --panel-file /path/to/genes.txt \
  --out cache/lihc_crops \
  --native-size
```

`--native-size` stores the full, unresized physical field of view for every
cell and performs the same PIL resize lazily when a training spot is loaded.
This losslessly reduces cache size for sub-224-pixel source crops. Optional
`--max-cells` and `--max-spots` caps remain available for exploratory runs;
their values, storage mode, and per-sample cell counts are recorded in
`cache_config.json` and propagated into fine-tuning result artifacts.

## Xenium

The Xenium runner consumes an aligned cache and metadata file:

- `<sample>_all_cls_patch_mean.npz`
  - `features`: `[N_cells, 1536]`
  - `coords`: `[N_cells, 2]`
  - `expression`: `[N_cells, G]`
- `<sample>_cell_metadata.parquet`
  - one row per cache entry
  - integer column `position_fold_5` with values 0--4
- newline-delimited gene list with exactly `G` symbols

The arrays and metadata rows must have identical cell order. Expression should
use the same normalization as the intended experiment.

## Bulk WSI

First extract per-slide cell features:

```bash
python scripts/preprocess/extract_bulk_cells.py \
  --wsi-dir /path/to/wsis \
  --seg-dir /path/to/cellvit_outputs \
  --tpm /path/to/tpm.csv \
  --out cache/bulk_cells \
  --h0-checkpoint /path/to/H0-mini/pytorch_model.bin \
  --device cuda:0
```

Then build tiled Delaunay subgraphs:

```bash
python scripts/preprocess/build_bulk_subgraphs.py \
  --npz-dir cache/bulk_cells \
  --out-dir cache/bulk_graphs \
  --tile-um 256 \
  --max-total-cells 40000
```

The graph builder creates patient-disjoint train/test graph dictionaries and a
slide-to-patient mapping. All slides from one TCGA case remain in the same
partition. `brest.train.bulk` pairs these with a panel-filtered TPM CSV. TPM
rows must match the gene-list order and columns must identify TCGA patients.

## Data leakage

Compute target normalization statistics and prediction calibration on the
training partition only. Bulk covariance may come from an external cohort, but
held-out spatial genes and test-fold predictions must not be used to choose the
completion map.
