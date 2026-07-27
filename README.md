# BREST

BREST predicts bulk, spot-level, and single-cell gene expression from
histology with one cell-centric graph model. H0-mini cell features are linked
with Delaunay edges, encoded by local graph attention and global
self-attention, and decoded by a gene-query head.

This repository contains the model, training entry points, bulk-prior
completion utilities, preprocessing scripts, tests, and a small synthetic
notebook demo. It does not redistribute patient data, pathology images, or
third-party model weights.

## Installation

Python 3.10 is recommended. Install the PyTorch build matching your system,
then install BREST:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[demo,training]"
```

Install the optional preprocessing dependencies only when building feature
caches:

```bash
pip install -e ".[preprocessing]"
```

## Executable demo

The committed notebook trains the real BREST architecture on a deterministic
synthetic cohort and demonstrates prediction-calibrated bulk completion:

```bash
jupyter lab demo/BREST_demo.ipynb
```

Rebuild the synthetic data and execute the notebook from source:

```bash
python demo/build_demo_data.py
python demo/make_notebook.py
jupyter nbconvert --to notebook --execute demo/BREST_demo.ipynb \
  --inplace --ExecutePreprocessor.timeout=600
python demo/validate_demo.py
```

The demo data contain no patient measurements or third-party features. Demo
scores are software checks and must not be reported as benchmark results.

## Architecture

```text
H&E image
  -> CellViT centroids
  -> H0-mini CLS + mean patch-token features (1,536 dimensions)
  -> Delaunay cell graph
  -> local GAT blocks
  -> coordinate MLP + global Transformer blocks
  -> gene-query cellular head
  -> cell, spot, or slide expression
```

`UnifiedExpressionModel` uses the same parameters at every resolution:

| Platform | `granularity` | Output |
|---|---|---|
| Xenium | `cell` | one expression vector per cell |
| Visium | `spot` | mean of cellular predictions in each spot |
| bulk RNA-seq | `slide` | aggregate prediction for each slide |

The public ablation controls are intentionally limited to the two supported
components: `head_type="linear"` removes the gene-query head and
`use_local=False` removes the local GAT.

## Training

Training consumes cached cell features described in
[`docs/preprocessing.md`](docs/preprocessing.md).

```bash
# Visium leave-one-slide-out
python -m brest.train.visium \
  --cancer COAD \
  --sp-dir /path/to/visium_cache \
  --epochs 40 \
  --device cuda:0 \
  --output-json runs/coad_visium.json

# Xenium five-fold spatial cross-validation
python -m brest.train.xenium \
  --feature-cache /path/to/sample_all_cls_patch_mean.npz \
  --metadata /path/to/sample_cell_metadata.parquet \
  --gene-list /path/to/genes.txt \
  --epochs 60 \
  --device cuda:0

# Bulk backbone
python -m brest.train.bulk \
  --graphs /path/to/bulk_subgraphs \
  --gene-list /path/to/genes.txt \
  --tpm /path/to/panel_tpm.csv \
  --out runs/bulk_backbone.pt
```

For the two final non-transfer ablations, add `--no-gat` or
`--head-type linear` to the Visium or Xenium command.

## Bulk-prior completion

For observed genes \(O\), held-out genes \(H\), and a bulk covariance
\(\Sigma\), BREST uses

\[
\widetilde{Y}_H =
\widetilde{Y}_O
(\Sigma_{OO}+\lambda I)^{-1}\Sigma_{OH}.
\]

Measured panels use \(\lambda=10\). When the observed panel is predicted from
histology, BREST uses the prediction-calibrated
\(\lambda_{\mathrm{pred}}=1\), which avoids compounding the variance
attenuation of image predictions. Calibration uses observed training genes
only, adds no trainable parameters, and never reads held-out genes.

See [`docs/bulk_completion.md`](docs/bulk_completion.md) for the API and
evaluation protocol.

## Repository layout

```text
brest/                 installable Python package
  data/                graph and bulk dataset construction
  models/              spatial encoder, gene-query head, transfer utilities
  train/               bulk, Visium, Xenium, and pseudo-spot entry points
  downstream/          completion, panel, and pathway utilities
demo/                  executed synthetic end-to-end notebook
docs/                  data formats and method-specific documentation
scripts/preprocess/    H0-mini feature and graph cache builders
tests/                 unit and regression tests
```

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q brest demo scripts/preprocess
python demo/validate_demo.py
```

## External assets

H0-mini and CellViT weights, HEST data, TCGA data, and 10x Genomics assets are
not included. Review [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before
downloading or redistributing external assets.

## License

BREST source code is released under the MIT License. The synthetic demo data
are released under CC0-1.0. Third-party data and weights retain their upstream
licenses.
