#!/usr/bin/env python3
"""Validate the bundled demo artifact and the executed notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "demo" / "demo_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    artifact = DATA_DIR / "brest_toy_coad.npz"
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    if sha256(artifact) != manifest["sha256"]:
        raise RuntimeError("Demo data checksum does not match manifest.json")

    with np.load(artifact, allow_pickle=False) as data:
        required = {
            "cell_features", "cell_coords", "bag_ptr", "expression", "spot_coords",
            "slide_index", "slide_names", "slide_splits", "genes", "bulk_expression",
            "heldout_indices",
        }
        if set(data.files) != required:
            raise RuntimeError(f"Unexpected demo schema: {sorted(data.files)}")
        if data["cell_features"].shape[1] != 1536:
            raise RuntimeError("Demo features must match the H0-mini 1536-d interface")
        if data["expression"].shape[1] != len(data["genes"]):
            raise RuntimeError("Expression and gene dimensions do not match")
        if data["bag_ptr"][-1] != len(data["cell_features"]):
            raise RuntimeError("bag_ptr does not cover every cell")
        if set(data["slide_splits"].tolist()) != {"train", "validation", "test"}:
            raise RuntimeError("Expected train/validation/test slide splits")

    notebook = nbformat.read(ROOT / "demo" / "BREST_demo.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(not cell.get("execution_count") for cell in code_cells):
        raise RuntimeError("Notebook contains unexecuted code cells")
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    if errors:
        raise RuntimeError(f"Notebook contains {len(errors)} execution errors")
    if not any("Demo completed successfully." in str(output) for cell in code_cells
               for output in cell.get("outputs", [])):
        raise RuntimeError("Notebook completion marker is missing")
    print(
        f"Validated {artifact.name}, {len(code_cells)} executed code cells, "
        f"{len(notebook.cells)} total cells"
    )


if __name__ == "__main__":
    main()
