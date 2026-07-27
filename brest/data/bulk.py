"""Dataset utilities for bulk WSI subgraphs paired with TPM expression."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required bulk cache is missing: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _match_tpm_column(patient_id: str, columns: list[str]) -> str:
    if patient_id in columns:
        return patient_id
    matches = [column for column in columns if column.startswith(patient_id)]
    if not matches:
        raise KeyError(f"No TPM column matches patient {patient_id!r}")
    return sorted(matches)[0]


def split_slides_by_patient(
    slide_to_patient: dict[str, str],
    *,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    """Split slide IDs without allowing a patient to cross partitions."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    patients = sorted({str(patient) for patient in slide_to_patient.values()})
    if len(patients) < 2:
        raise ValueError("At least two patients are required for a train/test split")
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    n_test = min(
        len(patients) - 1,
        max(1, int(round(len(patients) * test_fraction))),
    )
    test_patients = set(patients[:n_test])
    split = {"train": [], "test": []}
    for slide_id in sorted(slide_to_patient):
        patient = str(slide_to_patient[slide_id])
        split["test" if patient in test_patients else "train"].append(slide_id)
    return split


class BulkGraphDataset(Dataset):
    """Load per-slide subgraphs produced by ``build_bulk_subgraphs.py``."""

    def __init__(
        self,
        graph_dir: str | Path,
        tpm_csv: str | Path,
        genes: list[str],
        *,
        split: str,
    ) -> None:
        import pandas as pd

        graph_dir = Path(graph_dir)
        prefix = graph_dir / f"bulk_{split}"
        self.graphs = _load_pickle(Path(f"{prefix}_intra_patch_graphs.pkl"))
        self.slide_to_patient = _load_pickle(
            Path(f"{prefix}_slide_to_patient_mapping.pkl")
        )
        self.slide_ids = sorted(self.graphs)
        tpm = pd.read_csv(tpm_csv, index_col=0)
        if tpm.shape[0] != len(genes):
            raise ValueError(
                f"TPM rows ({tpm.shape[0]}) must match gene list ({len(genes)}); "
                "provide a panel-filtered TPM matrix in gene-list order"
            )
        self.expression: dict[str, np.ndarray] = {}
        columns = [str(column) for column in tpm.columns]
        for slide_id in self.slide_ids:
            patient_id = str(self.slide_to_patient[slide_id])
            column = _match_tpm_column(patient_id, columns)
            self.expression[slide_id] = tpm[column].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, index: int) -> dict:
        slide_id = self.slide_ids[index]
        patient_id = str(self.slide_to_patient[slide_id])
        slide_graphs = self.graphs[slide_id]
        return {
            "slide_id": slide_id,
            "patient_id": patient_id,
            "spot_graphs": list(slide_graphs.values()),
            "expression": torch.from_numpy(self.expression[slide_id]),
        }


def collate_bulk(batch: list[dict]) -> dict:
    """Collate slides while preserving each slide's variable graph list."""
    return {
        "slide_ids": [item["slide_id"] for item in batch],
        "patient_ids": [item["patient_id"] for item in batch],
        "spot_graphs_list": [item["spot_graphs"] for item in batch],
        "expressions": torch.stack([item["expression"] for item in batch]),
    }
