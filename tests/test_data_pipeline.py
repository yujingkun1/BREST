import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from brest.data import (
    BulkGraphDataset,
    build_delaunay_edges,
    collate_bulk,
    split_slides_by_patient,
)


class GraphConstructionTest(unittest.TestCase):
    def test_collinear_points_fall_back_to_chain(self):
        coords = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        edges = build_delaunay_edges(coords)
        self.assertEqual(edges.shape, (2, 2))
        self.assertEqual({tuple(edge) for edge in edges}, {(1, 2), (0, 2)})


class BulkDatasetTest(unittest.TestCase):
    def test_patient_split_prevents_slide_leakage(self):
        mapping = {
            "slide-a": "patient-1",
            "slide-b": "patient-1",
            "slide-c": "patient-2",
            "slide-d": "patient-3",
        }
        split = split_slides_by_patient(mapping, test_fraction=0.34, seed=42)
        train_patients = {mapping[slide] for slide in split["train"]}
        test_patients = {mapping[slide] for slide in split["test"]}
        self.assertTrue(train_patients)
        self.assertTrue(test_patients)
        self.assertTrue(train_patients.isdisjoint(test_patients))

    def test_multiple_slides_can_share_one_patient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = Data(
                x=torch.ones(3, 4),
                pos=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                edge_index=torch.tensor(
                    [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]
                ),
            )
            graphs = {
                "slide-a": {0: graph},
                "slide-b": {0: graph.clone()},
            }
            mapping = {
                "slide-a": "TCGA-AA-0001",
                "slide-b": "TCGA-AA-0001",
            }
            with (root / "bulk_train_intra_patch_graphs.pkl").open("wb") as handle:
                pickle.dump(graphs, handle)
            with (
                root / "bulk_train_slide_to_patient_mapping.pkl"
            ).open("wb") as handle:
                pickle.dump(mapping, handle)

            tpm = pd.DataFrame(
                {"TCGA-AA-0001-01A": [2.0, 3.0]},
                index=["GENE_A", "GENE_B"],
            )
            tpm_path = root / "panel_tpm.csv"
            tpm.to_csv(tpm_path)

            dataset = BulkGraphDataset(
                root,
                tpm_path,
                ["GENE_A", "GENE_B"],
                split="train",
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["expression"].tolist(), [2.0, 3.0])
            self.assertEqual(dataset[1]["expression"].tolist(), [2.0, 3.0])

            batch = collate_bulk([dataset[0], dataset[1]])
            self.assertEqual(batch["expressions"].shape, (2, 2))
            self.assertEqual(len(batch["spot_graphs_list"]), 2)


if __name__ == "__main__":
    unittest.main()
