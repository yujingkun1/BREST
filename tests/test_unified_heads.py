import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from brest.downstream import blup_map, estimate_prediction_ridge, impute_blup
from brest.models import (
    GeneQueryHead,
    LinearGeneHead,
    UnifiedExpressionModel,
    load_backbone_by_symbol,
)
from brest.utils import completion_pearsons, mean_gene_pearson, overall_pearson


class UnifiedHeadTest(unittest.TestCase):
    def test_head_shapes_match(self):
        units = torch.randn(5, 16)
        for head_cls in (GeneQueryHead, LinearGeneHead):
            with self.subTest(head=head_cls.__name__):
                self.assertEqual(head_cls(16, 7)(units).shape, (5, 7))

    def test_linear_model_has_no_gene_query_parameters(self):
        model = UnifiedExpressionModel(in_dim=8, num_genes=7, hidden_dim=16,
                                       n_layers=1, heads=4, head_type="linear")
        self.assertFalse(any("gene_queries" in name for name, _ in model.named_parameters()))

    def test_invalid_head_type_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "head_type"):
            UnifiedExpressionModel(num_genes=7, head_type="unknown")

    def test_no_gat_ablation_preserves_shape(self):
        data = Data(
            x=torch.randn(6, 8), pos=torch.randn(6, 2),
            edge_index=torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5],
                                     [1, 0, 2, 1, 4, 3, 5, 4]]),
            batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        )
        model = UnifiedExpressionModel(
            in_dim=8, num_genes=7, hidden_dim=16, n_layers=1, heads=4,
            use_local=False,
        )
        self.assertEqual(model(data)["prediction"].shape, (6, 7))

    def test_gene_query_checkpoint_loads_into_linear_ablation(self):
        source = UnifiedExpressionModel(
            in_dim=8,
            num_genes=3,
            hidden_dim=16,
            n_layers=1,
            heads=4,
        )
        target = UnifiedExpressionModel(
            in_dim=8,
            num_genes=3,
            hidden_dim=16,
            n_layers=1,
            heads=4,
            head_type="linear",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "backbone.pt"
            torch.save(
                {"model": source.state_dict(), "genes": ["A", "B", "C"]},
                checkpoint,
            )
            report = load_backbone_by_symbol(
                target,
                str(checkpoint),
                ["A", "B", "C"],
            )
        self.assertEqual(report["gene_queries_copied"], 0)
        self.assertIn("head.gene_queries", report["skipped"])


class MetricTest(unittest.TestCase):
    def test_overall_and_gene_pearson_are_distinct_metrics(self):
        target = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]], dtype=np.float32)
        prediction = np.array([[0.0, 30.0], [1.0, 20.0], [2.0, 10.0]], dtype=np.float32)
        self.assertNotAlmostEqual(overall_pearson(prediction, target),
                                  mean_gene_pearson(prediction, target))

    def test_completion_metrics_restore_log_expression(self):
        target = np.array([[1.0, 5.0], [2.0, 7.0], [3.0, 9.0]], dtype=np.float32)
        mean = np.array([2.0, 7.0], dtype=np.float32)
        std = np.array([1.0, 2.0], dtype=np.float32)
        prediction_z = (target - mean) / std
        overall, gene = completion_pearsons(prediction_z, target, mean, std)
        self.assertAlmostEqual(overall, 1.0, places=6)
        self.assertAlmostEqual(gene, 1.0, places=6)


class BulkCompletionTest(unittest.TestCase):
    def test_blup_map_matches_direct_imputation(self):
        covariance = np.array(
            [[1.0, 0.2, 0.4], [0.2, 1.0, -0.1], [0.4, -0.1, 1.0]],
            dtype=np.float32,
        )
        observed = np.array([0, 1])
        heldout = np.array([2])
        expression = np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
        mapping = blup_map(covariance, observed, heldout, lam=1.0)
        np.testing.assert_allclose(
            impute_blup(expression, covariance, observed, heldout, lam=1.0),
            expression @ mapping,
        )

    def test_prediction_ridge_uses_observed_genes_only(self):
        target = np.zeros((2, 3), dtype=np.float32)
        prediction = np.array([[1.0, 50.0, -1.0], [1.0, -50.0, -1.0]])
        ridge = estimate_prediction_ridge(
            prediction, target, observed=np.array([0, 2])
        )
        self.assertAlmostEqual(ridge, 1.0)


if __name__ == "__main__":
    unittest.main()
