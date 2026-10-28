import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from scripts.generate_esm_residue_embeddings_v1 import sequence_windows
from scripts.evaluate_enhancements_v1 import _validate_calibration_split
from scripts.model_enhancements_v1 import (
    AtomResidueCrossAttention,
    DTIEnhancementModel,
    TargetEmbeddingStore,
    WindowAttentionPool,
    build_pic50_bin_weights,
    conformal_scale_quantile,
    hybrid_ranking_regression_loss,
    target_group_conformal_quantile,
)


class EnhancementModelTests(unittest.TestCase):
    def test_calibration_targets_must_be_disjoint_from_training_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training_protocol.json").write_text(
                json.dumps(
                    {
                        "experiment": "section4_enhancements_v1",
                        "target_mapping_sha256": "mapping-hash",
                        "seeds": [42],
                        "train_target_indices": [0, 1],
                        "validation_target_indices": [2],
                    }
                ),
                encoding="utf-8",
            )
            calibration_graph = Data(target_idx=torch.tensor([3]))
            _validate_calibration_split(
                root,
                "mapping-hash",
                (42,),
                [calibration_graph],
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                _validate_calibration_split(
                    root,
                    "mapping-hash",
                    (42,),
                    [Data(target_idx=torch.tensor([1]))],
                )

    def test_sliding_windows_cover_long_sequence_with_expected_overlap(self):
        windows = sequence_windows("A" * 1600)

        self.assertEqual([start for start, _ in windows], [0, 512, 1024])
        self.assertEqual([len(sequence) for _, sequence in windows], [1022, 1022, 576])
        covered = set()
        for start, sequence in windows:
            covered.update(range(start, start + len(sequence)))
        self.assertEqual(covered, set(range(1600)))

    def test_window_attention_ignores_padding(self):
        module = WindowAttentionPool()
        windows = torch.randn(2, 3, 1280)
        windows[0, 2] = 1e6
        mask = torch.tensor([[True, True, False], [True, False, False]])

        pooled, weights = module(windows, mask)

        self.assertEqual(pooled.shape, (2, 1280))
        self.assertEqual(weights.shape, (2, 3))
        self.assertEqual(float(weights[0, 2].detach()), 0.0)
        self.assertEqual(float(weights[1, 1:].sum().detach()), 0.0)

    def test_cross_attention_masks_invalid_residues(self):
        module = AtomResidueCrossAttention(hidden_dim=16, heads=4)
        atoms = torch.randn(1, 3, 16)
        residues = torch.randn(1, 4, 640)
        mask = torch.tensor([[True, True, False, False]])

        conditioned, weights = module(
            atoms,
            residues,
            mask,
            return_attention=True,
        )

        self.assertEqual(conditioned.shape, (1, 3, 16))
        self.assertEqual(weights.shape, (1, 3, 4))
        self.assertTrue(torch.equal(weights[:, :, 2:], torch.zeros_like(weights[:, :, 2:])))

    def test_enhancement_model_forward_backward_and_attention_shapes(self):
        graphs = [
            Data(
                x=torch.randn(3, 18),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                edge_attr=torch.randn(2, 14),
                target_idx=torch.tensor([0]),
                y=torch.tensor([6.0]),
            ),
            Data(
                x=torch.randn(2, 18),
                edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.randn(1, 14),
                target_idx=torch.tensor([1]),
                y=torch.tensor([7.0]),
            ),
        ]
        batch = Batch.from_data_list(graphs)
        model = DTIEnhancementModel(hidden_dim=16)
        windows = torch.randn(2, 2, 1280)
        window_mask = torch.tensor([[True, True], [True, False]])
        residues = torch.randn(2, 4, 640)
        residue_mask = torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        )

        prediction, attention = model(
            batch,
            windows,
            window_mask,
            residues,
            residue_mask,
            return_attention=True,
        )
        prediction.sum().backward()

        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertEqual(attention["window_weights"].shape, (2, 2))
        self.assertEqual(len(attention["atom_residue_weights"]), 2)
        self.assertEqual(attention["atom_residue_weights"][0].shape, (3, 3))

    def test_hybrid_loss_is_finite_and_differentiable(self):
        labels = np.asarray([5.0, 5.5, 9.5, 10.0])
        weights, _ = build_pic50_bin_weights(labels)
        predictions = torch.tensor([5.1, 5.3, 9.4, 9.8], requires_grad=True)
        targets = torch.tensor(labels, dtype=torch.float32)

        loss, components = hybrid_ranking_regression_loss(
            predictions,
            targets,
            weights,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(components["ranking"]), 0.0)
        self.assertTrue(torch.isfinite(predictions.grad).all())

    def test_conformal_quantiles_support_row_and_target_group_calibration(self):
        targets = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        means = np.zeros(5)
        scales = np.ones(5)
        groups = np.asarray([10, 10, 20, 30, 30])

        row_quantile = conformal_scale_quantile(
            targets,
            means,
            scales,
            coverage=0.6,
        )
        group_quantile = target_group_conformal_quantile(
            targets,
            means,
            scales,
            groups,
            coverage=0.5,
        )

        self.assertEqual(row_quantile, 4.0)
        self.assertEqual(group_quantile, 3.0)

    def test_embedding_store_validates_manifest_and_safe_tensor_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "target_id": "P00001",
                "target_idx": 0,
                "sequence_length": 2,
                "sequence_sha256": "abc",
                "window_embeddings": torch.randn(1, 1280).half(),
                "residue_embeddings": torch.randn(2, 640).half(),
            }
            torch.save(payload, root / "0.pt")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_version": "recovered_baseline_v1",
                        "model_name": "esm2_t30_150M_UR50D",
                        "targets": [
                            {
                                "target_id": "P00001",
                                "target_idx": 0,
                                "sequence_length": 2,
                                "window_count": 1,
                                "sequence_sha256": "abc",
                                "file": "0.pt",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            store = TargetEmbeddingStore(root)

            windows, residues = store.get(0)

            self.assertEqual(windows.shape, (1, 1280))
            self.assertEqual(residues.shape, (2, 640))
            self.assertEqual(windows.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
