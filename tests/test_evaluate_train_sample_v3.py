import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from evaluate_train_sample_v3 import graph_record_id, stratified_sample_indices


class TrainingSampleDiagnosticTests(unittest.TestCase):
    def test_stratified_sampling_includes_rare_high_scores(self):
        targets = np.concatenate([
            np.full(100, 5.0),
            np.full(100, 7.0),
            np.full(4, 11.0),
            np.full(2, 12.5),
        ])

        indices = stratified_sample_indices(targets, 12, seed=42)

        self.assertEqual(len(indices), 12)
        self.assertEqual(len(np.unique(indices)), 12)
        self.assertTrue(np.any(targets[indices] == 11.0))
        self.assertTrue(np.any(targets[indices] == 12.5))
        np.testing.assert_array_equal(
            indices,
            stratified_sample_indices(targets, 12, seed=42),
        )

    def test_graph_record_id_is_repeatable_and_graph_sensitive(self):
        graph = Data(
            x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        same_graph = Data(x=graph.x.clone(), edge_index=graph.edge_index.clone())
        other_graph = Data(
            x=graph.x.clone(),
            edge_index=torch.tensor([[0], [1]]),
        )

        self.assertEqual(graph_record_id(graph), graph_record_id(same_graph))
        self.assertNotEqual(graph_record_id(graph), graph_record_id(other_graph))


if __name__ == "__main__":
    unittest.main()
