import unittest

import numpy as np

from loss_weighting_v4_2 import (
    MAX_PIC50_BIN_WEIGHT,
    PIC50_BIN_EDGES,
    build_pic50_bin_weights,
    pic50_bin_indices,
)


class Pic50BinWeightTests(unittest.TestCase):
    def test_rare_high_pic50_bins_receive_larger_capped_weights(self):
        labels = np.array([5.0] * 100 + [9.5] * 4 + [11.0])

        weights, counts = build_pic50_bin_weights(labels)

        self.assertEqual(counts.tolist(), [0, 100, 0, 0, 4, 1, 0])
        self.assertGreater(weights[4], weights[1])
        self.assertGreater(weights[5], weights[1])
        populated_weights = weights[counts > 0]
        self.assertLessEqual(
            populated_weights.max() / populated_weights.min(),
            MAX_PIC50_BIN_WEIGHT,
        )
        self.assertAlmostEqual(
            float(np.dot(weights, counts) / counts.sum()),
            1.0,
            places=6,
        )

    def test_bin_edges_match_evaluation_ranges_and_include_extremes(self):
        labels = [-2.0, *PIC50_BIN_EDGES, 20.0]

        self.assertEqual(
            pic50_bin_indices(labels).tolist(),
            [0, 1, 2, 3, 4, 5, 6, 6],
        )

    def test_empty_or_non_finite_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            build_pic50_bin_weights([])
        with self.assertRaisesRegex(ValueError, "finite"):
            build_pic50_bin_weights([5.0, np.nan])


if __name__ == "__main__":
    unittest.main()
