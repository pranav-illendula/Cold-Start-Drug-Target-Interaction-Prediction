import sys
import unittest
from pathlib import Path

OLDER_DIR = Path(__file__).resolve().parents[1] / "scripts" / "Older versions"
if str(OLDER_DIR) not in sys.path:
    sys.path.insert(0, str(OLDER_DIR))

import numpy as np

from metrics_v4 import concordance_index


class ConcordanceIndexTests(unittest.TestCase):
    def test_perfect_and_reversed_rankings(self):
        targets = np.array([1.0, 2.0, 3.0])

        self.assertEqual(concordance_index(targets, targets), 1.0)
        self.assertEqual(concordance_index(targets, targets[::-1]), 0.0)

    def test_prediction_ties_count_as_half_and_target_ties_are_excluded(self):
        self.assertEqual(
            concordance_index([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
            0.5,
        )
        self.assertEqual(
            concordance_index([1.0, 1.0, 2.0], [1.0, 3.0, 2.0]),
            0.5,
        )

    def test_no_comparable_pairs_returns_nan(self):
        self.assertTrue(np.isnan(concordance_index([2.0, 2.0], [1.0, 2.0])))

    def test_rejects_mismatched_or_non_finite_inputs(self):
        with self.assertRaises(ValueError):
            concordance_index([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            concordance_index([1.0, np.nan], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
