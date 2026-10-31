import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

OLDER_DIR = Path(__file__).resolve().parents[1] / "scripts" / "Older versions"
if str(OLDER_DIR) not in sys.path:
    sys.path.insert(0, str(OLDER_DIR))

import numpy as np

import evaluate_v4_2


class EvaluateV4ReportTests(unittest.TestCase):
    def test_score_by_pic50_bin_reports_error_correlation_and_accuracy(self):
        targets = np.array([3.1, 3.8, 4.6, 5.2])
        predictions = np.array([3.2, 4.0, 4.8, 5.0])

        bins = evaluate_v4_2.score_by_pic50_bin(targets, predictions)

        self.assertEqual([row["range"] for row in bins], ["3.0-4.5", "4.5-6.0"])
        self.assertEqual([row["n"] for row in bins], [2, 2])
        self.assertAlmostEqual(bins[0]["mae"], 0.15)
        self.assertAlmostEqual(bins[0]["mse"], 0.025)
        self.assertTrue(np.isfinite(bins[0]["pearson_r"]))
        self.assertEqual(bins[0]["accuracy_0_5"], 100.0)
        self.assertEqual(bins[0]["same_bin_accuracy"], 100.0)
        self.assertEqual(
            evaluate_v4_2.score(
                np.array([3.1, 3.8]),
                np.array([3.2, 4.0]),
            )["ci"],
            1.0,
        )

    def test_report_is_written_to_requested_path_with_bin_metrics(self):
        result = {
            "metrics": {
                **evaluate_v4_2.score(
                    np.array([3.1, 3.8]),
                    np.array([3.2, 4.0]),
                ),
                "evaluation_minutes": 0.25,
            },
            "bins": evaluate_v4_2.score_by_pic50_bin(
                np.array([3.1, 3.8]),
                np.array([3.2, 4.0]),
            ),
        }
        results = {
            "COLD-DRUG": result,
            "COLD-PAIR": result,
            "COLD-TARGET": result,
        }
        checkpoint = {
            "epoch": 2,
            "best_validation_mae": 0.75,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = str(Path(temp_dir) / "evaluation_report_v4.txt")
            with patch.object(evaluate_v4_2, "REPORT_PATH", report_path):
                evaluate_v4_2.write_report(checkpoint, "cpu", results)
            report = Path(report_path).read_text(encoding="utf-8")

        self.assertIn("PER-pIC50-BIN RESULTS", report)
        self.assertIn("Pearson r", report)
        self.assertIn("Concordance index", report)
        self.assertIn("Acc±1.0", report)
        self.assertIn("MSE", report)
        self.assertIn("3.0-4.5", report)

    def test_locked_summary_is_saved_to_a_timestamped_fallback(self):
        result = {
            "metrics": evaluate_v4_2.score(
                np.array([3.1, 3.8]),
                np.array([3.2, 4.0]),
            ),
            "bins": [],
        }
        results = {"COLD-DRUG": result}
        real_open = open

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            locked_path = output_dir / "v4_summary.csv"

            def open_with_locked_summary(path, *args, **kwargs):
                if Path(path) == locked_path:
                    raise PermissionError("File is locked")
                return real_open(path, *args, **kwargs)

            with patch.object(evaluate_v4_2, "OUTPUT_DIR", str(output_dir)):
                with patch(
                    "builtins.open",
                    side_effect=open_with_locked_summary,
                ):
                    saved_path = Path(evaluate_v4_2.write_summary(results))

            self.assertNotEqual(saved_path, locked_path)
            self.assertTrue(saved_path.name.startswith("v4_summary_"))
            self.assertTrue(saved_path.is_file())


if __name__ == "__main__":
    unittest.main()
