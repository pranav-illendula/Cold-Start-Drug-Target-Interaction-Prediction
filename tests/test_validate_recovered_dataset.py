import contextlib
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.validate_recovered_dataset import main


class ValidateRecoveredDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _write_pickle(self, name, value):
        path = self.root / name
        with path.open("wb") as destination:
            pickle.dump(value, destination)
        return path

    def _write_csv(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def _args(self, *, csv_content=None, mapping=None, sequences=None, zipped=False):
        if csv_content is None:
            csv_content = (
                "Canonical_SMILES,Target_Name,UniProt_ID,pIC50\n"
                "CC,Target A,P00001,6.5\n"
                "CO,Target B,Q00002,7\n"
            )
        mapping_value = {"P00001": 0, "Q00002": 1} if mapping is None else mapping
        sequence_value = ["SEQONE", "SEQTWO"] if sequences is None else sequences

        if zipped:
            archive_path = self.root / "inputs.zip"
            csv_paths = []
            with ZipFile(archive_path, "w") as archive:
                for split in ("train", "cold-drug", "cold-target", "cold-pair"):
                    member = f"csv/{split}.csv"
                    archive.writestr(member, csv_content)
                    csv_paths.append(f"{archive_path}::{member}")
                archive.writestr(
                    "pickle/target_to_idx.pkl",
                    pickle.dumps(mapping_value),
                )
                archive.writestr(
                    "pickle/target_sequences.pkl",
                    pickle.dumps(sequence_value),
                )
            mapping_path = f"{archive_path}::pickle/target_to_idx.pkl"
            sequence_path = f"{archive_path}::pickle/target_sequences.pkl"
        else:
            csv_paths = []
            for split in ("train", "cold-drug", "cold-target", "cold-pair"):
                csv_paths.append(self._write_csv(f"{split}.csv", csv_content))
            mapping_path = self._write_pickle("target_to_idx.pkl", mapping_value)
            sequence_path = self._write_pickle("target_sequences.pkl", sequence_value)

        return [
            "--train-csv", str(csv_paths[0]),
            "--cold-drug-csv", str(csv_paths[1]),
            "--cold-target-csv", str(csv_paths[2]),
            "--cold-pair-csv", str(csv_paths[3]),
            "--target-to-idx", str(mapping_path),
            "--target-sequences", str(sequence_path),
        ]

    def test_valid_inputs_pass_and_report_split_counts(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(self._args())

        self.assertEqual(result, 0)
        self.assertIn("train: rows=2", output.getvalue())
        self.assertIn("unique_target_ids=2", output.getvalue())
        self.assertIn("target_ids_missing_from_mapping=none", output.getvalue())

    def test_zip_member_inputs_pass_without_extraction(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(self._args(zipped=True))

        self.assertEqual(result, 0)
        self.assertIn("train: rows=2", output.getvalue())
        self.assertIn("Validation passed.", output.getvalue())

    def test_invalid_csv_values_and_unmapped_targets_fail(self):
        csv_content = (
            "Canonical_SMILES,Target_Name,UniProt_ID,pIC50\n"
            "CC,Target A,P00001,not-a-number\n"
            ",Target B,P99999,inf\n"
        )
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = main(self._args(csv_content=csv_content))

        self.assertEqual(result, 1)
        self.assertIn("missing Canonical_SMILES=1", output.getvalue())
        self.assertIn("invalid_or_nonfinite_pIC50=2", output.getvalue())
        self.assertIn("P99999", output.getvalue())
        self.assertIn("Validation failed", errors.getvalue())

    def test_mapping_and_sequence_alignment_errors_fail(self):
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = main(
                self._args(
                    mapping={"P00001": 1, "Q00002": 2},
                    sequences=["SEQONE"],
                )
            )

        self.assertEqual(result, 1)
        self.assertIn("mapping_valid=False", output.getvalue())
        self.assertIn("unique and contiguous", errors.getvalue())
        self.assertIn("sequence count must match", errors.getvalue())

    def test_missing_zip_member_fails_clearly(self):
        arguments = self._args(zipped=True)
        arguments[1] = arguments[1].replace("csv/train.csv", "csv/missing.csv")
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = main(arguments)

        self.assertEqual(result, 1)
        self.assertIn("not found", errors.getvalue())

    def test_invalid_zip_archive_fails_clearly(self):
        bad_archive = self.root / "not-a-zip.zip"
        bad_archive.write_text("not a zip", encoding="utf-8")
        arguments = self._args()
        arguments[1] = f"{bad_archive}::csv/train.csv"
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = main(arguments)

        self.assertEqual(result, 1)
        self.assertIn("could not open ZIP archive", errors.getvalue())

    def test_malformed_csv_member_fails_clearly(self):
        archive_path = self.root / "malformed-csv.zip"
        valid_csv = (
            "Canonical_SMILES,Target_Name,UniProt_ID,pIC50\n"
            "CC,Target A,P00001,6.5\n"
        )
        with ZipFile(archive_path, "w") as archive:
            for split in ("train", "cold-drug", "cold-target", "cold-pair"):
                content = 'header\n"unterminated' if split == "train" else valid_csv
                archive.writestr(f"csv/{split}.csv", content)
            archive.writestr(
                "pickle/target_to_idx.pkl",
                pickle.dumps({"P00001": 0}),
            )
            archive.writestr(
                "pickle/target_sequences.pkl",
                pickle.dumps(["SEQONE"]),
            )
        arguments = [
            "--train-csv", f"{archive_path}::csv/train.csv",
            "--cold-drug-csv", f"{archive_path}::csv/cold-drug.csv",
            "--cold-target-csv", f"{archive_path}::csv/cold-target.csv",
            "--cold-pair-csv", f"{archive_path}::csv/cold-pair.csv",
            "--target-to-idx", f"{archive_path}::pickle/target_to_idx.pkl",
            "--target-sequences", f"{archive_path}::pickle/target_sequences.pkl",
        ]
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = main(arguments)

        self.assertEqual(result, 1)
        self.assertIn("could not read CSV", errors.getvalue())

    def test_malformed_pickle_member_fails_clearly(self):
        archive_path = self.root / "malformed-pickle.zip"
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("target_to_idx.pkl", b"not a pickle")
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = main(
                [
                    *self._args(),
                    "--target-to-idx",
                    f"{archive_path}::target_to_idx.pkl",
                ]
            )

        self.assertEqual(result, 1)
        self.assertIn("could not load target mapping", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
