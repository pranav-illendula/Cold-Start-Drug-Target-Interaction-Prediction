import sys
import unittest
from pathlib import Path

OLDER_DIR = Path(__file__).resolve().parents[1] / "scripts" / "Older versions"
if str(OLDER_DIR) not in sys.path:
    sys.path.insert(0, str(OLDER_DIR))

from extend_target_index_v4_2 import extend_target_index


class ExtendTargetIndexTests(unittest.TestCase):
    def test_appends_only_missing_targets_in_stable_order(self):
        fetched = []

        def fetch_sequence(uniprot_id):
            fetched.append(uniprot_id)
            return "MSEQ"

        mapping, sequences, added = extend_target_index(
            {"P00533": 0},
            ["MSEQ"],
            {"P00533", "Q93009", "P18031"},
            fetch_sequence,
        )

        self.assertEqual(added, ["P18031", "Q93009"])
        self.assertEqual(fetched, added)
        self.assertEqual(mapping, {"P00533": 0, "P18031": 1, "Q93009": 2})
        self.assertEqual(sequences, ["MSEQ", "MSEQ", "MSEQ"])

    def test_requires_mapping_and_sequence_alignment(self):
        with self.assertRaisesRegex(ValueError, "different lengths"):
            extend_target_index({"P00533": 0}, [], set(), lambda _: "MSEQ")


if __name__ == "__main__":
    unittest.main()
