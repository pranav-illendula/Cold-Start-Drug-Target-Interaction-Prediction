import unittest

from rdkit import Chem

from bond_features_v4 import (
    BOND_FEATURE_DIM,
    bond_features,
    molecule_bond_features,
)


class BondFeatureTests(unittest.TestCase):
    def test_double_bond_feature_has_expected_width_and_categories(self):
        molecule = Chem.MolFromSmiles("C=C")
        features = bond_features(molecule.GetBondWithIdx(0))

        self.assertEqual(len(features), BOND_FEATURE_DIM)
        self.assertEqual(features[:4], (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(sum(features[:4]), 1.0)
        self.assertEqual(sum(features[7:13]), 1.0)
        self.assertEqual(features[13], 0.0)

    def test_molecule_helper_emits_symmetric_directed_edges(self):
        molecule = Chem.MolFromSmiles("CC")
        edge_index, edge_attr = molecule_bond_features(molecule)

        self.assertEqual(edge_index, [[0, 1], [1, 0]])
        self.assertEqual(edge_attr[0], edge_attr[1])
        self.assertEqual(len(edge_attr[0]), BOND_FEATURE_DIM)

    def test_dative_bond_is_encoded_with_directed_donor_feature(self):
        molecule = Chem.MolFromSmiles("N->[Cu+2]")
        edge_index, edge_attr = molecule_bond_features(molecule)

        self.assertEqual(edge_index, [[0, 1], [1, 0]])
        self.assertEqual(edge_attr[0][4], 1.0)
        self.assertEqual(edge_attr[1][4], 1.0)
        self.assertEqual(edge_attr[0][13], 1.0)
        self.assertEqual(edge_attr[1][13], 0.0)
        self.assertEqual(len(edge_attr[0]), BOND_FEATURE_DIM)


if __name__ == "__main__":
    unittest.main()
