"""Build PyG-compatible molecular graph representations for the KIBA dataset."""

import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, BondType

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "kiba")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
GRAPH_DIR = os.path.join(DATA_DIR, "graphs")

ATOMIC_NUMBERS = (6, 7, 8, 16, 15, 9, 17, 35)
BOND_TYPES = (
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
    BondType.DATIVE,
)
BOND_STEREO_TYPES = (
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
)


def atom_features(atom):
    atomic_number = atom.GetAtomicNum()
    atom_type = (
        ATOMIC_NUMBERS.index(atomic_number)
        if atomic_number in ATOMIC_NUMBERS
        else 8
    )
    degree = min(atom.GetTotalDegree(), 5)
    charge = atom.GetFormalCharge()
    charge_type = 0 if charge < 0 else 2 if charge > 0 else 1
    return (
        *(float(index == atom_type) for index in range(9)),
        *(float(index == degree) for index in range(6)),
        *(float(index == charge_type) for index in range(3)),
    )


def bond_features(bond, reverse=False):
    bond_type = bond.GetBondType()
    stereo_type = bond.GetStereo()
    if bond_type not in BOND_TYPES:
        bond_type = BondType.SINGLE
    if stereo_type not in BOND_STEREO_TYPES:
        stereo_type = BondStereo.STEREONONE

    return (
        *(float(bond_type == known_type) for known_type in BOND_TYPES),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *(float(stereo_type == known_type) for known_type in BOND_STEREO_TYPES),
        float(bond_type == BondType.DATIVE and not reverse),
    )


def molecule_bond_features(molecule):
    edge_index = [[], []]
    edge_attr = []
    for bond in molecule.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        forward_features = bond_features(bond)
        reverse_features = bond_features(bond, reverse=True)
        edge_index[0].extend((source, target))
        edge_index[1].extend((target, source))
        edge_attr.extend((forward_features, reverse_features))
    return edge_index, edge_attr


def precompute_drug_graphs(smiles_series):
    unique_smiles = smiles_series.unique()
    print(f"Precomputing molecular graphs for {len(unique_smiles)} unique drugs in KIBA...")
    drug_cache = {}
    for sm in unique_smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_feats:
            continue
        edge_index, edge_attr = molecule_bond_features(mol)
        drug_cache[sm] = {
            "x": np.array(atom_feats, dtype=np.float32),
            "edge_index": np.array(edge_index, dtype=np.int64),
            "edge_attr": np.array(edge_attr, dtype=np.float32),
        }
    print(f"Precomputed {len(drug_cache)} valid KIBA drug graphs.")
    return drug_cache


def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "target_to_idx.pkl"), "rb") as f:
        target_to_idx = pickle.load(f)

    full_df = pd.read_csv(os.path.join(DATA_DIR, "kiba_full.csv"))
    drug_cache = precompute_drug_graphs(full_df["Canonical_SMILES"])

    splits = ["train", "cold_drug", "cold_target", "cold_pair", "warm_test"]
    for split_name in splits:
        csv_path = os.path.join(SPLIT_DIR, f"split_{split_name}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        graphs = []
        smiles_list = df["Canonical_SMILES"].values
        target_list = df["Target_Name"].values
        score_list = df["KIBA_Score"].values

        for sm, t_name, sc in zip(smiles_list, target_list, score_list):
            if sm not in drug_cache or t_name not in target_to_idx:
                continue
            base = drug_cache[sm]
            graphs.append({
                "x": base["x"],
                "edge_index": base["edge_index"],
                "edge_attr": base["edge_attr"],
                "target_idx": target_to_idx[t_name],
                "y": float(sc),
            })
        out_path = os.path.join(GRAPH_DIR, f"kiba_{split_name}_graphs.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(graphs, f)
        print(f"Saved {len(graphs)} graphs -> {out_path}")


if __name__ == "__main__":
    main()

