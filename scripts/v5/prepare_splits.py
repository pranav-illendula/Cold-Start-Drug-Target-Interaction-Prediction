"""Create leakage-checked cold-start CSV splits from the cleaned BindingDB data."""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEFAULT_INPUT = os.path.join(DATA_DIR, "BindingDB_final_clean.csv")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, "splits")


def get_scaffold(smiles, fallback):
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return str(fallback)
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
        if scaffold is None:
            return str(fallback)
        return Chem.MolToSmiles(scaffold)
    except (RuntimeError, ValueError):
        return str(fallback)


def scaffold_balanced_split(frame, scaffold_column, test_fraction, seed):
    counts = frame.groupby(scaffold_column).size().sort_values(ascending=False)
    scaffold_names = list(counts.index)
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_names)
    scaffold_names.sort(key=lambda scaffold: -counts[scaffold])

    target_size = int(len(frame) * test_fraction)
    test_groups = set()
    train_groups = set()
    test_count = 0
    for scaffold in scaffold_names:
        if test_count < target_size:
            test_groups.add(scaffold)
            test_count += int(counts[scaffold])
        else:
            train_groups.add(scaffold)
    if not test_groups or not train_groups:
        raise ValueError("Scaffold split produced an empty train or test group.")
    return train_groups, test_groups


def group_split(values, test_fraction, seed):
    groups = np.asarray(pd.Series(values).dropna().unique(), dtype=object)
    if len(groups) < 2:
        raise ValueError("At least two target groups are required for the split.")
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    test_count = min(len(groups) - 1, max(1, int(len(groups) * test_fraction)))
    test_groups = set(groups[:test_count])
    train_groups = set(groups[test_count:])
    return train_groups, test_groups


def create_splits(frame, seed=42, drug_test_fraction=0.20, target_test_fraction=0.15):
    required = {"Canonical_SMILES", "Ligand_InChI_Key", "Target_Name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}.")
    if frame.empty:
        raise ValueError("Input BindingDB table is empty.")
    if frame["Target_Name"].isna().any():
        raise ValueError("Input data contains rows without a target name.")

    frame = frame.copy()
    drug_keys = frame["Ligand_InChI_Key"].fillna("").astype(str).str.strip()
    missing_keys = drug_keys == ""
    drug_keys.loc[missing_keys] = (
        "SMILES:" + frame.loc[missing_keys, "Canonical_SMILES"].astype(str)
    )
    frame["_drug_key"] = drug_keys
    frame["scaffold"] = [
        get_scaffold(smiles, key if key else f"row-{index}")
        for index, (smiles, key) in enumerate(
            zip(frame["Canonical_SMILES"], frame["_drug_key"])
        )
    ]
    primary_scaffold = (
        frame.groupby("_drug_key")["scaffold"]
        .agg(lambda values: values.value_counts().index[0])
    )
    frame["scaffold"] = frame["_drug_key"].map(primary_scaffold)

    _, test_targets = group_split(
        frame["Target_Name"],
        target_test_fraction,
        seed,
    )
    _, test_scaffolds = scaffold_balanced_split(
        frame,
        "scaffold",
        drug_test_fraction,
        seed,
    )
    is_test_drug = frame["scaffold"].isin(test_scaffolds)
    is_test_target = frame["Target_Name"].isin(test_targets)
    splits = {
        "train": frame[~is_test_drug & ~is_test_target].copy(),
        "cold_drug": frame[is_test_drug & ~is_test_target].copy(),
        "cold_target": frame[~is_test_drug & is_test_target].copy(),
        "cold_pair": frame[is_test_drug & is_test_target].copy(),
    }

    train_drugs = set(splits["train"]["_drug_key"])
    train_target_names = set(splits["train"]["Target_Name"].dropna())
    leakage = {
        "cold_drug drug": splits["cold_drug"]["_drug_key"].isin(train_drugs).sum(),
        "cold_target target": splits["cold_target"]["Target_Name"].isin(train_target_names).sum(),
        "cold_pair drug": splits["cold_pair"]["_drug_key"].isin(train_drugs).sum(),
        "cold_pair target": splits["cold_pair"]["Target_Name"].isin(train_target_names).sum(),
    }
    if any(leakage.values()):
        raise ValueError(f"Cold-start leakage detected: {leakage}")
    if any(split.empty for split in splits.values()):
        raise ValueError(
            "At least one split is empty; check source data and split fractions."
        )
    return {
        name: split.drop(columns=["_drug_key", "scaffold"])
        for name, split in splits.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drug-test-fraction", type=float, default=0.20)
    parser.add_argument("--target-test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    if not 0 < args.drug_test_fraction < 1:
        raise ValueError("--drug-test-fraction must be between 0 and 1.")
    if not 0 < args.target_test_fraction < 1:
        raise ValueError("--target-test-fraction must be between 0 and 1.")
    frame = pd.read_csv(args.input)
    splits = create_splits(
        frame,
        seed=args.seed,
        drug_test_fraction=args.drug_test_fraction,
        target_test_fraction=args.target_test_fraction,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    for name, split in splits.items():
        output_path = os.path.join(args.output_dir, f"split_{name}.csv")
        split.to_csv(output_path, index=False)
        print(f"{name}: {len(split):,} rows -> {output_path}")
    print("Leakage verification passed for all cold-start splits.")


if __name__ == "__main__":
    main()
