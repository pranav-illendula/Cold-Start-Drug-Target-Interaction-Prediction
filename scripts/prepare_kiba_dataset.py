"""Download and prepare the KIBA kinase bioactivity benchmark dataset.

KIBA dataset summary:
- 2,111 unique small-molecule kinase inhibitors
- 229 unique kinase protein targets
- 118,254 measured bioactivity interaction pairs
- Continuous KIBA score: [0.0, 17.2], mean ~11.72
- Higher score indicates higher binding affinity (cutoff ~12.1)
"""

import os
import json
import pickle
import urllib.request
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "kiba")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")

KIBA_URLS = {
    "Y": "https://raw.githubusercontent.com/thinng/GraphDTA/master/data/kiba/Y",
    "ligands": "https://raw.githubusercontent.com/thinng/GraphDTA/master/data/kiba/ligands_can.txt",
    "proteins": "https://raw.githubusercontent.com/thinng/GraphDTA/master/data/kiba/proteins.txt",
}


def download_file(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"File already exists: {target_path}")
        return
    print(f"Downloading {url} -> {target_path}...")
    urllib.request.urlretrieve(url, target_path)
    print(f"Downloaded ({os.path.getsize(target_path)} bytes).")


def get_scaffold(smiles, fallback):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return str(fallback)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return str(fallback)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return str(fallback)


def get_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            return Chem.MolToInchiKey(mol)
    except Exception:
        pass
    return ""


def build_kiba_dataframe(raw_dir):
    ligands_file = os.path.join(raw_dir, "ligands_can.txt")
    proteins_file = os.path.join(raw_dir, "proteins.txt")
    y_file = os.path.join(raw_dir, "Y")

    with open(ligands_file, "r") as f:
        ligands_dict = json.load(f)
    with open(proteins_file, "r") as f:
        proteins_dict = json.load(f)
    with open(y_file, "rb") as f:
        y_matrix = pickle.load(f, encoding="latin1")

    y_arr = np.array(y_matrix, dtype=np.float64)  # Shape: (2111, 229)
    drug_keys = list(ligands_dict.keys())         # 2111 drugs
    protein_keys = list(proteins_dict.keys())     # 229 proteins

    assert y_arr.shape == (len(drug_keys), len(protein_keys)), (
        f"Shape mismatch: {y_arr.shape} vs ({len(drug_keys)}, {len(protein_keys)})"
    )

    rows = []
    # Vectorized / indexed retrieval of non-NaN entries
    d_indices, p_indices = np.where(~np.isnan(y_arr))
    print(f"Extracting {len(d_indices)} valid interaction pairs...")

    # Precompute drug metadata for unique 2,111 compounds
    drug_meta = {}
    for d_idx, d_key in enumerate(drug_keys):
        smiles = ligands_dict[d_key]
        inchi = get_inchikey(smiles)
        scaff = get_scaffold(smiles, d_key)
        drug_meta[d_idx] = {
            "Drug_ID": str(d_key),
            "Canonical_SMILES": smiles,
            "Ligand_InChI_Key": inchi,
            "scaffold": scaff,
        }

    for d_idx, p_idx in zip(d_indices, p_indices):
        d_info = drug_meta[d_idx]
        p_key = protein_keys[p_idx]
        val = float(y_arr[d_idx, p_idx])
        rows.append({
            "Drug_ID": d_info["Drug_ID"],
            "Canonical_SMILES": d_info["Canonical_SMILES"],
            "Ligand_InChI_Key": d_info["Ligand_InChI_Key"],
            "scaffold": d_info["scaffold"],
            "Target_Name": str(p_key),
            "Target_Seq": proteins_dict[p_key],
            "KIBA_Score": val,
            "drug_idx": int(d_idx),
            "target_idx": int(p_idx),
        })

    df = pd.DataFrame(rows)
    return df, proteins_dict, ligands_dict


def create_cold_start_splits(df, seed=42, drug_test_fraction=0.20, target_test_fraction=0.20):
    rng = np.random.RandomState(seed)

    unique_drugs = np.array(sorted(df["Drug_ID"].unique()))
    unique_targets = np.array(sorted(df["Target_Name"].unique()))

    rng.shuffle(unique_drugs)
    rng.shuffle(unique_targets)

    n_test_drugs = int(len(unique_drugs) * drug_test_fraction)
    n_test_targets = int(len(unique_targets) * target_test_fraction)

    test_drugs = set(unique_drugs[:n_test_drugs])
    train_drugs = set(unique_drugs[n_test_drugs:])

    test_targets = set(unique_targets[:n_test_targets])
    train_targets = set(unique_targets[n_test_targets:])

    is_test_drug = df["Drug_ID"].isin(test_drugs)
    is_test_target = df["Target_Name"].isin(test_targets)

    splits = {
        "train": df[~is_test_drug & ~is_test_target].copy(),
        "cold_drug": df[is_test_drug & ~is_test_target].copy(),
        "cold_target": df[~is_test_drug & is_test_target].copy(),
        "cold_pair": df[is_test_drug & is_test_target].copy(),
    }

    # Warm-start 5-fold CV test fold
    warm_indices = rng.permutation(len(df))
    warm_test_size = int(len(df) * 0.20)
    splits["warm_test"] = df.iloc[warm_indices[:warm_test_size]].copy()
    splits["warm_train"] = df.iloc[warm_indices[warm_test_size:]].copy()

    # Leakage checks
    train_drug_set = set(splits["train"]["Drug_ID"])
    train_target_set = set(splits["train"]["Target_Name"])

    leakage = {
        "cold_drug drug leakage": splits["cold_drug"]["Drug_ID"].isin(train_drug_set).sum(),
        "cold_target target leakage": splits["cold_target"]["Target_Name"].isin(train_target_set).sum(),
        "cold_pair drug leakage": splits["cold_pair"]["Drug_ID"].isin(train_drug_set).sum(),
        "cold_pair target leakage": splits["cold_pair"]["Target_Name"].isin(train_target_set).sum(),
    }
    print("KIBA leakage verification:", leakage)
    if any(leakage.values()):
        raise ValueError(f"Cold-start leakage detected in KIBA: {leakage}")

    return splits


def main():
    raw_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    for name, url in KIBA_URLS.items():
        fname = os.path.basename(url)
        download_file(url, os.path.join(raw_dir, fname))

    print("Building KIBA dataset DataFrame...")
    df, proteins_dict, ligands_dict = build_kiba_dataframe(raw_dir)
    full_csv = os.path.join(DATA_DIR, "kiba_full.csv")
    df.to_csv(full_csv, index=False)
    print(f"Saved {len(df)} total pairs to {full_csv}")
    print(f"KIBA score range: [{df['KIBA_Score'].min():.2f}, {df['KIBA_Score'].max():.2f}], Mean: {df['KIBA_Score'].mean():.2f}, Std: {df['KIBA_Score'].std():.2f}")

    # Target lookup
    target_names = list(proteins_dict.keys())
    target_to_idx = {name: idx for idx, name in enumerate(target_names)}
    target_sequences = [proteins_dict[name] for name in target_names]

    with open(os.path.join(DATA_DIR, "target_to_idx.pkl"), "wb") as f:
        pickle.dump(target_to_idx, f)
    with open(os.path.join(DATA_DIR, "target_sequences.pkl"), "wb") as f:
        pickle.dump(target_sequences, f)
    print(f"Saved {len(target_to_idx)} target indices and sequences to {DATA_DIR}.")

    # Generate splits
    splits = create_cold_start_splits(df, seed=42)
    for name, split_df in splits.items():
        out_csv = os.path.join(SPLIT_DIR, f"split_{name}.csv")
        split_df.to_csv(out_csv, index=False)
        print(f"Split {name}: {len(split_df)} samples, {split_df['Drug_ID'].nunique()} drugs, {split_df['Target_Name'].nunique()} targets -> {out_csv}")


if __name__ == "__main__":
    main()

