"""Download and process the Davis kinase benchmark dataset for Cold-Start DTI evaluation.

This script:
1. Downloads the canonical Davis dataset (ligands, targets, affinity matrix Y).
2. Converts raw Kd (nM) affinities to standard pKd = 9 - log10(Kd).
3. Extracts canonical SMILES, InChIKeys, and Murcko scaffolds using RDKit.
4. Generates zero-leakage splits:
   - train: seen drugs & seen targets
   - cold_drug: unseen drugs & seen targets
   - cold_target: seen drugs & unseen targets
   - cold_pair: strictly unseen drugs & strictly unseen targets
   - warm_test: random 5-fold CV test fold for warm-start reference
5. Builds target lookup artifacts (target_to_idx.pkl, target_sequences.pkl).
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
DATA_DIR = os.path.join(PROJECT_DIR, "data", "davis")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")

DAVIS_URLS = {
    "Y": "https://raw.githubusercontent.com/thinng/GraphDTA/master/data/davis/Y",
    "ligands": "https://raw.githubusercontent.com/thinng/GraphDTA/master/data/davis/ligands_can.txt",
    "proteins": "https://raw.githubusercontent.com/thinng/GraphDTA/master/data/davis/proteins.txt",
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


def build_davis_dataframe(raw_dir):
    ligands_file = os.path.join(raw_dir, "ligands_can.txt")
    proteins_file = os.path.join(raw_dir, "proteins.txt")
    y_file = os.path.join(raw_dir, "Y")

    with open(ligands_file, "r") as f:
        ligands_dict = json.load(f)
    with open(proteins_file, "r") as f:
        proteins_dict = json.load(f)
    with open(y_file, "rb") as f:
        y_matrix = pickle.load(f, encoding="latin1")

    y_arr = np.array(y_matrix, dtype=np.float64)  # Shape: (68, 442)
    drug_keys = list(ligands_dict.keys())         # 68 drugs
    protein_keys = list(proteins_dict.keys())     # 442 proteins

    assert y_arr.shape == (len(drug_keys), len(protein_keys)), (
        f"Shape mismatch: {y_arr.shape} vs ({len(drug_keys)}, {len(protein_keys)})"
    )

    rows = []
    for d_idx, d_key in enumerate(drug_keys):
        smiles = ligands_dict[d_key]
        inchi_key = get_inchikey(smiles)
        scaffold = get_scaffold(smiles, d_key)
        for p_idx, p_key in enumerate(protein_keys):
            kd_nm = float(y_arr[d_idx, p_idx])
            # Standard pKd transformation: -log10(Kd * 1e-9) = 9 - log10(Kd)
            pkd = 9.0 - np.log10(max(kd_nm, 1e-6))
            rows.append({
                "Drug_ID": str(d_key),
                "Canonical_SMILES": smiles,
                "Ligand_InChI_Key": inchi_key,
                "scaffold": scaffold,
                "Target_Name": str(p_key),
                "Target_Seq": proteins_dict[p_key],
                "Kd_nM": kd_nm,
                "pKd": pkd,
                "drug_idx": d_idx,
                "target_idx": p_idx,
            })

    df = pd.DataFrame(rows)
    return df, proteins_dict, ligands_dict


def create_cold_start_splits(df, seed=42, drug_test_fraction=0.20, target_test_fraction=0.20):
    """Create disjoint cold-start splits with strict leakage validation."""
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
    
    # Warm-start 5-fold split (fold 0 as held-out test)
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
    print("Leakage verification:", leakage)
    if any(leakage.values()):
        raise ValueError(f"Cold-start leakage detected: {leakage}")
        
    return splits


def main():
    raw_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    
    for name, url in DAVIS_URLS.items():
        fname = os.path.basename(url)
        download_file(url, os.path.join(raw_dir, fname))
        
    print("Building Davis dataset DataFrame...")
    df, proteins_dict, ligands_dict = build_davis_dataframe(raw_dir)
    full_csv = os.path.join(DATA_DIR, "davis_full.csv")
    df.to_csv(full_csv, index=False)
    print(f"Saved {len(df)} total pairs to {full_csv}")
    print(f"pKd range: [{df['pKd'].min():.2f}, {df['pKd'].max():.2f}], Mean: {df['pKd'].mean():.2f}, Std: {df['pKd'].std():.2f}")
    
    # Save target lookup artifacts
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

