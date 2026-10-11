import pandas as pd
df = pd.read_csv("data/BindingDB_final_clean.csv")
df.head()

print(df.shape)

!pip install rdkit

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None

df['scaffold'] = df['Canonical_SMILES'].apply(get_scaffold)

# fallback: rows where scaffold failed get their own singleton group
missing = df['scaffold'].isna()
df.loc[missing, 'scaffold'] = df.loc[missing, 'Ligand_InChI_Key']

print(f"Missing scaffolds (fell back to InChIKey): {missing.sum()}")
print(f"Unique scaffolds: {df['scaffold'].nunique()}")

import numpy as np

def scaffold_balanced_split(df, scaffold_col, test_frac, seed=42):
    scaffold_counts = df.groupby(scaffold_col).size().sort_values(ascending=False)
    rng = np.random.RandomState(seed)
    scaffold_list = list(scaffold_counts.index)
    rng.shuffle(scaffold_list)
    scaffold_list.sort(key=lambda s: -scaffold_counts[s])

    total = len(df)
    target_test_size = int(total * test_frac)

    test_groups, train_groups = set(), set()
    test_count = 0
    for scaf in scaffold_list:
        if test_count < target_test_size:
            test_groups.add(scaf)
            test_count += scaffold_counts[scaf]
        else:
            train_groups.add(scaf)
    return train_groups, test_groups


def group_split(groups, test_frac, seed=42):
    rng = np.random.RandomState(seed)
    unique_groups = np.array(groups.dropna().unique(), dtype=object)
    rng.shuffle(unique_groups)
    n_test = int(len(unique_groups) * test_frac)
    test_groups = set(unique_groups[:n_test])
    train_groups = set(unique_groups[n_test:])
    return train_groups, test_groups

target_train_groups, target_test_groups = group_split(
    df['Target_Name'], test_frac=0.15, seed=42
)

print(f"Train target groups: {len(target_train_groups)}")
print(f"Test target groups: {len(target_test_groups)}")


scaffolds_per_drug = df.groupby('Ligand_InChI_Key')['scaffold'].nunique()
multi_scaffold_drugs = scaffolds_per_drug[scaffolds_per_drug > 1]
print(f"Drugs with >1 scaffold: {len(multi_scaffold_drugs)}")
multi_scaffold_drugs.sort_values(ascending=False).head()

# pick the most frequent scaffold for each drug as its canonical scaffold
primary_scaffold = (
    df.groupby('Ligand_InChI_Key')['scaffold']
    .agg(lambda x: x.value_counts().idxmax())
)

df['scaffold'] = df['Ligand_InChI_Key'].map(primary_scaffold)

print(f"Unique scaffolds after consolidation: {df['scaffold'].nunique()}")

# re-run the scaffold split with the fixed column
drug_train_scaffolds, drug_test_scaffolds = scaffold_balanced_split(
    df, 'scaffold', test_frac=0.2, seed=42
)
print(f"Train scaffold groups: {len(drug_train_scaffolds)}")
print(f"Test scaffold groups: {len(drug_test_scaffolds)}")

is_test_drug = df['scaffold'].isin(drug_test_scaffolds)
is_test_target = df['Target_Name'].isin(target_test_groups)

quadrants = {
    'train':       df[~is_test_drug & ~is_test_target],
    'cold_drug':   df[ is_test_drug & ~is_test_target],
    'cold_target': df[~is_test_drug &  is_test_target],
    'cold_pair':   df[ is_test_drug &  is_test_target],
}

for name, sub in quadrants.items():
    print(f"{name:<12} {len(sub):>8} rows")

# leakage checks
train_drugs = set(quadrants['train']['Ligand_InChI_Key'])
train_targets = set(quadrants['train']['Target_Name'])

cd_leak = quadrants['cold_drug']['Ligand_InChI_Key'].isin(train_drugs).sum()
ct_leak = quadrants['cold_target']['Target_Name'].isin(train_targets).sum()
cp_drug_leak = quadrants['cold_pair']['Ligand_InChI_Key'].isin(train_drugs).sum()
cp_target_leak = quadrants['cold_pair']['Target_Name'].isin(train_targets).sum()

print("\nLeakage checks (all should be 0):")
print(f"  cold_drug rows with drug also in train:     {cd_leak}")
print(f"  cold_target rows with target also in train: {ct_leak}")
print(f"  cold_pair rows with drug in train:           {cp_drug_leak}")
print(f"  cold_pair rows with target in train:         {cp_target_leak}")


import os
save_dir = 'data/splits'
os.makedirs(save_dir, exist_ok=True)

for name, sub in quadrants.items():
    path = f'{save_dir}/split_{name}.csv'
    sub.to_csv(path, index=False)
    print(f"Saved {name}: {len(sub)} rows -> {path}")


