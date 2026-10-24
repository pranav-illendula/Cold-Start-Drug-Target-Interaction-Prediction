# data/

Local data files for the cold-start DTI model. Everything in this folder except this README is ignored by git (see `.gitignore`), so these files must be created or copied in locally.

## Target lookup files

The model never receives a protein sequence directly in the training tensors. Each drug–target example stores only an integer `target_idx`. The two files below translate that integer back into a protein.

| File | Type | Size | Maps |
|---|---|---|---|
| `target_to_idx.pkl` | `dict[str, int]` | 2,667 entries | UniProt accession (e.g. `P00533`) → integer index |
| `target_sequences.pkl` | `list[str]` | 2,667 entries | integer index → amino-acid sequence |

### How they fit together

```
graph.target_idx  ──►  target_sequences[target_idx]  ──►  ESM-2 150M ──►  protein vector
UniProt accession ──►  target_to_idx[accession]       ──►  target_idx
```

- `target_to_idx.pkl` assigns every target a unique index from `0` to `2666`, with no gaps.
- `target_sequences.pkl` is a plain list ordered by that same index, so `target_sequences[i]` is the sequence of the target whose index is `i`.
- Sequence lengths range from 71 to 1,022 residues (median 503). The 1,022 ceiling matches the ESM-2 input limit, so no sequence needs further truncation.

### Who reads them

- `scripts/train_v4.py`: loads both files and uses the sequence-index order to validate training graph targets.
- `scripts/evaluate_v4.py`: loads `target_sequences.pkl` to validate cached ESM embeddings before loading a checkpoint.
- `scripts/evaluate_train_sample.py`: loads both. It inverts `target_to_idx` to report the UniProt accession for each prediction.

### Keep these consistent

The index order is baked into the training tensors and into every checkpoint. If you rebuild either file with a different order or a different set of targets, the existing `train_data_v2.pt`, `cold_*_data_v2.pt` files and checkpoints will point at the wrong proteins. Regenerate all of them together, or none.

### Loading them

```python
import pickle

with open("data/target_to_idx.pkl", "rb") as f:
    target_to_idx = pickle.load(f)        # {"P49653": 0, "P00533": 1, ...}

with open("data/target_sequences.pkl", "rb") as f:
    target_sequences = pickle.load(f)     # ["MVRRLARG...", "MRPSGTAG...", ...]

seq = target_sequences[target_to_idx["P00533"]]
```

Only unpickle files you created yourself or trust. Pickle can run arbitrary code when loaded.

### Regenerating

`scripts/build_target_sequences.py` maps BindingDB target names to UniProt IDs, downloads the sequences from UniProt, and should write both files. Check that it keeps the index order stable (see above).

## Other files expected in this folder

| File | Purpose |
|---|---|
| `train_data_v2.pt` | Legacy training graphs (one PyG `Data` object per drug–target pair: 18 atom features, `edge_index`, `target_idx`, `y` = pIC50; no bond `edge_attr`) |
| `train_data_v4.pt` | Rebuilt v4 training graphs with 18 atom features and directed 14-d `edge_attr` bond features |
| `BindingDB_final_clean.csv` | Cleaned BindingDB table used as input to `scripts/prepare_splits.py` |
| `splits/` | CSV splits (`split_train`, `split_cold_drug`, `split_cold_target`, `split_cold_pair`) written by `scripts/prepare_splits.py` |

Legacy cold-start evaluation tensors (`cold_drug_data_v2.pt`, `cold_target_data_v2.pt`, `cold_pair_data_v2.pt`) live in `evaluation/`, not here. Rebuilt v4 tensors use the `_data_v4.pt` suffix and leave legacy files untouched.

## v4 graph features

The updated v4 model requires `edge_attr` on every training and evaluation graph. It must have shape `[number_of_directed_edges, 14]`, with rows aligned to `edge_index`:

1. Bond type one-hot: single, double, triple, aromatic, dative.
2. Conjugated and in-ring flags.
3. Stereo one-hot: none, any, Z, E, cis, trans.
4. A donor-to-acceptor direction flag, set on the forward dative edge only.

Each undirected molecular bond must be represented by both directed edges; dative edges have distinct direction-flag values. Existing graph tensors with the previous bond feature width must be regenerated before running `scripts/train_v4.py` or `scripts/evaluate_v4.py`. Training and evaluation intentionally stop with a descriptive error rather than silently substituting missing bond information.

`scripts/build_graphs_v4.py` calls `bond_features_v4.molecule_bond_features(rdkit_molecule)` to produce directed `edge_index` entries and their aligned feature rows.

### Building the v4 graph tensors

From an activated project virtual environment, install a CUDA-compatible PyTorch build using the command provided by the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the project requirements:

```powershell
python -m pip install -r requirements.txt
```

If you have the cleaned CSV but not its split files, create the leakage-checked splits:

```powershell
python -m scripts.prepare_splits --input data\BindingDB_final_clean.csv --output-dir data\splits
```

Then build the v4 tensors:

```powershell
python -m scripts.build_graphs_v4 --bindingdb-tsv data\BindingDB_All_202609_tsv.zip
```

Replace the example BindingDB ZIP path with your actual local TSV/ZIP path. The graph builder reads the four split CSVs from `data/splits/`; it uses `Canonical_SMILES`, `Target_Name`, and `pIC50` by default. It maps names to indices through `data/splits/target_to_uniprot.pkl` when that file exists; otherwise it reads the target-name and UniProt ID columns from the supplied BindingDB TSV/ZIP. You can instead omit `--bindingdb-tsv` if the split CSVs already have a `UniProt_ID` column. Every UniProt ID must exist in the existing `data/target_to_idx.pkl`, whose ordering must remain unchanged.

If a split contains UniProt IDs missing from `target_to_idx.pkl`, append those IDs and their sequences together before graph generation:

```powershell
python -m scripts.extend_target_index `
  --train-csv "C:\path\to\split_train_with_uniprot.csv" `
  --cold-drug-csv "C:\path\to\split_cold_drug_with_uniprot.csv" `
  --cold-target-csv "C:\path\to\split_cold_target_with_uniprot.csv" `
  --cold-pair-csv "C:\path\to\split_cold_pair_with_uniprot.csv"
```

The script preserves existing indices, fetches sequences for new UniProt IDs, and appends entries in sorted ID order. It updates `data/target_to_idx.pkl` and `data/target_sequences.pkl` together. Extending the target list changes the ESM sequence fingerprint and training targets; move aside the stale embedding cache `data/esm2_t30_150M_cls_mean_embeddings.pt`, resume checkpoints `checkpoints/dti_model_v4_gine_last.pt` and `checkpoints/dti_model_v4_gine_best.pt`, and saved target-validation split `data/v4_target_validation_split.json` before retraining from scratch. If a UniProt request fails, the script stops before updating these files; rerun it after network access is restored.

The rebuild process preserves the split files and never repartitions data. It creates an explicit 18-dimensional atom encoding: element one-hot for C, N, O, S, P, F, Cl, Br, other; total-degree one-hot for 0–4 and 5-or-more; and formal-charge sign one-hot for negative, zero, positive. Bond features are the 14 dimensions described above. This explicitly defined encoder is used because the original v2 atom-feature generation code is not in the repository; retrain the model from scratch on these rebuilt tensors.

The script validates each split before publishing the new files and writes temporary outputs first, so a failed rebuild leaves the existing v2 tensors untouched. Outputs:

- `data/train_data_v4.pt`
- `evaluation/cold_drug_data_v4.pt`
- `evaluation/cold_target_data_v4.pt`
- `evaluation/cold_pair_data_v4.pt`

Run training and evaluation from the repository root:

```powershell
python -m scripts.train_v4
python -m scripts.evaluate_v4
```

The training script creates a target-held-out validation split and derives the capped inverse-frequency pIC50 loss weights only from that split's training examples.
