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
graph.target_idx  ──►  target_sequences[target_idx]  ──►  ESM-2 35M  ──►  protein vector
UniProt accession ──►  target_to_idx[accession]       ──►  target_idx
```

- `target_to_idx.pkl` assigns every target a unique index from `0` to `2666`, with no gaps.
- `target_sequences.pkl` is a plain list ordered by that same index, so `target_sequences[i]` is the sequence of the target whose index is `i`.
- Sequence lengths range from 71 to 1,022 residues (median 503). The 1,022 ceiling matches the ESM-2 input limit, so no sequence needs further truncation.

### Who reads them

- `scripts/train.py`: loads both files. `target_to_idx` gives the number of targets, and `target_sequences` is passed to the target encoder, which looks up the sequences for the targets in each batch and runs them through ESM-2.
- `scripts/evaluate.py`: loads `target_sequences.pkl` to rebuild the same target encoder before loading a checkpoint.
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
| `train_data_v2.pt` | Training graphs (one PyG `Data` object per drug–target pair: atom features, edges, `target_idx`, `y` = pIC50) |
| `BindingDB_final_clean.csv` | Cleaned BindingDB table used as input to `scripts/prepare_splits.py` |
| `splits/` | CSV splits (`split_train`, `split_cold_drug`, `split_cold_target`, `split_cold_pair`) written by `scripts/prepare_splits.py` |

The cold-start evaluation tensors (`cold_drug_data_v2.pt`, `cold_target_data_v2.pt`, `cold_pair_data_v2.pt`) live in `evaluation/`, not here.
