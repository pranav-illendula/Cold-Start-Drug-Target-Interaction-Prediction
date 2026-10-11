# data/

Local data files. Everything here except this README is ignored by git.

Expected files:

- `train_data_v2.pt` : training graphs
- `target_to_idx.pkl` : target accession to index mapping
- `target_sequences.pkl` : target index to protein sequence
- `BindingDB_final_clean.csv` : cleaned BindingDB input for `scripts/prepare_splits.py`
- `splits/` : CSV splits written by `scripts/prepare_splits.py`

The cold-start evaluation tensors (`cold_drug_data_v2.pt`, `cold_target_data_v2.pt`,
`cold_pair_data_v2.pt`) live in `evaluation/`, not here.
