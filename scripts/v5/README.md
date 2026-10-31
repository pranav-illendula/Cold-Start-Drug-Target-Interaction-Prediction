# Version 5 Enhancement Package

This folder contains the Section 4 model, its data preparation and evaluation scripts, and the shared graph-feature helpers required to reproduce it. The original v1-v4 scripts remain in the parent `scripts/` folder.

## Package contents

- `model_enhancements_v5.py`: sliding-window target pooling, GINE drug encoding, atom-to-residue cross-attention, and ranking-regression loss.
- `generate_esm_residue_embeddings_v5.py`: generates frozen ESM-2 window and per-residue features.
- `train_enhancements_v5.py`: trains the target-held-out ensemble with default seeds 42-46 and early stopping.
- `evaluate_enhancements_v5.py`: evaluates the ensemble and calibrates target-group conformal intervals.
- `write_enhancement_report_v5.py` and `report_metrics_v5.py`: write a v4-style text report from evaluation outputs.
- `build_recovered_graphs_v5.py`, `build_graphs_v5.py`, and `bond_features_v5.py`: validate and build graph artifacts; the latter two are bundled v5 copies of shared graph helpers.
- `validate_recovered_dataset_v5.py`: validates split CSVs and target mapping/sequence alignment.

## Requirements

Use the repository virtual environment and install the project requirements. Install a PyTorch build compatible with the machine's CUDA runtime before the remaining requirements if GPU training is desired. CUDA training automatically uses FP16 autocast and gradient scaling. `torch-scatter` is an optional PyG acceleration; use the wheel index matching the installed PyTorch, CUDA, and Python versions.

The scripts expect a versioned recovered dataset containing `target_to_idx.pkl`, `target_sequences.pkl`, train-fit and calibration CSVs, and cold-drug, cold-target, and cold-pair CSVs. The CSVs must have `Canonical_SMILES`, `Target_Name`, `UniProt_ID`, and `pIC50` columns. Keep one target mapping consistent across graph data and ESM features; the manifests enforce this alignment.

## Reproduction workflow

Run commands from the repository root in PowerShell. Set these paths to the actual recovered dataset and split CSVs on your machine:

```powershell
$py = 'python'
$root = (Get-Location).Path
$data = Join-Path $root 'data\recovered_baseline_v1'
$map = Join-Path $data 'target_to_idx.pkl'
$sequences = Join-Path $data 'target_sequences.pkl'
$trainFitCsv = Join-Path $data 'calibration\train_fit.csv'
$calibrationCsv = Join-Path $data 'calibration\calibration.csv'
$coldDrugCsv = 'C:\path\to\cold_drug_with_uniprot.csv'
$coldTargetCsv = 'C:\path\to\cold_target_with_uniprot.csv'
$coldPairCsv = 'C:\path\to\cold_pair_with_uniprot.csv'
$graphDir = Join-Path $data 'graphs'
$embeddingDir = Join-Path $data 'esm2'
$checkpointDir = Join-Path $root 'checkpoints\version5_run'
$evaluationDir = Join-Path $root 'evaluation\section4_run'
$reportPath = Join-Path $root 'results\version5\evaluation_report_section4.txt'
```

Validate the recovered inputs, then build all five graph datasets. Replace the three cold-split CSV variables above with the files for the experiment.

```powershell
& $py 'scripts\v5\validate_recovered_dataset_v5.py' `
  --train-csv $trainFitCsv `
  --cold-drug-csv $coldDrugCsv `
  --cold-target-csv $coldTargetCsv `
  --cold-pair-csv $coldPairCsv `
  --target-to-idx $map `
  --target-sequences $sequences

& $py 'scripts\v5\build_recovered_graphs_v5.py' `
  --train-fit-csv $trainFitCsv `
  --calibration-csv $calibrationCsv `
  --cold-drug-csv $coldDrugCsv `
  --cold-target-csv $coldTargetCsv `
  --cold-pair-csv $coldPairCsv `
  --target-to-idx $map `
  --output-dir $graphDir
```

Generate protein features, train the five seeds, evaluate the ensemble, and write the text report. Each output directory must be new or empty; use a different run folder for a new training or evaluation run.

```powershell
& $py 'scripts\v5\generate_esm_residue_embeddings_v5.py' `
  --target-to-idx $map `
  --target-sequences $sequences `
  --output-dir $embeddingDir `
  --device cuda

& $py 'scripts\v5\train_enhancements_v5.py' `
  --train-graphs (Join-Path $graphDir 'train_fit_data_v4.pt') `
  --embedding-dir $embeddingDir `
  --output-dir $checkpointDir `
  --device cuda

& $py 'scripts\v5\evaluate_enhancements_v5.py' `
  --calibration-graphs (Join-Path $graphDir 'calibration_data_v4.pt') `
  --cold-drug-graphs (Join-Path $graphDir 'cold_drug_data_v4.pt') `
  --cold-target-graphs (Join-Path $graphDir 'cold_target_data_v4.pt') `
  --cold-pair-graphs (Join-Path $graphDir 'cold_pair_data_v4.pt') `
  --checkpoint-dir $checkpointDir `
  --embedding-dir $embeddingDir `
  --output-dir $evaluationDir `
  --device cuda

& $py 'scripts\v5\write_enhancement_report_v5.py' `
  --evaluation-dir $evaluationDir `
  --checkpoint-dir $checkpointDir `
  --output $reportPath `
  --device cuda
```

Use `--device cpu` when CUDA is unavailable. Training supports `--epochs`, `--batch-size`, `--patience`, `--seeds`, and optimizer/loss options; run a script with `--help` for the complete CLI. The evaluator's target-group conformal intervals are calibrated on a held-out target set; coverage under a different cold-target distribution is not guaranteed.
