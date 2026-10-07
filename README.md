# Cold-Start Drug-Target Interaction Prediction with a Two-Tower GNN

Predict BindingDB drug-target pIC50 from molecular graphs and protein sequences, with cold-start evaluation for unseen drugs, targets, and drug-target pairs.

## Problem and motivation

Drug-target interaction (DTI) prediction estimates the binding affinity between a small-molecule drug and a protein target. This project treats pIC50 as a regression target. A random split can put close analogues, the same drug, or the same target on both sides of evaluation, making performance look better than it is for genuinely new entities.

Cold-start prediction is harder because one or both entities at test time were not represented in training:

- **Cold drug:** unseen drug/scaffold.
- **Cold target:** unseen protein target.
- **Cold pair:** a drug-target interaction withheld as a pair.

## Dataset

- **Source:** [BindingDB](https://www.bindingdb.org/rwd/bind/BindingDB_All.tsv.zip) (downloadable TSV archive).
- The raw BindingDB data and prepared graph tensors are **not included in this repository**.
- The project data preparation used approximately **997K BindingDB rows** before graph construction and filtering. The prepared training and cold-start tensors contain a smaller number of usable examples.
- Download and use BindingDB in accordance with its terms and citation requirements.

## Data splits and leakage

The prepared splits are intended to support scaffold-aware and cold-start evaluation:

| Split file | Intended evaluation |
|---|---|
| `data/train_data_v2.pt` | Training examples |
| `evaluation/cold_drug_data_v2.pt` | Held-out drugs/scaffolds |
| `evaluation/cold_target_data_v2.pt` | Held-out protein targets |
| `evaluation/cold_pair_data_v2.pt` | Held-out drug-target pairs |

The prepared split protocol includes leakage checks between training and the corresponding cold-start holdout (drug/scaffold identity for cold-drug, target identity for cold-target, and pair identity for cold-pair). The split tensors are pre-generated; **the repository does not currently contain the original split-generation or standalone leakage-audit script**, so the exact split assignment and checks cannot be independently rerun from this repository alone. Keep the source-row identifiers and split-generation code with the dataset if exact reproducibility is required. A fresh split should preserve scaffold groups and re-run explicit train/holdout overlap checks before evaluation.

To regenerate splits, download the BindingDB source above, repeat the project's original curation and pIC50 conversion, build the drug graphs and protein-target index, apply the scaffold/cold-entity split protocol, run overlap checks, and serialize the four files to the paths above. The preparation script and its exact command need to be added to this repository to make that process fully reproducible.

## Model architecture

The current model is a two-tower regressor:

1. **Drug tower:** three graph convolution (`GCNConv`) layers operate on the molecular graph (18 atom features per node), followed by global mean pooling.
2. **Protein tower:** ESM-2 35M produces 480-dimensional residue embeddings from the protein sequence. Residues are mean-pooled; the final ESM transformer layer is fine-tuned while earlier layers are frozen. A 480 → 256 → 128 projection uses LayerNorm, ReLU, and dropout.
3. **Fusion and output:** drug and protein vectors are concatenated and passed through an MLP (256 → 128 → 64 → 1) to predict pIC50.

The current shared implementation is in `model_v3.py`. Architecture version 2 of v3 uses LayerNorm in the target projection to avoid the collapsed BatchNorm running variance found in the previous checkpoint.

## Model versions

- **v1:** Initial two-tower baseline; ESM protein embeddings were frozen.
- **v2:** Tested a deeper target tower while retaining frozen ESM embeddings. Results indicated that target-tower depth alone was not the main bottleneck.
- **v3:** Fine-tunes the last ESM layer and uses a pIC50 range-weighted regression loss. The stable v3 architecture version 2 also replaces target-projection BatchNorm with LayerNorm after the original checkpoint produced extreme predictions.

## Results

Pearson correlation (`r`) on the available cold-start evaluation report:

| Version | Cold drug | Cold target | Cold pair |
|---|---:|---:|---:|
| v1 | N/A — verified result not available in this repository | N/A | N/A |
| v2 | N/A — verified result not available in this repository | N/A | N/A |
| v3 (stable, architecture v2) | **0.5528** | **0.2223** | **0.2357** |

The v3 values are from `evaluation/v3_results/v3_summary.csv`, using checkpoint `dti_model_v3_stable_epoch15.pt`. Cold-target and cold-pair performance is substantially weaker than cold-drug performance. Do not compare versions until their scores are measured using the same splits and metrics.

## Known limitations

- Performance is weak at high pIC50 values (roughly **≥ 8**) and very low values (**< 4**), with predictions tending toward the middle of the label range.
- The training distribution is sparse at the highest values: the inspected training set has 739 examples from 10.5–12 and 27 examples above 12. Loss weighting cannot replace representative, diverse examples.
- Cold-target and cold-pair Pearson correlations are low. The current model should not be treated as highly accurate for unseen targets or pairs.
- The prepared graph tensors do not include drug names or SMILES; diagnostics identify a drug graph by a hash unless original drug metadata is supplied.

## Repository structure

- `model_v3.py` — shared v3 architecture (stable architecture version 2).
- `train_v3.py` — training loop; writes one checkpoint per epoch.
- `evaluate_v3.py` — cold-start evaluation, metrics, and report generation.
- `evaluate_train_sample_v3.py` — stratified in-sample training diagnostic.
- `generate_v3_change_report.py` — generates the detailed architecture/results PDF.
- `tests/` — architecture and diagnostic regression tests.
- `DTI_GNN_v3.ipynb` — v3 development/training notebook.
- `evaluate_test_data.ipynb` — evaluation notebook.
- `data/` — local training tensors and protein mappings; not tracked.
- `evaluation/` — local cold-start tensors and generated results; not tracked.
- `checkpoints/` — local model checkpoints; not tracked.

## Installation

The environment used for the reported run was Python **3.11.7**, PyTorch **2.11.0+cu128**, PyTorch Geometric **2.8.0.post1**, Windows 11 Pro for Workstations, and an NVIDIA RTX 2000 Ada Generation GPU.

Create and activate a virtual environment (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the CUDA 12.8 PyTorch build used for the reported run, then the project's pinned dependencies:

```powershell
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

PyTorch/CUDA wheels are hardware- and driver-dependent. If using a different GPU or platform, choose the compatible PyTorch command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining requirements. The pinned `requirements.txt` reflects the development environment and may need adjustment for other platforms.

## How to run

Place the prepared data files in the paths listed in **Data splits and leakage** and the trained model files, if evaluating an existing model, in `checkpoints/`. Split preparation is currently an external prerequisite: this repository does not provide a command to regenerate the `.pt` datasets from raw BindingDB.

From the repository root, train first and then evaluate the cold-start splits:

```powershell
python train_v3.py
python evaluate_v3.py
```

`train_v3.py` reads `data/train_data_v2.pt`, `data/target_to_idx.pkl`, and `data/target_sequences.pkl`, and writes checkpoints named `checkpoints/dti_model_v3_stable_epoch1.pt` through `dti_model_v3_stable_epoch15.pt`. It automatically resumes from the latest stable checkpoint; if epoch 15 already exists, it has no additional epochs to run. To start a fresh run without deleting the previous model, archive the stable checkpoints first, for example in PowerShell:

```powershell
$archive = "checkpoints\archive_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path $archive
Move-Item checkpoints\dti_model_v3_stable_epoch*.pt -Destination $archive
python train_v3.py
```

`evaluate_v3.py` loads the three cold-start split files from `evaluation/` and the epoch-15 stable checkpoint. Results are saved under `evaluation/v3_results/`.

To run the stratified diagnostic on a sample of the training data:

```powershell
python evaluate_train_sample_v3.py
```

This is an **in-sample diagnostic**, not an unbiased test score.

## Hardware and runtime

The reported training run used an NVIDIA RTX 2000 Ada Generation GPU on Windows 11 Pro for Workstations and took about **75–90 minutes for 15 epochs** (approximately five to six minutes per epoch in the recorded training log). Runtime varies with GPU, thermals, and software versions.

## Future work

- Add more quality-controlled, chemically and biologically diverse high-pIC50 examples.
- Create a reproducible split-generation and leakage-audit script, with a target-held-out validation set for model selection.
- Compare loss functions and reweighting strategies on validation data without tuning on the cold-start test set.
- Evaluate fine-tuning additional ESM layers or alternative protein encoders.
- Preserve source drug identifiers/SMILES alongside graph tensors to make prediction reports human-readable.

## Author

**Pranav Illendula** — [pranav.illendula@gmail.com](mailto:pranav.illendula@gmail.com)
