"""Evaluate a reproducible, pIC50-stratified sample of the v3 training set."""

import csv
import hashlib
import os
import pickle

import esm
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from model_v3 import ARCHITECTURE_VERSION, DTIModelV3


PROJECT_DIR = r"C:\DTI_GNN"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "evaluation", "v3_results")

CHECKPOINT_NAME = "dti_model_v3_stable_epoch15.pt"
SAMPLE_SIZE = 256
SAMPLE_SEED = 42
BATCH_SIZE = 32
BIN_WIDTH = 1.5


def stratified_sample_indices(targets, sample_size, seed, bin_width=BIN_WIDTH):
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array.")
    if not np.isfinite(targets).all():
        raise ValueError("Training targets contain non-finite values.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    rng = np.random.default_rng(seed)
    if sample_size >= targets.size:
        indices = np.arange(targets.size)
        rng.shuffle(indices)
        return indices

    lower_edge = np.floor(targets.min() / bin_width) * bin_width
    upper_edge = np.ceil(targets.max() / bin_width) * bin_width
    edges = np.arange(lower_edge, upper_edge + bin_width, bin_width)
    bin_ids = np.digitize(targets, edges[1:-1])
    strata = [
        np.flatnonzero(bin_ids == bin_id)
        for bin_id in np.unique(bin_ids)
    ]
    allocations = np.zeros(len(strata), dtype=np.int64)

    while allocations.sum() < sample_size:
        active = [
            index
            for index, stratum in enumerate(strata)
            if allocations[index] < len(stratum)
        ]
        if not active:
            raise RuntimeError("Could not fill the requested stratified sample.")
        share = max(1, int(np.ceil(
            (sample_size - allocations.sum()) / len(active)
        )))
        for index in active:
            available = len(strata[index]) - allocations[index]
            take = min(share, available, sample_size - allocations.sum())
            allocations[index] += take
            if allocations.sum() == sample_size:
                break

    selected = np.concatenate([
        rng.choice(stratum, size=count, replace=False)
        for stratum, count in zip(strata, allocations)
        if count
    ])
    rng.shuffle(selected)
    return selected


def graph_record_id(graph):
    digest = hashlib.sha256()
    for tensor in (graph.x, graph.edge_index):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_path = os.path.join(DATA_DIR, "train_data_v2.pt")
    sequences_path = os.path.join(DATA_DIR, "target_sequences.pkl")
    target_mapping_path = os.path.join(DATA_DIR, "target_to_idx.pkl")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
    for path in (train_path, sequences_path, target_mapping_path, checkpoint_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required model-evaluation file not found: {path}")

    training_data = torch.load(train_path, map_location="cpu", weights_only=False)
    targets = np.asarray(
        [float(sample.y.item()) for sample in training_data],
        dtype=np.float64,
    )
    sample_indices = stratified_sample_indices(
        targets,
        SAMPLE_SIZE,
        SAMPLE_SEED,
    )
    selected_data = [training_data[int(index)] for index in sample_indices]

    with open(sequences_path, "rb") as source:
        target_sequences = pickle.load(source)
    with open(target_mapping_path, "rb") as source:
        target_to_idx = pickle.load(source)
    target_ids = {int(index): accession for accession, index in target_to_idx.items()}

    print(
        f"Selected {len(selected_data):,} of {len(training_data):,} training examples "
        f"(stratified by {BIN_WIDTH:g} pIC50, seed={SAMPLE_SEED})."
    )
    print(
        "NOTE: These examples were used during training; this is an in-sample "
        "diagnostic, not an unbiased test-set accuracy estimate."
    )
    print("Drug identifiers below are graph fingerprints; the saved graph data has no SMILES.")

    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    for parameter in esm_model.parameters():
        parameter.requires_grad = False
    for parameter in esm_model.layers[-1].parameters():
        parameter.requires_grad = True
    esm_model = esm_model.to(device)

    model = DTIModelV3(
        drug_in_channels=18,
        esm_model=esm_model,
        batch_converter=alphabet.get_batch_converter(),
        target_sequences=target_sequences,
        hidden_dim=128,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            f"Checkpoint architecture version "
            f"{checkpoint.get('architecture_version')} does not match "
            f"model version {ARCHITECTURE_VERSION}."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(selected_data), BATCH_SIZE):
            chunk = selected_data[start : start + BATCH_SIZE]
            batch = next(iter(DataLoader(
                chunk,
                batch_size=len(chunk),
                shuffle=False,
            ))).to(device)
            prediction = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.target_idx,
                device,
            )
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(
                    f"Non-finite prediction in sample batch starting at {start}."
                )
            if prediction.abs().max().item() > 20:
                raise ValueError(
                    "Prediction outside the expected pIC50 scale "
                    "(absolute value > 20)."
                )
            predictions.extend(prediction.cpu().numpy().tolist())

    actuals = np.asarray(
        [float(sample.y.item()) for sample in selected_data],
        dtype=np.float64,
    )
    predictions = np.asarray(predictions, dtype=np.float64)
    errors = np.abs(predictions - actuals)
    rows = []
    for index, sample, actual, prediction, error in zip(
        sample_indices,
        selected_data,
        actuals,
        predictions,
        errors,
    ):
        target_idx = int(sample.target_idx.item())
        if target_idx not in target_ids:
            raise KeyError(f"Target index {target_idx} is missing from target_to_idx.pkl.")
        rows.append({
            "training_index": int(index),
            "drug_graph_sha256": graph_record_id(sample),
            "target_accession": target_ids[target_idx],
            "target_idx": target_idx,
            "actual_pIC50": float(actual),
            "predicted_pIC50": float(prediction),
            "absolute_error": float(error),
        })

    mae = float(errors.mean())
    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    print(f"\nSamples: {len(rows):,}")
    print(f"MAE: {mae:.4f} pIC50")
    print(f"RMSE: {rmse:.4f} pIC50")

    print("\nFirst 20 predictions vs targets")
    print("-" * 112)
    for row in rows[:20]:
        print(
            f"Index {row['training_index']:>6} | "
            f"Drug graph {row['drug_graph_sha256'][:12]} | "
            f"Target {row['target_accession']} | "
            f"Actual {row['actual_pIC50']:7.4f} | "
            f"Predicted {row['predicted_pIC50']:7.4f} | "
            f"Error {row['absolute_error']:7.4f}"
        )

    print("\nLargest 10 prediction errors")
    print("-" * 112)
    for row in sorted(rows, key=lambda item: item["absolute_error"], reverse=True)[:10]:
        print(
            f"Index {row['training_index']:>6} | "
            f"Drug graph {row['drug_graph_sha256'][:12]} | "
            f"Target {row['target_accession']} | "
            f"Actual {row['actual_pIC50']:7.4f} | "
            f"Predicted {row['predicted_pIC50']:7.4f} | "
            f"Error {row['absolute_error']:7.4f}"
        )

    report_path = os.path.join(OUTPUT_DIR, "train_sample_v3_report.txt")
    csv_path = os.path.join(OUTPUT_DIR, "train_sample_v3_predictions.csv")
    npz_path = os.path.join(OUTPUT_DIR, "train_sample_v3_predictions.npz")

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("DTI-GNN V3 TRAINING-SAMPLE DIAGNOSTIC\n")
        report.write("=" * 80 + "\n")
        report.write(f"Checkpoint: {CHECKPOINT_NAME}\n")
        report.write(f"Checkpoint epoch: {checkpoint['epoch']}\n")
        report.write(f"Architecture version: {ARCHITECTURE_VERSION}\n")
        report.write(f"Samples: {len(rows)} of {len(training_data)}\n")
        report.write(f"Sampling: stratified pIC50 bins, seed={SAMPLE_SEED}\n")
        report.write(
            "Interpretation: in-sample diagnostic; these records were used to train "
            "the checkpoint.\n"
        )
        report.write(
            "Drug labels are graph fingerprints because the saved graph dataset "
            "does not include SMILES.\n"
        )
        report.write(f"MAE: {mae:.6f}\nRMSE: {rmse:.6f}\n\n")
        report.write("First 20 predictions vs targets\n")
        for row in rows[:20]:
            report.write(
                f"Index {row['training_index']} | Drug graph "
                f"{row['drug_graph_sha256'][:12]} | Target "
                f"{row['target_accession']} | Actual {row['actual_pIC50']:.4f} | "
                f"Predicted {row['predicted_pIC50']:.4f} | "
                f"Error {row['absolute_error']:.4f}\n"
            )
        report.write("\nLargest 10 prediction errors\n")
        for row in sorted(
            rows,
            key=lambda item: item["absolute_error"],
            reverse=True,
        )[:10]:
            report.write(
                f"Index {row['training_index']} | Drug graph "
                f"{row['drug_graph_sha256'][:12]} | Target "
                f"{row['target_accession']} | Actual {row['actual_pIC50']:.4f} | "
                f"Predicted {row['predicted_pIC50']:.4f} | "
                f"Error {row['absolute_error']:.4f}\n"
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        npz_path,
        training_indices=sample_indices,
        predictions=predictions,
        targets=actuals,
        errors=errors,
    )
    print(f"\nSaved report: {report_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved arrays: {npz_path}")


if __name__ == "__main__":
    main()
