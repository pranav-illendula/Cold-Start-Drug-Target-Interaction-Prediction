"""Evaluate the selected frozen-ESM-2 150M model on cold-start splits."""

import csv
from datetime import datetime
import hashlib
import os
import pickle
import time

import numpy as np
import torch
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

from metrics_v4 import concordance_index
from model_v4_2 import (
    ARCHITECTURE_VERSION,
    BOND_FEATURE_DIM,
    DTIModelV4,
    ESM_MODEL_NAME,
)


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EVALUATION_DIR = os.path.join(PROJECT_DIR, "evaluation")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(EVALUATION_DIR, "v4_results")
REPORT_DIR = os.path.join(PROJECT_DIR, "results", "version3")
REPORT_PATH = os.path.join(REPORT_DIR, "evaluation_report_v4.txt")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dti_model_v4_gine_best.pt")
CACHE_PATH = os.path.join(DATA_DIR, "esm2_t30_150M_cls_mean_embeddings.pt")
BATCH_SIZE = 64
BIN_WIDTH = 1.5
TOLERANCES = (0.5, 1.0, 1.5)


def sequence_fingerprint(target_sequences):
    digest = hashlib.sha256()
    for index, sequence in enumerate(target_sequences):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(sequence).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def score(targets, predictions):
    errors = predictions - targets
    return {
        "samples": len(targets),
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "pearson_r": (
            float(pearsonr(targets, predictions).statistic)
            if np.std(targets) > 0 and np.std(predictions) > 0
            else float("nan")
        ),
        "ci": concordance_index(targets, predictions),
        "accuracy_0_5": float(np.mean(np.abs(errors) <= 0.5) * 100),
        "accuracy_1_0": float(np.mean(np.abs(errors) <= 1.0) * 100),
        "accuracy_1_5": float(np.mean(np.abs(errors) <= 1.5) * 100),
    }


def validate_prediction_scale(values, description):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f"{description} contain non-finite values.")
    if values.size and np.max(np.abs(values)) > 20:
        raise ValueError(
            f"{description} are outside a reasonable pIC50 range "
            f"({values.min():.4f} to {values.max():.4f})."
        )


def score_by_pic50_bin(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("Targets and predictions must be equal-length 1D arrays.")
    if not len(targets):
        raise ValueError("Cannot calculate pIC50-bin metrics for an empty split.")

    bin_indices = np.floor(targets / BIN_WIDTH).astype(np.int64)
    results = []
    for bin_index in range(int(bin_indices.min()), int(bin_indices.max()) + 1):
        mask = bin_indices == bin_index
        bin_targets = targets[mask]
        bin_predictions = predictions[mask]
        errors = bin_predictions - bin_targets
        low = bin_index * BIN_WIDTH
        high = low + BIN_WIDTH
        results.append({
            "range": f"{low:.1f}-{high:.1f}",
            "low": low,
            "high": high,
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(errors))),
            "mse": float(np.mean(errors**2)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "pearson_r": (
                float(pearsonr(bin_targets, bin_predictions).statistic)
                if np.std(bin_targets) > 0 and np.std(bin_predictions) > 0
                else float("nan")
            ),
            "accuracy_0_5": float(np.mean(np.abs(errors) <= 0.5) * 100),
            "accuracy_1_0": float(np.mean(np.abs(errors) <= 1.0) * 100),
            "accuracy_1_5": float(np.mean(np.abs(errors) <= 1.5) * 100),
            "same_bin_accuracy": float(
                np.mean(
                    np.floor(bin_predictions / BIN_WIDTH) == bin_indices[mask]
                ) * 100
            ),
        })
    return results


def write_report(checkpoint, device_name, results):
    os.makedirs(REPORT_DIR, exist_ok=True)
    lines = [
        "DTI-GNN V4 COLD-START EVALUATION REPORT",
        "=" * 90,
        "",
        f"Checkpoint: {os.path.basename(CHECKPOINT_PATH)}",
        f"Checkpoint epoch: {checkpoint['epoch']}",
        f"Best validation MAE: {checkpoint['best_validation_mae']:.6f}",
        f"Model: {ESM_MODEL_NAME}",
        f"Device: {device_name}",
    ]
    if torch.cuda.is_available() and device_name.startswith("cuda"):
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    lines.extend([
        f"Batch size: {BATCH_SIZE}",
        f"Bin width: {BIN_WIDTH}",
        f"Cold-start data source: {EVALUATION_DIR}",
        "",
        "=" * 90,
        "GLOBAL RESULTS",
        "=" * 90,
        "",
    ])

    for split_name in ("COLD-DRUG", "COLD-PAIR", "COLD-TARGET"):
        result = results[split_name]
        metrics = result["metrics"]
        lines.extend([
            split_name,
            "-" * 60,
            f"Samples       : {metrics['samples']}",
            f"RMSE          : {metrics['rmse']:.4f}",
            f"MSE           : {metrics['mse']:.4f}",
            f"MAE           : {metrics['mae']:.4f}",
            f"Pearson r     : {metrics['pearson_r']:.4f}",
            f"Concordance index: {metrics['ci']:.4f}",
            f"Acc @ ±0.5    : {metrics['accuracy_0_5']:.2f}%",
            f"Acc @ ±1.0    : {metrics['accuracy_1_0']:.2f}%",
            f"Acc @ ±1.5    : {metrics['accuracy_1_5']:.2f}%",
        ])
        evaluation_minutes = metrics.get("evaluation_minutes")
        if evaluation_minutes is not None:
            lines.append(f"Evaluation time: {evaluation_minutes:.2f} min")
        lines.append("")

    lines.extend([
        "=" * 90,
        "PER-pIC50-BIN RESULTS",
        "=" * 90,
        "",
    ])
    header = (
        f"{'Range':<13}{'N':>8}{'MAE':>10}{'MSE':>10}{'RMSE':>10}"
        f"{'Pearson r':>12}{'Acc±0.5':>12}{'Acc±1.0':>12}"
        f"{'Acc±1.5':>12}{'Same Bin':>12}"
    )
    for split_name in ("COLD-DRUG", "COLD-PAIR", "COLD-TARGET"):
        lines.extend([
            split_name,
            "-" * len(header),
            header,
            "-" * len(header),
        ])
        for row in results[split_name]["bins"]:
            lines.append(
                f"{row['range']:<13}{row['n']:>8}"
                f"{row['mae']:>10.3f}{row['mse']:>10.3f}"
                f"{row['rmse']:>10.3f}{row['pearson_r']:>12.4f}"
                f"{row['accuracy_0_5']:>11.2f}%"
                f"{row['accuracy_1_0']:>11.2f}%"
                f"{row['accuracy_1_5']:>11.2f}%"
                f"{row['same_bin_accuracy']:>11.2f}%"
            )
        lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as destination:
        destination.write("\n".join(lines) + "\n")


def write_summary(results):
    fields = [
        "split",
        "samples",
        "mae",
        "mse",
        "rmse",
        "pearson_r",
        "ci",
        "evaluation_minutes",
        "accuracy_0_5",
        "accuracy_1_0",
        "accuracy_1_5",
    ]
    summary_path = os.path.join(OUTPUT_DIR, "v4_summary.csv")

    def write_csv(path):
        with open(path, "w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for split_name, result in results.items():
                writer.writerow({
                    "split": split_name,
                    **result["metrics"],
                })

    try:
        write_csv(summary_path)
        return summary_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = os.path.join(
            OUTPUT_DIR,
            f"v4_summary_{timestamp}.csv",
        )
        write_csv(fallback_path)
        print(
            f"Could not overwrite {summary_path}; it may be open in another "
            f"program. Saved this run's summary to {fallback_path} instead."
        )
        return fallback_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(DATA_DIR, "target_sequences.pkl"), "rb") as source:
        target_sequences = pickle.load(source)
    with open(os.path.join(DATA_DIR, "target_to_idx.pkl"), "rb") as source:
        target_to_idx = pickle.load(source)

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    if (
        cache.get("model_name") != ESM_MODEL_NAME
        or cache.get("pooling") != "cls_and_mean"
        or cache.get("sequence_fingerprint") != sequence_fingerprint(target_sequences)
    ):
        raise ValueError("Cached target embeddings do not match this model's targets.")
    embeddings = cache["embeddings"].float()
    if embeddings.shape != (len(target_sequences), 1280):
        raise ValueError(f"Unexpected cached embedding shape: {tuple(embeddings.shape)}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    if (
        checkpoint.get("architecture_version") != ARCHITECTURE_VERSION
        or checkpoint.get("model_name") != ESM_MODEL_NAME
    ):
        raise ValueError(f"Incompatible checkpoint: {CHECKPOINT_PATH}")

    model = DTIModelV4(
        drug_in_channels=18,
        target_embeddings=embeddings,
        hidden_dim=256,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = {}
    prediction_arrays = {}
    for split_name in ("cold_drug", "cold_target", "cold_pair"):
        dataset_path = os.path.join(EVALUATION_DIR, f"{split_name}_data_v4.pt")
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Evaluation split not found: {dataset_path}")
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
        if not dataset:
            raise ValueError(f"Evaluation split is empty: {dataset_path}")
        if not hasattr(dataset[0], "edge_attr"):
            raise ValueError(
                f"Evaluation graphs in {dataset_path} are missing edge_attr. "
                f"Rebuild the v4 graph data with {BOND_FEATURE_DIM}-dimensional "
                "directed bond features."
            )
        predictions = []
        targets = []
        evaluation_start = time.time()
        with torch.inference_mode():
            for start in range(0, len(dataset), BATCH_SIZE):
                chunk = dataset[start : start + BATCH_SIZE]
                batch = next(iter(DataLoader(
                    chunk,
                    batch_size=len(chunk),
                    shuffle=False,
                ))).to(device)
                prediction = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                    batch.target_idx,
                )
                prediction = (
                    prediction.float() * checkpoint["label_std"]
                    + checkpoint["label_mean"]
                )
                predictions.append(prediction.cpu().numpy())
                targets.append(batch.y.float().cpu().numpy())

        y_pred = np.concatenate(predictions)
        y_true = np.concatenate(targets)
        validate_prediction_scale(y_true, f"{split_name} targets")
        validate_prediction_scale(y_pred, f"{split_name} predictions")
        split_metrics = score(y_true, y_pred)
        split_metrics["evaluation_minutes"] = (time.time() - evaluation_start) / 60
        split_label = split_name.upper().replace("_", "-")
        results[split_label] = {
            "metrics": split_metrics,
            "bins": score_by_pic50_bin(y_true, y_pred),
        }
        prediction_arrays[f"{split_name}_predictions"] = y_pred
        prediction_arrays[f"{split_name}_targets"] = y_true
        print(
            f"{split_name.upper()}: n={split_metrics['samples']:,} "
            f"MAE={split_metrics['mae']:.4f} "
            f"MSE={split_metrics['mse']:.4f} "
            f"RMSE={split_metrics['rmse']:.4f} "
            f"Pearson r={split_metrics['pearson_r']:.4f} "
            f"Acc@0.5/1.0/1.5="
            f"{split_metrics['accuracy_0_5']:.2f}%/"
            f"{split_metrics['accuracy_1_0']:.2f}%/"
            f"{split_metrics['accuracy_1_5']:.2f}%"
        )

    write_report(
        checkpoint,
        str(device),
        results,
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "v4_predictions.npz"),
        **prediction_arrays,
    )
    summary_path = write_summary(results)
    print(
        f"Selected checkpoint epoch {checkpoint['epoch']} "
        f"(validation MAE {checkpoint['best_validation_mae']:.4f}); "
        f"target mapping has {len(target_to_idx):,} entries."
    )
    print(f"Saved results to {OUTPUT_DIR}")
    print(f"Saved v3-schema evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
