"""Evaluate the selected frozen-ESM-2 150M model on cold-start splits."""

import csv
import hashlib
import os
import pickle

import numpy as np
import torch
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

from model_v4 import ARCHITECTURE_VERSION, DTIModelV4, ESM_MODEL_NAME


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EVALUATION_DIR = os.path.join(PROJECT_DIR, "evaluation")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(EVALUATION_DIR, "v4_results")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dti_model_v4_best.pt")
CACHE_PATH = os.path.join(DATA_DIR, "esm2_t30_150M_cls_mean_embeddings.pt")
BATCH_SIZE = 64


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
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "pearson_r": (
            float(pearsonr(targets, predictions).statistic)
            if np.std(targets) > 0 and np.std(predictions) > 0
            else float("nan")
        ),
        "accuracy_0_5": float(np.mean(np.abs(errors) <= 0.5) * 100),
        "accuracy_1_0": float(np.mean(np.abs(errors) <= 1.0) * 100),
        "accuracy_1_5": float(np.mean(np.abs(errors) <= 1.5) * 100),
    }


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
        hidden_dim=128,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = {}
    prediction_arrays = {}
    for split_name in ("cold_drug", "cold_target", "cold_pair"):
        dataset_path = os.path.join(EVALUATION_DIR, f"{split_name}_data_v2.pt")
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Evaluation split not found: {dataset_path}")
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
        predictions = []
        targets = []
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
        split_metrics = score(y_true, y_pred)
        results[split_name.upper()] = split_metrics
        prediction_arrays[f"{split_name}_predictions"] = y_pred
        prediction_arrays[f"{split_name}_targets"] = y_true
        print(
            f"{split_name.upper()}: n={split_metrics['samples']:,} "
            f"MAE={split_metrics['mae']:.4f} "
            f"RMSE={split_metrics['rmse']:.4f} "
            f"Pearson r={split_metrics['pearson_r']:.4f} "
            f"Acc@0.5/1.0/1.5="
            f"{split_metrics['accuracy_0_5']:.2f}%/"
            f"{split_metrics['accuracy_1_0']:.2f}%/"
            f"{split_metrics['accuracy_1_5']:.2f}%"
        )

    summary_path = os.path.join(OUTPUT_DIR, "v4_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as destination:
        fields = [
            "split",
            "samples",
            "mae",
            "rmse",
            "pearson_r",
            "accuracy_0_5",
            "accuracy_1_0",
            "accuracy_1_5",
        ]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for split_name, metrics in results.items():
            writer.writerow({"split": split_name, **metrics})
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "v4_predictions.npz"),
        **prediction_arrays,
    )
    print(
        f"Selected checkpoint epoch {checkpoint['epoch']} "
        f"(validation MAE {checkpoint['best_validation_mae']:.4f}); "
        f"target mapping has {len(target_to_idx):,} entries."
    )
    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
