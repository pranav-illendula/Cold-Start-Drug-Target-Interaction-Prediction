"""Train the five-seed Section 4 model on isolated recovered_baseline_v1 data."""

import argparse
import gc
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_enhancements_v1 import (  # noqa: E402
    DTIEnhancementModel,
    TargetEmbeddingStore,
    build_pic50_bin_weights,
    collate_graphs,
    hybrid_ranking_regression_loss,
)


VERSION_DIR = PROJECT_DIR / "data" / "recovered_baseline_v1"
GRAPH_DIR = VERSION_DIR / "graphs"
ESM_DIR = VERSION_DIR / "esm2"
DEFAULT_OUTPUT_DIR = VERSION_DIR / "ensemble"
SEEDS = (42, 43, 44, 45, 46)
VALIDATION_SEED = 31415
VALIDATION_FRACTION = 0.15


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-graphs", default=str(GRAPH_DIR / "train_fit_data_v4.pt"))
    parser.add_argument("--embedding-dir", default=str(ESM_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_graphs(path):
    # These are user-generated PyG files; only load files in the selected version directory.
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"Expected a nonempty PyG graph list in {path}")
    return graphs


def _validate_artifact_alignment(graph_path, embedding_dir):
    graph_manifest_path = Path(graph_path).parent / "manifest.json"
    embedding_manifest_path = Path(embedding_dir) / "manifest.json"
    if not graph_manifest_path.is_file():
        raise FileNotFoundError(f"Graph manifest missing: {graph_manifest_path}")
    if not embedding_manifest_path.is_file():
        raise FileNotFoundError(f"ESM manifest missing: {embedding_manifest_path}")
    with graph_manifest_path.open(encoding="utf-8") as source:
        graph_manifest = json.load(source)
    with embedding_manifest_path.open(encoding="utf-8") as source:
        embedding_manifest = json.load(source)
    graph_mapping_hash = graph_manifest.get("target_mapping_sha256")
    embedding_mapping_hash = embedding_manifest.get("target_mapping_sha256")
    if not graph_mapping_hash or graph_mapping_hash != embedding_mapping_hash:
        raise ValueError(
            "Graph and ESM artifacts were not generated from the same target mapping."
        )
    return graph_mapping_hash


def _validate_graphs(graphs):
    target_ids = set()
    labels = []
    for index, graph in enumerate(graphs):
        if not hasattr(graph, "edge_attr") or graph.edge_attr.shape[1] != 14:
            raise ValueError(f"Graph {index} is missing 14-dimensional edge_attr.")
        if not hasattr(graph, "target_idx") or graph.target_idx.numel() != 1:
            raise ValueError(f"Graph {index} has no scalar target_idx.")
        target_ids.add(int(graph.target_idx.item()))
        label = float(graph.y.reshape(-1)[0])
        if not math.isfinite(label):
            raise ValueError(f"Graph {index} has a non-finite pIC50 label.")
        labels.append(label)
    return target_ids, np.asarray(labels, dtype=np.float64)


def split_target_validation(graphs, validation_fraction=VALIDATION_FRACTION):
    target_ids, _ = _validate_graphs(graphs)
    ordered = np.asarray(sorted(target_ids), dtype=np.int64)
    if len(ordered) < 2:
        raise ValueError("At least two targets are needed for target-held-out validation.")
    rng = np.random.RandomState(VALIDATION_SEED)
    rng.shuffle(ordered)
    count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    validation_targets = set(ordered[:count].tolist())
    train_targets = target_ids - validation_targets
    train_indices = [
        index for index, graph in enumerate(graphs)
        if int(graph.target_idx.item()) in train_targets
    ]
    validation_indices = [
        index for index, graph in enumerate(graphs)
        if int(graph.target_idx.item()) in validation_targets
    ]
    if not train_indices or not validation_indices:
        raise ValueError("Target validation split has an empty partition.")
    return train_indices, validation_indices, train_targets, validation_targets


def _predict(model, loader, embedding_store, device, label_mean, label_std):
    model.eval()
    predictions = []
    targets = []
    with torch.inference_mode():
        for graphs in loader:
            batch, windows, window_mask, residues, residue_mask = collate_graphs(
                graphs,
                embedding_store,
            )
            prediction = model(
                batch.to(device),
                windows.to(device),
                window_mask.to(device),
                residues.to(device),
                residue_mask.to(device),
            )
            predictions.append(
                (prediction.float() * label_std + label_mean).cpu().numpy()
            )
            targets.append(batch.y.reshape(-1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def _train_seed(
    graphs,
    train_indices,
    validation_indices,
    labels,
    embedding_store,
    args,
    seed,
    output_dir,
    device,
    target_mapping_sha256,
):
    _set_seed(seed)
    train_labels = labels[train_indices]
    label_mean = float(train_labels.mean())
    label_std = float(train_labels.std())
    if label_std == 0:
        raise ValueError("Training labels have zero variance.")
    bin_weights, bin_counts = build_pic50_bin_weights(train_labels)

    model = DTIEnhancementModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_mae = float("inf")
    stale_epochs = 0
    best_path = output_dir / f"seed_{seed}_best.pt"
    if best_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {best_path}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        order = np.random.permutation(train_indices)
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            graph_items = [graphs[int(index)] for index in indices]
            batch, windows, window_mask, residues, residue_mask = collate_graphs(
                graph_items,
                embedding_store,
            )
            batch = batch.to(device)
            windows = windows.to(device)
            window_mask = window_mask.to(device)
            residues = residues.to(device)
            residue_mask = residue_mask.to(device)
            targets = batch.y.reshape(-1).float()
            optimizer.zero_grad(set_to_none=True)
            normalized_predictions = model(
                batch,
                windows,
                window_mask,
                residues,
                residue_mask,
            )
            predictions = normalized_predictions * label_std + label_mean
            loss, _ = hybrid_ranking_regression_loss(
                predictions,
                targets,
                bin_weights,
                ranking_weight=args.ranking_weight,
                ranking_margin=args.ranking_margin,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(graphs, validation_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=list,
        )
        validation_predictions, validation_targets = _predict(
            model,
            validation_loader,
            embedding_store,
            device,
            label_mean,
            label_std,
        )
        del validation_loader
        mae = float(np.mean(np.abs(validation_predictions - validation_targets)))
        mean_loss = float(np.mean(losses))
        print(
            f"seed={seed} epoch={epoch} loss={mean_loss:.5f} "
            f"validation_mae={mae:.5f}",
            flush=True,
        )
        if mae < best_mae:
            best_mae = mae
            stale_epochs = 0
            torch.save(
                {
                    "experiment": "section4_enhancements_v1",
                    "seed": seed,
                    "epoch": epoch,
                    "best_validation_mae": best_mae,
                    "label_mean": label_mean,
                    "label_std": label_std,
                    "ranking_weight": args.ranking_weight,
                    "ranking_margin": args.ranking_margin,
                    "target_mapping_sha256": target_mapping_sha256,
                    "bin_counts": bin_counts.tolist(),
                    "model_state_dict": model.state_dict(),
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_path, best_mae


def main(argv=None):
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive.")
    if (
        args.learning_rate <= 0
        or args.weight_decay < 0
        or args.ranking_weight < 0
        or args.ranking_margin < 0
    ):
        raise ValueError(
            "learning-rate must be positive; weight-decay, ranking-weight, "
            "and ranking-margin must be nonnegative."
        )
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique comma-separated integers.")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_mapping_sha256 = _validate_artifact_alignment(
        args.train_graphs,
        args.embedding_dir,
    )
    graphs = _load_graphs(args.train_graphs)
    _, labels = _validate_graphs(graphs)
    train_indices, validation_indices, train_targets, validation_targets = (
        split_target_validation(graphs)
    )
    print(
        f"Examples: train={len(train_indices):,} validation={len(validation_indices):,}; "
        f"targets: train={len(train_targets):,} validation={len(validation_targets):,}",
        flush=True,
    )
    embedding_store = TargetEmbeddingStore(args.embedding_dir)
    device = torch.device(args.device)
    results = []
    for seed in seeds:
        checkpoint, validation_mae = _train_seed(
            graphs,
            train_indices,
            validation_indices,
            labels,
            embedding_store,
            args,
            seed,
            output_dir,
            device,
            target_mapping_sha256,
        )
        results.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "best_validation_mae": validation_mae,
            }
        )
    protocol = {
        "experiment": "section4_enhancements_v1",
        "seeds": list(seeds),
        "validation_seed": VALIDATION_SEED,
        "validation_fraction_of_targets": VALIDATION_FRACTION,
        "target_mapping_sha256": target_mapping_sha256,
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "train_target_indices": sorted(train_targets),
        "validation_target_indices": sorted(validation_targets),
        "results": results,
    }
    protocol_path = output_dir / "training_protocol.json"
    with protocol_path.open("x", encoding="utf-8") as destination:
        json.dump(protocol, destination, indent=2)
        destination.write("\n")
    print(f"Saved ensemble training protocol: {protocol_path}", flush=True)


if __name__ == "__main__":
    main()
