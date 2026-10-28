"""Calibrate ensemble uncertainty and evaluate Section 4 on cold-start splits."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from torch_geometric.data import Data


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_enhancements_v1 import (  # noqa: E402
    DTIEnhancementModel,
    TargetEmbeddingStore,
    collate_graphs,
    target_group_conformal_quantile,
)


VERSION_DIR = PROJECT_DIR / "data" / "recovered_baseline_v1"
GRAPH_DIR = VERSION_DIR / "graphs"
CHECKPOINT_DIR = VERSION_DIR / "ensemble"
EMBEDDING_DIR = VERSION_DIR / "esm2"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "evaluation" / "section4_v1"
SEEDS = (42, 43, 44, 45, 46)
COVERAGE = 0.90


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-graphs", default=str(GRAPH_DIR / "calibration_data_v4.pt"))
    parser.add_argument("--cold-drug-graphs", default=str(GRAPH_DIR / "cold_drug_data_v4.pt"))
    parser.add_argument("--cold-target-graphs", default=str(GRAPH_DIR / "cold_target_data_v4.pt"))
    parser.add_argument("--cold-pair-graphs", default=str(GRAPH_DIR / "cold_pair_data_v4.pt"))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--embedding-dir", default=str(EMBEDDING_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--coverage", type=float, default=COVERAGE)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    return parser.parse_args(argv)


def _load_graphs(path):
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"Expected nonempty graph list: {path}")
    for index, graph in enumerate(graphs):
        if not isinstance(graph, Data):
            raise ValueError(f"Invalid graph record {index} in {path}")
        if not hasattr(graph, "target_idx") or graph.target_idx.numel() != 1:
            raise ValueError(f"Graph {index} has no scalar target index in {path}")
        if not hasattr(graph, "edge_attr") or graph.edge_attr.shape[-1] != 14:
            raise ValueError(f"Graph {index} lacks the v4.2 bond feature matrix in {path}")
    return graphs


def _validate_mapping_alignment(graph_path, embedding_dir):
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
    graph_hash = graph_manifest.get("target_mapping_sha256")
    embedding_hash = embedding_manifest.get("target_mapping_sha256")
    if not graph_hash or graph_hash != embedding_hash:
        raise ValueError("Evaluation graphs and ESM features use different target mappings.")
    return graph_hash


def _load_models(checkpoint_dir, device, target_mapping_sha256, seeds):
    models = []
    for seed in seeds:
        path = Path(checkpoint_dir) / f"seed_{seed}_best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Ensemble checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("experiment") != "section4_enhancements_v1":
            raise ValueError(f"Incompatible enhancement checkpoint: {path}")
        if checkpoint.get("seed") != seed:
            raise ValueError(f"Seed metadata mismatch in enhancement checkpoint: {path}")
        if checkpoint.get("target_mapping_sha256") != target_mapping_sha256:
            raise ValueError(f"Target mapping mismatch in enhancement checkpoint: {path}")
        model = DTIEnhancementModel()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        models.append((model, float(checkpoint["label_mean"]), float(checkpoint["label_std"])))
    return models


def _validate_calibration_split(checkpoint_dir, mapping_hash, seeds, calibration_graphs):
    protocol_path = Path(checkpoint_dir) / "training_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Training protocol missing: {protocol_path}")
    with protocol_path.open(encoding="utf-8") as source:
        protocol = json.load(source)
    if (
        protocol.get("experiment") != "section4_enhancements_v1"
        or protocol.get("target_mapping_sha256") != mapping_hash
        or protocol.get("seeds") != list(seeds)
    ):
        raise ValueError("Training protocol does not match the requested model ensemble.")
    fit_target_values = protocol.get("train_target_indices")
    validation_target_values = protocol.get("validation_target_indices")
    if not isinstance(fit_target_values, list) or not isinstance(
        validation_target_values,
        list,
    ):
        raise ValueError("Training protocol is missing target-held-out split indices.")
    fit_targets = set(fit_target_values)
    validation_targets = set(validation_target_values)
    if fit_targets & validation_targets:
        raise ValueError("Training and validation targets overlap in the training protocol.")
    calibration_targets = {
        int(graph.target_idx.item()) for graph in calibration_graphs
    }
    if calibration_targets & (fit_targets | validation_targets):
        raise ValueError("Calibration targets overlap model training or validation targets.")


def _predict(graphs, models, embedding_store, device, batch_size):
    means = []
    standard_deviations = []
    targets = []
    target_indices = []
    with torch.inference_mode():
        for start in range(0, len(graphs), batch_size):
            items = graphs[start : start + batch_size]
            batch, windows, window_mask, residues, residue_mask = collate_graphs(
                items,
                embedding_store,
            )
            batch = batch.to(device)
            windows = windows.to(device)
            window_mask = window_mask.to(device)
            residues = residues.to(device)
            residue_mask = residue_mask.to(device)
            member_predictions = []
            for model, label_mean, label_std in models:
                prediction = model(
                    batch,
                    windows,
                    window_mask,
                    residues,
                    residue_mask,
                )
                member_predictions.append(
                    prediction.float() * label_std + label_mean
                )
            predictions = torch.stack(member_predictions)
            means.append(predictions.mean(dim=0).cpu().numpy())
            standard_deviations.append(
                predictions.std(dim=0, unbiased=False).cpu().numpy()
            )
            targets.append(batch.y.reshape(-1).cpu().numpy())
            target_indices.append(batch.target_idx.reshape(-1).cpu().numpy())
    return (
        np.concatenate(targets),
        np.concatenate(means),
        np.concatenate(standard_deviations),
        np.concatenate(target_indices),
    )


def _metrics(targets, means, lower, upper, target_indices):
    errors = means - targets
    if not len(targets):
        raise ValueError("Cannot evaluate an empty dataset.")
    covered_rows = (targets >= lower) & (targets <= upper)
    group_coverage = [
        bool(covered_rows[target_indices == target_id].all())
        for target_id in np.unique(target_indices)
    ]
    return {
        "samples": int(len(targets)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "pearson_r": (
            float(pearsonr(targets, means).statistic)
            if np.std(targets) > 0 and np.std(means) > 0
            else None
        ),
        "target_groups": int(len(group_coverage)),
        "row_interval_coverage": float(np.mean(covered_rows)),
        "target_group_interval_coverage": float(np.mean(group_coverage)),
        "mean_interval_width": float(np.mean(upper - lower)),
    }


def _write_predictions(path, targets, means, deviations, lower, upper, target_indices):
    with path.open("x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "target_idx",
                "pIC50_true",
                "ensemble_mean",
                "ensemble_std",
                "interval_lower",
                "interval_upper",
                "inside_interval",
            ]
        )
        writer.writerows(
            (
                int(target_idx),
                float(target),
                float(mean),
                float(std),
                float(lo),
                float(hi),
                bool(lo <= target <= hi),
            )
            for target_idx, target, mean, std, lo, hi in zip(
                target_indices,
                targets,
                means,
                deviations,
                lower,
                upper,
            )
        )


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not 0 < args.coverage < 1:
        raise ValueError("--coverage must be between zero and one.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique comma-separated integers.")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    mapping_hash = _validate_mapping_alignment(
        args.calibration_graphs,
        args.embedding_dir,
    )
    calibration_graphs = _load_graphs(args.calibration_graphs)
    _validate_calibration_split(
        args.checkpoint_dir,
        mapping_hash,
        seeds,
        calibration_graphs,
    )
    models = _load_models(args.checkpoint_dir, device, mapping_hash, seeds)
    store = TargetEmbeddingStore(args.embedding_dir)
    calibration_targets, calibration_means, calibration_stds, calibration_indices = (
        _predict(
            calibration_graphs,
            models,
            store,
            device,
            args.batch_size,
        )
    )
    quantile = target_group_conformal_quantile(
        calibration_targets,
        calibration_means,
        calibration_stds,
        calibration_indices,
        coverage=args.coverage,
    )
    calibration_target_count = int(np.unique(calibration_indices).size)
    print(
        f"Calibration targets={calibration_target_count} "
        f"coverage={args.coverage:.3f} scale_quantile={quantile:.6f}",
        flush=True,
    )
    del calibration_graphs

    results = {
        "experiment": "section4_enhancements_v1",
        "ensemble_seeds": list(seeds),
        "coverage": args.coverage,
        "calibration_unit": "target group; score is maximum normalized absolute residual per target",
        "calibration_target_count": calibration_target_count,
        "conformal_scale_quantile": quantile,
        "target_mapping_sha256": mapping_hash,
        "splits": {},
    }
    split_paths = {
        "cold_drug": args.cold_drug_graphs,
        "cold_target": args.cold_target_graphs,
        "cold_pair": args.cold_pair_graphs,
    }
    for split_name, split_path in split_paths.items():
        graphs = _load_graphs(split_path)
        targets, means, deviations, target_indices = _predict(
            graphs,
            models,
            store,
            device,
            args.batch_size,
        )
        radius = quantile * np.maximum(deviations, 1e-6)
        lower = means - radius
        upper = means + radius
        metrics = _metrics(
            targets,
            means,
            lower,
            upper,
            target_indices,
        )
        results["splits"][split_name] = metrics
        _write_predictions(
            output_dir / f"{split_name}_predictions.csv",
            targets,
            means,
            deviations,
            lower,
            upper,
            target_indices,
        )
        pearson = (
            f"{metrics['pearson_r']:.4f}"
            if metrics["pearson_r"] is not None
            else "n/a"
        )
        print(
            f"{split_name}: n={metrics['samples']:,} MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} Pearson={pearson} "
            f"row_coverage={metrics['row_interval_coverage']:.3f} "
            f"target_group_coverage={metrics['target_group_interval_coverage']:.3f}",
            flush=True,
        )
        del graphs

    report_path = output_dir / "summary.json"
    with report_path.open("x", encoding="utf-8") as destination:
        json.dump(results, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(f"Saved calibrated ensemble results under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
