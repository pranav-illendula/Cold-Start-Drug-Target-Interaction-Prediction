"""Write a v4-style text report from Section 4 ensemble evaluation outputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .report_metrics_v5 import BIN_WIDTH, score, score_by_pic50_bin
except ImportError:
    from report_metrics_v5 import BIN_WIDTH, score, score_by_pic50_bin


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_DIR / "results" / "version5" / "evaluation_report_section4.txt"
SPLITS = ("cold_drug", "cold_pair", "cold_target")


def _read_predictions(path):
    required = {
        "pIC50_true",
        "ensemble_mean",
        "interval_lower",
        "interval_upper",
    }
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Prediction file is missing required columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    try:
        targets = np.asarray([float(row["pIC50_true"]) for row in rows])
        predictions = np.asarray([float(row["ensemble_mean"]) for row in rows])
        lower = np.asarray([float(row["interval_lower"]) for row in rows])
        upper = np.asarray([float(row["interval_upper"]) for row in rows])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Prediction file contains invalid numeric values: {path}") from error
    if not all(np.isfinite(values).all() for values in (targets, predictions, lower, upper)):
        raise ValueError(f"Prediction file contains non-finite values: {path}")
    if (lower > upper).any():
        raise ValueError(f"Prediction intervals have inverted bounds: {path}")
    return targets, predictions, lower, upper


def _format_metric(value, precision=4):
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{precision}f}"


def build_report(evaluation_dir, checkpoint_dir, device="unknown"):
    evaluation_dir = Path(evaluation_dir)
    checkpoint_dir = Path(checkpoint_dir)
    with (evaluation_dir / "summary.json").open(encoding="utf-8") as source:
        summary = json.load(source)
    with (checkpoint_dir / "training_protocol.json").open(encoding="utf-8") as source:
        protocol = json.load(source)

    seeds = summary.get("ensemble_seeds")
    if (
        summary.get("experiment") != "section4_enhancements_v1"
        or not isinstance(seeds, list)
        or protocol.get("experiment") != "section4_enhancements_v1"
        or protocol.get("seeds") != seeds
    ):
        raise ValueError("Evaluation summary and training protocol do not match.")

    validation_mae = {
        int(item["seed"]): item["best_validation_mae"]
        for item in protocol.get("results", [])
    }
    if any(int(seed) not in validation_mae for seed in seeds):
        raise ValueError("Training protocol is missing a seed validation result.")

    lines = [
        "DTI-GNN SECTION 4 ENHANCEMENT COLD-START EVALUATION REPORT",
        "=" * 90,
        "",
        "Architecture: Sliding-window ESM-2, GINE, atom-to-residue cross-attention",
        f"Ensemble seeds: {', '.join(str(seed) for seed in seeds)}",
        f"Device: {device}",
        f"Evaluation outputs: {evaluation_dir}",
        f"Checkpoint directory: {checkpoint_dir}",
        f"Target-group conformal coverage: {summary['coverage']:.1%}",
        f"Calibration target groups: {summary['calibration_target_count']}",
        f"Conformal scale quantile: {summary['conformal_scale_quantile']:.6f}",
        "",
        "BEST VALIDATION MAE BY SEED",
        "-" * 60,
    ]
    for seed in seeds:
        lines.append(f"Seed {seed}: {float(validation_mae[int(seed)]):.6f}")

    lines.extend(["", "=" * 90, "GLOBAL RESULTS", "=" * 90, ""])
    split_results = summary.get("splits", {})
    prediction_data = {}
    for split in SPLITS:
        if split not in split_results:
            raise ValueError(f"Evaluation summary is missing split: {split}")
        targets, predictions, lower, upper = _read_predictions(
            evaluation_dir / f"{split}_predictions.csv"
        )
        stored_metrics = split_results[split]
        if len(targets) != int(stored_metrics["samples"]):
            raise ValueError(f"Prediction count disagrees with summary for {split}.")
        metrics = score(targets, predictions)
        covered = (targets >= lower) & (targets <= upper)
        prediction_data[split] = (targets, predictions, lower, upper, covered)
        label = split.replace("_", "-").upper()
        lines.extend([
            label,
            "-" * 60,
            f"Samples       : {metrics['samples']}",
            f"RMSE          : {_format_metric(metrics['rmse'])}",
            f"MSE           : {_format_metric(metrics['mse'])}",
            f"MAE           : {_format_metric(metrics['mae'])}",
            f"Pearson r     : {_format_metric(metrics['pearson_r'])}",
            f"Concordance index: {_format_metric(metrics['ci'])}",
            f"Acc @ ±0.5    : {metrics['accuracy_0_5']:.2f}%",
            f"Acc @ ±1.0    : {metrics['accuracy_1_0']:.2f}%",
            f"Acc @ ±1.5    : {metrics['accuracy_1_5']:.2f}%",
            f"Row interval coverage: {np.mean(covered):.2%}",
            f"Target-group interval coverage: {stored_metrics['target_group_interval_coverage']:.2%}",
            f"Mean interval width: {np.mean(upper - lower):.4f}",
            "",
        ])

    lines.extend(["=" * 90, "PER-pIC50-BIN RESULTS", "=" * 90, ""])
    header = (
        f"{'Range':<13}{'N':>8}{'MAE':>10}{'MSE':>10}{'RMSE':>10}"
        f"{'Pearson r':>12}{'Acc±0.5':>12}{'Acc±1.0':>12}"
        f"{'Acc±1.5':>12}{'Same Bin':>12}{'PI Cover':>12}{'PI Width':>12}"
    )
    for split in SPLITS:
        targets, predictions, lower, upper, covered = prediction_data[split]
        bins = score_by_pic50_bin(targets, predictions)
        bin_indices = np.floor(targets / BIN_WIDTH).astype(np.int64)
        lines.extend([
            split.replace("_", "-").upper(),
            "-" * len(header),
            header,
            "-" * len(header),
        ])
        for row in bins:
            mask = bin_indices == int(np.floor(row["low"] / BIN_WIDTH))
            metrics = {**row}
            lines.append(
                f"{row['range']:<13}{row['n']:>8}"
                f"{row['mae']:>10.3f}{row['mse']:>10.3f}"
                f"{row['rmse']:>10.3f}{_format_metric(row['pearson_r']):>12}"
                f"{row['accuracy_0_5']:>11.2f}%"
                f"{row['accuracy_1_0']:>11.2f}%"
                f"{row['accuracy_1_5']:>11.2f}%"
                f"{row['same_bin_accuracy']:>11.2f}%"
                f"{np.mean(covered[mask]):>11.2%}"
                f"{np.mean(upper[mask] - lower[mask]):>12.4f}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--device", default="unknown")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(args.evaluation_dir, args.checkpoint_dir, args.device)
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved Section 4 evaluation report: {output_path}")


if __name__ == "__main__":
    main()