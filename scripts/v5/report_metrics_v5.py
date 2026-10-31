"""Metrics used to write the standalone version 5 evaluation report."""

import numpy as np
from scipy.stats import pearsonr


BIN_WIDTH = 1.5


def concordance_index(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("Targets and predictions must be equal-length 1D arrays.")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Targets and predictions must contain only finite values.")
    if len(targets) < 2:
        return float("nan")

    prediction_ranks = np.searchsorted(np.unique(predictions), predictions)
    order = np.argsort(targets, kind="mergesort")
    tree = np.zeros(len(np.unique(predictions)) + 1, dtype=np.int64)

    def count_at_or_below(rank):
        total = 0
        index = rank + 1
        while index:
            total += tree[index]
            index -= index & -index
        return total

    def add_rank(rank):
        index = rank + 1
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    concordant = 0.0
    comparable_pairs = 0
    prior_count = 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and targets[order[end]] == targets[order[start]]:
            end += 1

        group_ranks = prediction_ranks[order[start:end]]
        for prediction_rank in group_ranks:
            less_or_equal = count_at_or_below(prediction_rank)
            less = (
                count_at_or_below(prediction_rank - 1)
                if prediction_rank > 0
                else 0
            )
            concordant += less + 0.5 * (less_or_equal - less)
        comparable_pairs += prior_count * (end - start)
        for prediction_rank in group_ranks:
            add_rank(prediction_rank)
        prior_count += end - start
        start = end

    if comparable_pairs == 0:
        return float("nan")
    return float(concordant / comparable_pairs)


def score(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("Targets and predictions must be equal-length 1D arrays.")
    if not len(targets):
        raise ValueError("Cannot score an empty split.")
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
        results.append(
            {
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
                    )
                    * 100
                ),
            }
        )
    return results