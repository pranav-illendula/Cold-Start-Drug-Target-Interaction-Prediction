import numpy as np


def concordance_index(targets, predictions):
    """Compute Harrell's concordance index in O(n log n) time."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("Targets and predictions must be equal-length 1D arrays.")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Targets and predictions must contain only finite values.")
    if len(targets) < 2:
        return float("nan")

    prediction_ranks = np.searchsorted(
        np.unique(predictions),
        predictions,
    )
    order = np.argsort(targets, kind="mergesort")
    tree = np.zeros(len(np.unique(predictions)) + 1, dtype=np.int64)

    def count_at_or_below(rank):
        total = 0
        index = rank + 1
        while index:
            total += tree[index]
            index -= index & -index
        return total

    def add(rank):
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
        for rank in group_ranks:
            less_or_equal = count_at_or_below(rank)
            less = count_at_or_below(rank - 1) if rank > 0 else 0
            tied = less_or_equal - less
            concordant += less + 0.5 * tied
        comparable_pairs += prior_count * (end - start)
        for rank in group_ranks:
            add(rank)
        prior_count += end - start
        start = end

    if comparable_pairs == 0:
        return float("nan")
    return float(concordant / comparable_pairs)
