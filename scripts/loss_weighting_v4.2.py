import numpy as np


PIC50_BIN_EDGES = (4.5, 6.0, 7.5, 9.0, 10.5, 12.0)
MAX_PIC50_BIN_WEIGHT = 5.0
PIC50_BIN_COUNT = len(PIC50_BIN_EDGES) + 1


def pic50_bin_indices(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("pIC50 values must be a finite 1D array.")
    return np.searchsorted(PIC50_BIN_EDGES, values, side="right")


def build_pic50_bin_weights(training_labels):
    """Build capped inverse-square-root frequency weights from training labels."""
    bin_indices = pic50_bin_indices(training_labels)
    if not len(bin_indices):
        raise ValueError("Cannot build pIC50 weights from an empty training set.")

    counts = np.bincount(bin_indices, minlength=PIC50_BIN_COUNT)
    max_count = int(counts.max())
    raw_weights = np.ones(PIC50_BIN_COUNT, dtype=np.float64)
    populated = counts > 0
    raw_weights[populated] = np.minimum(
        np.sqrt(max_count / counts[populated]),
        MAX_PIC50_BIN_WEIGHT,
    )
    mean_weight = float(np.dot(raw_weights, counts) / len(bin_indices))
    weights = raw_weights / mean_weight
    return weights.astype(np.float32), counts
