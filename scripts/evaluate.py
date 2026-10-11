# ============================================================
# DTI-GNN V3 - Cold-Start Evaluation
#
# IMPORTANT:
# Cold-start datasets are loaded from:
#     C:\DTI_GNN\data
#
# NOT from:
#     C:\DTI_GNN\evaluation
#
# This version also performs a pIC50 sanity check before
# loading ESM-2 so raw nM affinity values are detected early.
#
# Evaluates:
#   1. Cold Drug
#   2. Cold Pair
#   3. Cold Target
#
# Metrics:
#   - RMSE
#   - MAE
#   - Pearson r
#   - Accuracy @ ±0.5
#   - Accuracy @ ±1.0
#   - Accuracy @ ±1.5
#   - Per-pIC50-bin accuracy
#   - Same-bin prediction accuracy
#
# Output:
# C:\DTI_GNN\evaluation\v3_results
# ============================================================


import os
import pickle
import csv
import time

import numpy as np
import torch

from scipy.stats import pearsonr

from torch_geometric.loader import DataLoader

import esm
from model_v3 import ARCHITECTURE_VERSION, DTIModelV3


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\DTI_GNN"

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

EVAL_DIR = os.path.join(
    PROJECT_DIR,
    "evaluation"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_DIR,
    "checkpoints"
)

OUTPUT_DIR = os.path.join(
    EVAL_DIR,
    "v3_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_NAME = "dti_model_v3_stable_epoch15.pt"

BATCH_SIZE = 64

BIN_WIDTH = 1.5

TOLERANCES = [
    0.5,
    1.0,
    1.5
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# START
# ============================================================

print()
print("=" * 90)
print("DTI-GNN V3 COLD-START EVALUATION")
print("=" * 90)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("DATA directory:")
print(DATA_DIR)

print()
print("Evaluation output directory:")
print(OUTPUT_DIR)

print()
print("Device:")
print(device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# VERIFY IMPORTANT PATHS
# ============================================================

required_directories = [
    DATA_DIR,
    CHECKPOINT_DIR,
    EVAL_DIR
]

for directory in required_directories:

    if not os.path.isdir(directory):

        raise FileNotFoundError(
            f"\nRequired directory not found:\n"
            f"{directory}"
        )


# ============================================================
# DATASET PATHS
# Cold-start splits are stored separately from the training dataset.
# ============================================================

cold_drug_path = os.path.join(
    EVAL_DIR,
    "cold_drug_data_v2.pt"
)

cold_pair_path = os.path.join(
    EVAL_DIR,
    "cold_pair_data_v2.pt"
)

cold_target_path = os.path.join(
    EVAL_DIR,
    "cold_target_data_v2.pt"
)


# ============================================================
# CHECK DATASET FILES
# ============================================================

print("=" * 90)
print("CHECKING COLD-START DATASET FILES")
print("=" * 90)

print()

print(
    "Cold-drug:"
)
print(
    cold_drug_path
)

print()

print(
    "Cold-pair:"
)
print(
    cold_pair_path
)

print()

print(
    "Cold-target:"
)
print(
    cold_target_path
)

print()


for path in [
    cold_drug_path,
    cold_pair_path,
    cold_target_path
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nCold-start dataset not found:\n"
            f"{path}\n\n"
            f"Make sure the cold-start files are inside:\n"
            f"{EVAL_DIR}"
        )


# ============================================================
# LOAD COLD-START DATASETS
#
# DO THIS BEFORE LOADING ESM
# SO WE CAN CHECK THE TARGET VALUES FIRST.
# ============================================================

print("=" * 90)
print("LOADING COLD-START DATASETS")
print("=" * 90)

print()

cold_drug = torch.load(
    cold_drug_path,
    weights_only=False
)

print(
    f"Loaded cold-drug: "
    f"{len(cold_drug):,} samples"
)

cold_pair = torch.load(
    cold_pair_path,
    weights_only=False
)

print(
    f"Loaded cold-pair: "
    f"{len(cold_pair):,} samples"
)

cold_target = torch.load(
    cold_target_path,
    weights_only=False
)

print(
    f"Loaded cold-target: "
    f"{len(cold_target):,} samples"
)

print()


# ============================================================
# LOAD TRAINING DATA FOR REFERENCE RANGE
# ============================================================

train_data_path = os.path.join(
    DATA_DIR,
    "train_data_v2.pt"
)

print("=" * 90)
print("CHECKING pIC50 TARGET DISTRIBUTION")
print("=" * 90)

print()

if not os.path.exists(train_data_path):

    raise FileNotFoundError(
        f"Training dataset not found:\n"
        f"{train_data_path}"
    )


train_data = torch.load(
    train_data_path,
    weights_only=False
)

train_y = np.array([
    float(d.y.item())
    for d in train_data
])

print(
    f"Training target range:"
)

print(
    f"  Min: {train_y.min():.4f}"
)

print(
    f"  Max: {train_y.max():.4f}"
)

print(
    f"  Mean: {train_y.mean():.4f}"
)

print(
    f"  Median: {np.median(train_y):.4f}"
)

print()


# ============================================================
# FUNCTION TO CHECK TARGET VALUES
# ============================================================

def check_target_distribution(
    data_list,
    split_name,
    train_min,
    train_max
):

    ys = np.array([
        float(d.y.item())
        for d in data_list
    ])

    print(
        f"{split_name}"
    )

    print(
        f"  Min    : {ys.min():.4f}"
    )

    print(
        f"  Max    : {ys.max():.4f}"
    )

    print(
        f"  Mean   : {ys.mean():.4f}"
    )

    print(
        f"  Median : {np.median(ys):.4f}"
    )

    print()

    # --------------------------------------------------------
    # Show first 20 values
    # --------------------------------------------------------

    print(
        "  First 20 y values:"
    )

    print(
        " ",
        ys[:20]
    )

    print()

    # --------------------------------------------------------
    # Detect obviously raw affinity values
    #
    # pIC50 values should not be in the thousands.
    # --------------------------------------------------------

    if np.max(np.abs(ys)) > 20:

        raise ValueError(
            "\n"
            + "=" * 90
            + "\n"
            + "DATASET SANITY CHECK FAILED\n"
            + "=" * 90
            + "\n"
            + f"{split_name} contains target values outside "
              f"a reasonable pIC50 scale.\n\n"
            + f"Observed range: "
              f"{ys.min():.4f} to {ys.max():.4f}\n\n"
            + "This strongly suggests the dataset contains "
              "raw affinity values (for example nM) rather "
              "than the pIC50 target used during training.\n\n"
            + "Evaluation has been stopped BEFORE ESM-2 loading.\n"
            + "Check the dataset generation pipeline.\n"
            + "=" * 90
        )

    # --------------------------------------------------------
    # Compare with training distribution
    # --------------------------------------------------------

    outside_train_range = np.sum(
        (ys < train_min) |
        (ys > train_max)
    )

    if outside_train_range > 0:

        percentage = (
            outside_train_range /
            len(ys) *
            100
        )

        print(
            f"  WARNING: "
            f"{outside_train_range:,} samples "
            f"({percentage:.2f}%) are outside "
            f"the training target range."
        )

        print()

    return ys


# ============================================================
# SANITY CHECK ALL THREE DATASETS
# ============================================================

drug_y = check_target_distribution(
    cold_drug,
    "COLD-DRUG",
    train_y.min(),
    train_y.max()
)

pair_y = check_target_distribution(
    cold_pair,
    "COLD-PAIR",
    train_y.min(),
    train_y.max()
)

target_y = check_target_distribution(
    cold_target,
    "COLD-TARGET",
    train_y.min(),
    train_y.max()
)


print("=" * 90)
print("TARGET SANITY CHECK PASSED")
print("=" * 90)

print()
print(
    "All three cold-start datasets contain values "
    "on a reasonable pIC50 scale."
)

print()


# ============================================================
# LOAD TARGET SEQUENCES
# ============================================================

target_sequences_path = os.path.join(
    DATA_DIR,
    "target_sequences.pkl"
)

if not os.path.exists(
    target_sequences_path
):

    raise FileNotFoundError(
        f"Target sequence file not found:\n"
        f"{target_sequences_path}"
    )


print(
    "Loading target sequences..."
)

with open(
    target_sequences_path,
    "rb"
) as f:

    target_sequences = pickle.load(f)

print(
    f"Targets loaded: "
    f"{len(target_sequences):,}"
)

print()


# ============================================================
# LOAD ESM-2 35M
# ============================================================

print("=" * 90)
print("LOADING ESM-2 35M")
print("=" * 90)

print()

esm_model, alphabet = (
    esm.pretrained.esm2_t12_35M_UR50D()
)

batch_converter = (
    alphabet.get_batch_converter()
)


# Same freezing configuration as V3 training
for p in esm_model.parameters():

    p.requires_grad = False


last_layer_idx = (
    len(esm_model.layers) - 1
)


for p in (
    esm_model.layers[last_layer_idx]
    .parameters()
):

    p.requires_grad = True


esm_model = esm_model.to(
    device
)

print(
    "ESM-2 35M loaded."
)

print()


# ============================================================
# CREATE MODEL
# ============================================================

print(
    "Creating V3 model..."
)

model = DTIModelV3(

    drug_in_channels=18,

    esm_model=esm_model,

    batch_converter=batch_converter,

    target_sequences=target_sequences,

    hidden_dim=128

).to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    CHECKPOINT_NAME
)

print()
print(
    "Loading checkpoint:"
)

print(
    checkpoint_path
)

if not os.path.exists(
    checkpoint_path
):

    raise FileNotFoundError(
        f"\nCheckpoint not found:\n"
        f"{checkpoint_path}"
    )


checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
    raise ValueError(
        f"Checkpoint architecture version "
        f"{checkpoint.get('architecture_version')} is incompatible with "
        f"model version {ARCHITECTURE_VERSION}; retrain using train_v3.py."
    )

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


print()

print(
    f"Checkpoint epoch : "
    f"{checkpoint['epoch']}"
)

print(
    f"Training loss    : "
    f"{checkpoint['train_loss']:.6f}"
)

print()

print(
    "Model loaded successfully."
)

print()


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_split(
    model,
    data_list,
    device,
    split_name,
    batch_size=64,
    bin_width=1.5
):

    print()
    print("=" * 110)
    print(
        f"EVALUATING: {split_name}"
    )
    print("=" * 110)

    model.eval()

    predictions = []
    targets = []

    total_samples = len(
        data_list
    )

    start_time = time.time()


    # ========================================================
    # PREDICTION
    # ========================================================

    with torch.no_grad():

        for start in range(
            0,
            total_samples,
            batch_size
        ):

            end = min(
                start + batch_size,
                total_samples
            )

            chunk = data_list[
                start:end
            ]

            loader = DataLoader(
                chunk,
                batch_size=len(chunk),
                shuffle=False
            )

            batch = next(
                iter(loader)
            )

            batch = batch.to(
                device
            )

            pred = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.target_idx,
                device
            )

            if not torch.isfinite(pred).all():
                raise FloatingPointError(
                    f"{split_name} produced non-finite predictions."
                )

            if pred.abs().max().item() > 20:
                raise ValueError(
                    f"{split_name} produced predictions outside the "
                    "expected pIC50 range (absolute value > 20). "
                    "Check the checkpoint and target normalization."
                )

            predictions.append(
                pred.cpu().numpy()
            )

            targets.append(
                batch.y.cpu().numpy()
            )

            processed = end

            percentage = (
                processed /
                total_samples *
                100
            )

            print(
                f"\rProgress: "
                f"{processed:,}/"
                f"{total_samples:,} "
                f"({percentage:.1f}%)",
                end="",
                flush=True
            )


    print()

    elapsed = (
        time.time() -
        start_time
    )


    predictions = np.concatenate(
        predictions
    )

    targets = np.concatenate(
        targets
    )


    # ========================================================
    # ERRORS
    # ========================================================

    errors = np.abs(
        predictions -
        targets
    )


    squared_errors = (
        predictions -
        targets
    ) ** 2


    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    rmse = np.sqrt(
        np.mean(
            squared_errors
        )
    )

    mae = np.mean(
        errors
    )


    if (
        len(np.unique(predictions)) > 1
        and
        len(np.unique(targets)) > 1
    ):

        pearson_r, _ = pearsonr(
            predictions,
            targets
        )

    else:

        pearson_r = np.nan


    tolerance_accuracy = {}


    for tolerance in TOLERANCES:

        tolerance_accuracy[
            tolerance
        ] = (
            errors <= tolerance
        ).mean() * 100


    # ========================================================
    # GLOBAL RESULTS
    # ========================================================

    print()

    print(
        f"Samples         : "
        f"{len(targets):,}"
    )

    print(
        f"Evaluation time : "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"RMSE            : "
        f"{rmse:.4f}"
    )

    print(
        f"MAE             : "
        f"{mae:.4f}"
    )

    print(
        f"Pearson r       : "
        f"{pearson_r:.4f}"
    )

    print()

    print(
        "Global Tolerance Accuracy"
    )

    print("-" * 70)


    for tolerance in TOLERANCES:

        print(
            f"Acc @ ±{tolerance:.1f} pIC50 : "
            f"{tolerance_accuracy[tolerance]:.2f}%"
        )


    # ========================================================
    # pIC50 BIN CREATION
    # ========================================================

    min_score = np.floor(
        targets.min() /
        bin_width
    ) * bin_width

    max_score = np.ceil(
        targets.max() /
        bin_width
    ) * bin_width


    bin_edges = np.arange(
        min_score,
        max_score + bin_width,
        bin_width
    )


    # ========================================================
    # PER-BIN RESULTS
    # ========================================================

    bin_results = []


    print()
    print("=" * 130)
    print(
        "pIC50 BIN ANALYSIS"
    )
    print("=" * 130)


    print(
        f"{'Range':<16}"
        f"{'N':>8}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'Acc±0.5':>14}"
        f"{'Acc±1.0':>14}"
        f"{'Acc±1.5':>14}"
        f"{'Same Bin':>14}"
    )


    print("-" * 130)


    for i in range(
        len(bin_edges) - 1
    ):

        low = bin_edges[i]

        high = bin_edges[i + 1]


        # ----------------------------------------------------
        # Actual target bin
        # ----------------------------------------------------

        if (
            i ==
            len(bin_edges) - 2
        ):

            mask = (
                (targets >= low) &
                (targets <= high)
            )

        else:

            mask = (
                (targets >= low) &
                (targets < high)
            )


        n = int(
            mask.sum()
        )


        if n == 0:

            continue


        bin_targets = (
            targets[mask]
        )

        bin_predictions = (
            predictions[mask]
        )

        bin_errors = (
            errors[mask]
        )


        # ----------------------------------------------------
        # Metrics inside bin
        # ----------------------------------------------------

        bin_mae = (
            bin_errors.mean()
        )


        bin_rmse = np.sqrt(
            np.mean(
                (
                    bin_predictions -
                    bin_targets
                ) ** 2
            )
        )


        acc_05 = (
            bin_errors <= 0.5
        ).mean() * 100


        acc_10 = (
            bin_errors <= 1.0
        ).mean() * 100


        acc_15 = (
            bin_errors <= 1.5
        ).mean() * 100


        # ----------------------------------------------------
        # Same-bin accuracy
        # ----------------------------------------------------

        actual_bin_index = np.floor(
            bin_targets /
            bin_width
        )


        predicted_bin_index = np.floor(
            bin_predictions /
            bin_width
        )


        same_bin_accuracy = (
            predicted_bin_index ==
            actual_bin_index
        ).mean() * 100


        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        range_text = (
            f"{low:.1f}-{high:.1f}"
        )


        print(
            f"{range_text:<16}"
            f"{n:>8}"
            f"{bin_mae:>12.3f}"
            f"{bin_rmse:>12.3f}"
            f"{acc_05:>13.2f}%"
            f"{acc_10:>13.2f}%"
            f"{acc_15:>13.2f}%"
            f"{same_bin_accuracy:>13.2f}%"
        )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        bin_results.append({

            "range":
                range_text,

            "low":
                low,

            "high":
                high,

            "n":
                n,

            "MAE":
                bin_mae,

            "RMSE":
                bin_rmse,

            "Acc@0.5":
                acc_05,

            "Acc@1.0":
                acc_10,

            "Acc@1.5":
                acc_15,

            "Same_Bin_Accuracy":
                same_bin_accuracy
        })


    print("=" * 130)


    # ========================================================
    # RETURN
    # ========================================================

    metrics = {

        "Split":
            split_name,

        "Samples":
            len(targets),

        "RMSE":
            rmse,

        "MAE":
            mae,

        "Pearson_r":
            pearson_r,

        "Acc@0.5":
            tolerance_accuracy[0.5],

        "Acc@1.0":
            tolerance_accuracy[1.0],

        "Acc@1.5":
            tolerance_accuracy[1.5],

        "Evaluation_Time_Min":
            elapsed / 60
    }


    return (
        predictions,
        targets,
        metrics,
        bin_results
    )


# ============================================================
# RUN COLD-DRUG
# ============================================================

pred_drug, y_drug, metrics_drug, bins_drug = evaluate_split(

    model,

    cold_drug,

    device,

    split_name="COLD-DRUG",

    batch_size=BATCH_SIZE,

    bin_width=BIN_WIDTH
)


# ============================================================
# RUN COLD-PAIR
# ============================================================

pred_pair, y_pair, metrics_pair, bins_pair = evaluate_split(

    model,

    cold_pair,

    device,

    split_name="COLD-PAIR",

    batch_size=BATCH_SIZE,

    bin_width=BIN_WIDTH
)


# ============================================================
# RUN COLD-TARGET
# ============================================================

pred_target, y_target, metrics_target, bins_target = evaluate_split(

    model,

    cold_target,

    device,

    split_name="COLD-TARGET",

    batch_size=BATCH_SIZE,

    bin_width=BIN_WIDTH
)


# ============================================================
# COLLECT RESULTS
# ============================================================

all_results = [

    metrics_drug,

    metrics_pair,

    metrics_target
]


all_bin_results = {

    "COLD-DRUG":
        bins_drug,

    "COLD-PAIR":
        bins_pair,

    "COLD-TARGET":
        bins_target
}


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_csv = os.path.join(
    OUTPUT_DIR,
    "v3_summary.csv"
)


with open(
    summary_csv,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            all_results[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )


# ============================================================
# SAVE PER-BIN CSV
# ============================================================

bin_csv = os.path.join(
    OUTPUT_DIR,
    "v3_per_bin_results.csv"
)


bin_fieldnames = [

    "Split",
    "range",
    "low",
    "high",
    "n",
    "MAE",
    "RMSE",
    "Acc@0.5",
    "Acc@1.0",
    "Acc@1.5",
    "Same_Bin_Accuracy"
]


with open(
    bin_csv,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=bin_fieldnames
    )

    writer.writeheader()


    for split_name, results in (
        all_bin_results.items()
    ):

        for row in results:

            output_row = {

                "Split":
                    split_name,

                **row
            }

            writer.writerow(
                output_row
            )


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

predictions_file = os.path.join(
    OUTPUT_DIR,
    "v3_predictions.npz"
)


np.savez_compressed(

    predictions_file,

    cold_drug_predictions=pred_drug,
    cold_drug_targets=y_drug,

    cold_pair_predictions=pred_pair,
    cold_pair_targets=y_pair,

    cold_target_predictions=pred_target,
    cold_target_targets=y_target
)


# ============================================================
# SAVE HUMAN-READABLE REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "v3_evaluation_report.txt"
)


with open(
    report_file,
    "w"
) as f:

    f.write(
        "DTI-GNN V3 COLD-START EVALUATION REPORT\n"
    )

    f.write(
        "=" * 90 + "\n\n"
    )

    f.write(
        f"Checkpoint: "
        f"{CHECKPOINT_NAME}\n"
    )

    f.write(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}\n"
    )

    f.write(
        f"Training loss: "
        f"{checkpoint['train_loss']:.6f}\n"
    )

    f.write(
        f"Device: "
        f"{device}\n"
    )

    if torch.cuda.is_available():

        f.write(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}\n"
        )

    f.write(
        f"Batch size: "
        f"{BATCH_SIZE}\n"
    )

    f.write(
        f"Bin width: "
        f"{BIN_WIDTH}\n"
    )

    f.write(
        f"Cold-start data source: "
        f"{EVAL_DIR}\n\n"
    )


    # --------------------------------------------------------
    # GLOBAL RESULTS
    # --------------------------------------------------------

    f.write(
        "=" * 90 + "\n"
    )

    f.write(
        "GLOBAL RESULTS\n"
    )

    f.write(
        "=" * 90 + "\n\n"
    )


    for result in all_results:

        f.write(
            f"{result['Split']}\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            f"Samples       : "
            f"{result['Samples']}\n"
        )

        f.write(
            f"RMSE          : "
            f"{result['RMSE']:.4f}\n"
        )

        f.write(
            f"MAE           : "
            f"{result['MAE']:.4f}\n"
        )

        f.write(
            f"Pearson r     : "
            f"{result['Pearson_r']:.4f}\n"
        )

        f.write(
            f"Acc @ ±0.5    : "
            f"{result['Acc@0.5']:.2f}%\n"
        )

        f.write(
            f"Acc @ ±1.0    : "
            f"{result['Acc@1.0']:.2f}%\n"
        )

        f.write(
            f"Acc @ ±1.5    : "
            f"{result['Acc@1.5']:.2f}%\n"
        )

        f.write(
            f"Evaluation time: "
            f"{result['Evaluation_Time_Min']:.2f} min\n\n"
        )


    # --------------------------------------------------------
    # PER-BIN RESULTS
    # --------------------------------------------------------

    f.write(
        "=" * 90 + "\n"
    )

    f.write(
        "PER-pIC50-BIN RESULTS\n"
    )

    f.write(
        "=" * 90 + "\n\n"
    )


    for split_name, results in (
        all_bin_results.items()
    ):

        f.write(
            f"{split_name}\n"
        )

        f.write(
            "-" * 110 + "\n"
        )

        f.write(
            f"{'Range':<16}"
            f"{'N':>8}"
            f"{'MAE':>12}"
            f"{'RMSE':>12}"
            f"{'Acc±0.5':>14}"
            f"{'Acc±1.0':>14}"
            f"{'Acc±1.5':>14}"
            f"{'Same Bin':>14}\n"
        )

        f.write(
            "-" * 110 + "\n"
        )


        for row in results:

            f.write(
                f"{row['range']:<16}"
                f"{row['n']:>8}"
                f"{row['MAE']:>12.3f}"
                f"{row['RMSE']:>12.3f}"
                f"{row['Acc@0.5']:>13.2f}%"
                f"{row['Acc@1.0']:>13.2f}%"
                f"{row['Acc@1.5']:>13.2f}%"
                f"{row['Same_Bin_Accuracy']:>13.2f}%\n"
            )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print()
print("=" * 90)
print("EVALUATION COMPLETE")
print("=" * 90)

print()

print(
    "Cold-start datasets were loaded from:"
)

print(
    DATA_DIR
)

print()

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Files created:"
)

print(
    "  1. v3_evaluation_report.txt"
)

print(
    "  2. v3_summary.csv"
)

print(
    "  3. v3_per_bin_results.csv"
)

print(
    "  4. v3_predictions.npz"
)

print()

print("=" * 90)