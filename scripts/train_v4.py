"""Train the frozen ESM-2 150M model with target-held-out validation."""

import csv
import hashlib
import json
import logging
import os
import pickle
import sys
import time

import esm
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader

from model_v4 import ARCHITECTURE_VERSION, DTIModelV4, ESM_MODEL_NAME


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
LOG_PATH = os.path.join(PROJECT_DIR, "train_v4.log")
CACHE_PATH = os.path.join(
    DATA_DIR,
    "esm2_t30_150M_cls_mean_embeddings.pt",
)
SPLIT_PATH = os.path.join(DATA_DIR, "v4_target_validation_split.json")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SEED = 17
VALIDATION_TARGET_FRACTION = 0.15
MAX_PROTEIN_LENGTH = 1024
EMBEDDING_BATCH_SIZE = 4
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-3
LOG_EVERY_N_BATCHES = 100


def sequence_fingerprint(target_sequences):
    digest = hashlib.sha256()
    for index, sequence in enumerate(target_sequences):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(sequence).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_create_embeddings(target_sequences, device):
    fingerprint = sequence_fingerprint(target_sequences)
    if os.path.isfile(CACHE_PATH):
        cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        if (
            cache.get("model_name") != ESM_MODEL_NAME
            or cache.get("pooling") != "cls_and_mean"
            or cache.get("max_sequence_length") != MAX_PROTEIN_LENGTH
            or cache.get("sequence_fingerprint") != fingerprint
        ):
            raise ValueError(
                f"Embedding cache metadata does not match the current target "
                f"sequences/model: {CACHE_PATH}. Move that specific cache file "
                "aside to regenerate it."
            )
        embeddings = cache["embeddings"].float()
        if embeddings.shape != (len(target_sequences), 1280):
            raise ValueError(
                f"Unexpected cached embedding shape {tuple(embeddings.shape)}."
            )
        log.info("Loaded cached frozen ESM-2 150M embeddings from %s", CACHE_PATH)
        return embeddings.to(device)

    log.info("Loading frozen ESM-2 150M to generate target embedding cache.")
    esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    for parameter in esm_model.parameters():
        parameter.requires_grad = False
    esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    ordered = sorted(
        (
            (index, str(sequence)[:MAX_PROTEIN_LENGTH])
            for index, sequence in enumerate(target_sequences)
        ),
        key=lambda item: len(item[1]),
    )
    embeddings = torch.empty((len(target_sequences), 1280), dtype=torch.float32)
    start_time = time.time()

    with torch.inference_mode():
        for start in range(0, len(ordered), EMBEDDING_BATCH_SIZE):
            chunk = ordered[start : start + EMBEDDING_BATCH_SIZE]
            sequences = [(str(index), sequence) for index, sequence in chunk]
            _, _, tokens = batch_converter(sequences)
            tokens = tokens.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                results = esm_model(
                    tokens,
                    repr_layers=[30],
                    return_contacts=False,
                )
            representations = results["representations"][30].float()
            pooled = []
            for row, (_, sequence) in enumerate(chunk):
                residues = representations[row, 1 : len(sequence) + 1]
                pooled.append(torch.cat([representations[row, 0], residues.mean(0)]))
            indices = [index for index, _ in chunk]
            embeddings[indices] = torch.stack(pooled).cpu()

            completed = min(start + len(chunk), len(ordered))
            if completed % 100 == 0 or completed == len(ordered):
                log.info(
                    "Embedded %d/%d targets (%.1f%%)",
                    completed,
                    len(ordered),
                    completed / len(ordered) * 100,
                )

    del esm_model
    torch.cuda.empty_cache()
    torch.save(
        {
            "model_name": ESM_MODEL_NAME,
            "pooling": "cls_and_mean",
            "max_sequence_length": MAX_PROTEIN_LENGTH,
            "sequence_fingerprint": fingerprint,
            "embeddings": embeddings,
        },
        CACHE_PATH,
    )
    log.info(
        "Saved %d cached target embeddings in %.1f minutes: %s",
        len(target_sequences),
        (time.time() - start_time) / 60,
        CACHE_PATH,
    )
    return embeddings.to(device)


def load_or_create_split(target_ids):
    if os.path.isfile(SPLIT_PATH):
        with open(SPLIT_PATH, encoding="utf-8") as source:
            split = json.load(source)
        train_targets = set(split["train_target_ids"])
        validation_targets = set(split["validation_target_ids"])
        if (
            train_targets | validation_targets != target_ids
            or train_targets & validation_targets
        ):
            raise ValueError(
                f"Saved validation split does not match current training targets: "
                f"{SPLIT_PATH}"
            )
        log.info("Loaded fixed target-held-out split from %s", SPLIT_PATH)
        return train_targets, validation_targets

    target_order = np.array(sorted(target_ids), dtype=np.int64)
    rng = np.random.RandomState(SEED)
    rng.shuffle(target_order)
    validation_count = max(
        1,
        min(
            len(target_order) - 1,
            round(len(target_order) * VALIDATION_TARGET_FRACTION),
        ),
    )
    validation_targets = set(target_order[:validation_count].tolist())
    train_targets = target_ids - validation_targets
    with open(SPLIT_PATH, "w", encoding="utf-8") as destination:
        json.dump(
            {
                "seed": SEED,
                "validation_target_fraction": VALIDATION_TARGET_FRACTION,
                "train_target_ids": sorted(train_targets),
                "validation_target_ids": sorted(validation_targets),
            },
            destination,
            indent=2,
        )
    log.info(
        "Saved target-held-out split: %d train targets, %d validation targets.",
        len(train_targets),
        len(validation_targets),
    )
    return train_targets, validation_targets


def collate_batch(data, indices, device):
    graphs = [data[int(index)] for index in indices]
    return next(iter(DataLoader(
        graphs,
        batch_size=len(graphs),
        shuffle=False,
    ))).to(device)


def evaluate(model, data, indices, device, label_mean, label_std):
    model.eval()
    targets = []
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            batch = collate_batch(
                data,
                indices[start : start + BATCH_SIZE],
                device,
            )
            prediction = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.target_idx,
            )
            prediction = prediction.float() * label_std + label_mean
            predictions.append(prediction.cpu().numpy())
            targets.append(batch.y.float().cpu().numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    errors = y_pred - y_true
    correlation = (
        float(pearsonr(y_true, y_pred).statistic)
        if np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "pearson_r": correlation,
        "accuracy_0_5": float(np.mean(np.abs(errors) <= 0.5)),
        "accuracy_1_0": float(np.mean(np.abs(errors) <= 1.0)),
        "accuracy_1_5": float(np.mean(np.abs(errors) <= 1.5)),
    }


def save_checkpoint(path, model, optimizer, epoch, best_mae, stale_epochs,
                    label_mean, label_std, metrics):
    torch.save(
        {
            "architecture_version": ARCHITECTURE_VERSION,
            "model_name": ESM_MODEL_NAME,
            "embedding_cache": os.path.basename(CACHE_PATH),
            "epoch": epoch,
            "best_validation_mae": best_mae,
            "stale_epochs": stale_epochs,
            "label_mean": label_mean,
            "label_std": label_std,
            "validation_metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this requested ESM-2 150M run.")
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log.info("Device: %s (%s)", device, torch.cuda.get_device_name(0))

    with open(os.path.join(DATA_DIR, "target_to_idx.pkl"), "rb") as source:
        target_to_idx = pickle.load(source)
    with open(os.path.join(DATA_DIR, "target_sequences.pkl"), "rb") as source:
        target_sequences = pickle.load(source)
    if len(target_sequences) != len(target_to_idx):
        raise ValueError("Target sequences and target-index mapping differ in size.")
    if set(target_to_idx.values()) != set(range(len(target_sequences))):
        raise ValueError("Target indices must be contiguous from zero.")

    training_data = torch.load(
        os.path.join(DATA_DIR, "train_data_v2.pt"),
        map_location="cpu",
        weights_only=False,
    )
    target_ids = {int(sample.target_idx.item()) for sample in training_data}
    if not target_ids.issubset(set(range(len(target_sequences)))):
        raise ValueError("Training data contains a target index without a sequence.")
    train_targets, validation_targets = load_or_create_split(target_ids)

    train_indices = np.asarray(
        [
            index for index, sample in enumerate(training_data)
            if int(sample.target_idx.item()) in train_targets
        ],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [
            index for index, sample in enumerate(training_data)
            if int(sample.target_idx.item()) in validation_targets
        ],
        dtype=np.int64,
    )
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Target-held-out split produced an empty train or validation set.")
    log.info(
        "Examples: %d train, %d validation; targets: %d train, %d validation.",
        len(train_indices),
        len(validation_indices),
        len(train_targets),
        len(validation_targets),
    )

    embeddings = load_or_create_embeddings(target_sequences, device)
    model = DTIModelV4(
        drug_in_channels=18,
        target_embeddings=embeddings,
        hidden_dim=128,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda")

    train_y = np.asarray(
        [float(training_data[int(index)].y.item()) for index in train_indices],
        dtype=np.float32,
    )
    label_mean = float(train_y.mean())
    label_std = float(train_y.std())
    if not np.isfinite(label_std) or label_std <= 0:
        raise ValueError(f"Invalid training label standard deviation: {label_std}")

    best_path = os.path.join(CHECKPOINT_DIR, "dti_model_v4_best.pt")
    last_path = os.path.join(CHECKPOINT_DIR, "dti_model_v4_last.pt")
    history_path = os.path.join(CHECKPOINT_DIR, "dti_model_v4_validation.csv")
    start_epoch = 1
    best_mae = float("inf")
    stale_epochs = 0
    if os.path.isfile(last_path):
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("architecture_version") != ARCHITECTURE_VERSION
            or checkpoint.get("model_name") != ESM_MODEL_NAME
        ):
            raise ValueError(f"Incompatible resume checkpoint: {last_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            abs(checkpoint["label_mean"] - label_mean) > 1e-6
            or abs(checkpoint["label_std"] - label_std) > 1e-6
        ):
            raise ValueError("Resume checkpoint label normalization does not match data.")
        start_epoch = checkpoint["epoch"] + 1
        best_mae = checkpoint["best_validation_mae"]
        stale_epochs = checkpoint["stale_epochs"]
        log.info(
            "Resuming after epoch %d (best validation MAE %.4f; patience %d/%d).",
            checkpoint["epoch"],
            best_mae,
            stale_epochs,
            EARLY_STOPPING_PATIENCE,
        )
    if start_epoch > 1 and not os.path.isfile(best_path):
        raise FileNotFoundError(
            f"Cannot resume without the selected best checkpoint: {best_path}"
        )

    epoch = start_epoch
    while True:
        model.train()
        rng = np.random.RandomState(SEED + epoch)
        shuffled_indices = rng.permutation(train_indices)
        epoch_start = time.time()
        train_loss_sum = 0.0
        examples_seen = 0
        batches = range(0, len(shuffled_indices), BATCH_SIZE)

        for batch_number, start in enumerate(batches, 1):
            batch_indices = shuffled_indices[start : start + BATCH_SIZE]
            batch = collate_batch(training_data, batch_indices, device)
            labels = (batch.y.float() - label_mean) / label_std
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    batch.target_idx,
                )
                loss = F.smooth_l1_loss(prediction.float(), labels, beta=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * batch.num_graphs
            examples_seen += batch.num_graphs

            if batch_number % LOG_EVERY_N_BATCHES == 0:
                log.info(
                    "Epoch %d | batch %d | train SmoothL1 %.5f",
                    epoch,
                    batch_number,
                    train_loss_sum / examples_seen,
                )

        metrics = evaluate(
            model,
            training_data,
            validation_indices,
            device,
            label_mean,
            label_std,
        )
        improved = metrics["mae"] < best_mae - EARLY_STOPPING_MIN_DELTA
        if improved:
            best_mae = metrics["mae"]
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_mae,
                stale_epochs,
                label_mean,
                label_std,
                metrics,
            )
        else:
            stale_epochs += 1

        history_exists = os.path.isfile(history_path)
        with open(history_path, "a", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=[
                    "epoch",
                    "train_smooth_l1",
                    "validation_mae",
                    "validation_rmse",
                    "validation_pearson_r",
                    "accuracy_0_5",
                    "accuracy_1_0",
                    "accuracy_1_5",
                    "best_validation_mae",
                    "stale_epochs",
                ],
            )
            if not history_exists:
                writer.writeheader()
            writer.writerow({
                "epoch": epoch,
                "train_smooth_l1": train_loss_sum / examples_seen,
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_pearson_r": metrics["pearson_r"],
                "accuracy_0_5": metrics["accuracy_0_5"],
                "accuracy_1_0": metrics["accuracy_1_0"],
                "accuracy_1_5": metrics["accuracy_1_5"],
                "best_validation_mae": best_mae,
                "stale_epochs": stale_epochs,
            })

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_mae,
            stale_epochs,
            label_mean,
            label_std,
            metrics,
        )
        log.info(
            "Epoch %d complete in %.1f min | train SmoothL1 %.5f | "
            "validation MAE %.4f RMSE %.4f Pearson r %.4f | "
            "within 0.5/1.0/1.5: %.1f%%/%.1f%%/%.1f%% | best MAE %.4f | "
            "patience %d/%d",
            epoch,
            (time.time() - epoch_start) / 60,
            train_loss_sum / examples_seen,
            metrics["mae"],
            metrics["rmse"],
            metrics["pearson_r"],
            metrics["accuracy_0_5"] * 100,
            metrics["accuracy_1_0"] * 100,
            metrics["accuracy_1_5"] * 100,
            best_mae,
            stale_epochs,
            EARLY_STOPPING_PATIENCE,
        )
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            log.info(
                "Early stopping after epoch %d; best checkpoint is %s.",
                epoch,
                best_path,
            )
            break
        epoch += 1


if __name__ == "__main__":
    main()
