\"""
DTI-GNN V3 training script — optimized for an unattended overnight run.

Key fixes vs. the notebook version:
  1. Batches are grouped by target_idx (not randomly shuffled across all
     569K rows), so the within-batch ESM dedup logic actually has repeated
     targets to deduplicate. This is the fix for "epoch 1 didn't finish
     after 6 hours" — random batches with ~2,667 unique targets almost
     never share a target within a batch of 32, so ESM was effectively
     running once per ROW, not once per unique target per batch.
  2. No CPU<->GPU sync per batch (bin lookup now done with torch.bucketize
     on-device instead of numpy .cpu().numpy() round-trips).
  3. Mixed precision (autocast + GradScaler) to use Ada tensor cores.
  4. Per-batch progress + timing logged to both console and a log file,
     so you can check progress without watching it live.
  5. Resumable checkpointing: if interrupted, re-running this script picks
     up from the last saved epoch instead of starting over.

Run with:  python train_v3.py
(runs in the foreground; use `start /min` or Task Scheduler on Windows,
or `nohup python train_v3.py &` on Linux/Mac, to leave it running
unattended and log to file instead of the terminal.)
"""

import os
import sys
import time
import pickle
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import esm

# ============================================================
# Paths (matches your local folder layout)
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "version3")
LOG_PATH = os.path.join(PROJECT_DIR, "train_v3.log")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Logging — writes to both console and a file, so you can
# `tail`/open train_v3.log in the morning to see exactly how
# far it got, without needing the terminal to stay visible.
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================
NUM_EPOCHS = 15
MAX_BATCH_SIZE = 128       # per-target-group batch size (see grouped sampler below)
LR = 5e-4
CHECKPOINT_EVERY = 1
N_BINS = 20
MAX_WEIGHT = 10.0
LOG_EVERY_N_BATCHES = 50   # progress line frequency


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load target vocabulary + sequences
    # --------------------------------------------------------
    with open(os.path.join(DATA_DIR, "target_to_idx.pkl"), "rb") as f:
        target_to_idx = pickle.load(f)
    with open(os.path.join(DATA_DIR, "target_sequences.pkl"), "rb") as f:
        target_sequences = pickle.load(f)
    log.info(f"Targets: {len(target_to_idx)}")

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------
    train_data = torch.load(os.path.join(DATA_DIR, "train_data_v2.pt"), weights_only=False)
    log.info(f"Training examples: {len(train_data)}")

    # --------------------------------------------------------
    # THE KEY FIX: group example indices by target_idx, then
    # chunk each target's group into batches of up to
    # MAX_BATCH_SIZE. Every batch now contains examples for
    # ONE target only, so TargetEncoder's dedup logic runs ESM
    # exactly once per batch (not once per row).
    # --------------------------------------------------------
    groups = defaultdict(list)
    for i, data in enumerate(train_data):
        groups[int(data.target_idx.item())].append(i)

    log.info(f"Unique targets in train set: {len(groups)}")
    avg_group_size = len(train_data) / len(groups)
    log.info(f"Avg examples per target: {avg_group_size:.1f}")

    def build_batches(seed):
        rng = np.random.RandomState(seed)
        batches = []
        leftovers = []  # collect any batch that would end up with size 1
        for tgt, idxs in groups.items():
            idxs = idxs.copy()
            rng.shuffle(idxs)
            for start in range(0, len(idxs), MAX_BATCH_SIZE):
                chunk = idxs[start:start + MAX_BATCH_SIZE]
                if len(chunk) == 1:
                    # BatchNorm1d needs >=2 examples per batch when training;
                    # stash singletons and re-batch them together below
                    # (mixed targets in one batch is fine — the dedup logic
                    # in TargetEncoder already handles multiple targets)
                    leftovers.extend(chunk)
                else:
                    batches.append(chunk)

        rng.shuffle(leftovers)
        for start in range(0, len(leftovers), MAX_BATCH_SIZE):
            chunk = leftovers[start:start + MAX_BATCH_SIZE]
            if len(chunk) == 1:
                # extremely rare: exactly one leftover-of-leftovers.
                # merge it into the last real batch rather than crash.
                if batches:
                    batches[-1] = batches[-1] + chunk
                else:
                    batches.append(chunk)  # only happens if dataset itself has 1 example
            else:
                batches.append(chunk)

        rng.shuffle(batches)  # shuffle BATCH order, not contents, each epoch
        return batches

    # --------------------------------------------------------
    # pIC50 range weights (unchanged logic from the notebook,
    # just computed once here)
    # --------------------------------------------------------
    train_y = np.array([d.y.item() for d in train_data])
    bin_edges = np.linspace(train_y.min(), train_y.max(), N_BINS + 1)
    bin_idx_all = np.digitize(train_y, bin_edges[1:-1])
    bin_counts = np.maximum(np.bincount(bin_idx_all, minlength=N_BINS), 1)
    bin_weights = 1.0 / np.sqrt(bin_counts)
    bin_weights = bin_weights / bin_weights.mean()
    bin_weights = np.clip(bin_weights, None, MAX_WEIGHT)
    log.info(f"Bin weights: min={bin_weights.min():.3f}, max={bin_weights.max():.3f}")

    bin_edges_t = torch.tensor(bin_edges[1:-1], dtype=torch.float32, device=device)
    bin_weights_t = torch.tensor(bin_weights, dtype=torch.float32, device=device)

    def weighted_mse_loss(pred, target):
        # torch.bucketize instead of numpy digitize -> stays on GPU, no sync
        idx = torch.bucketize(target.detach(), bin_edges_t)
        w = bin_weights_t[idx]
        return (w * (pred - target) ** 2).mean()

    # --------------------------------------------------------
    # ESM-2, last transformer layer unfrozen
    # --------------------------------------------------------
    log.info("Loading ESM-2 35M...")
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    for p in esm_model.parameters():
        p.requires_grad = False
    last_layer_idx = len(esm_model.layers) - 1
    for p in esm_model.layers[last_layer_idx].parameters():
        p.requires_grad = True
    esm_model = esm_model.to(device)

    trainable = sum(p.numel() for p in esm_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in esm_model.parameters())
    log.info(f"ESM trainable params: {trainable:,} / {total:,} (layer {last_layer_idx} unfrozen)")

    # --------------------------------------------------------
    # Model definition
    # --------------------------------------------------------
    class DrugEncoder(nn.Module):
        def __init__(self, in_channels, hidden_channels=128, out_channels=128):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
            return global_mean_pool(x, batch)

    class TargetEncoder(nn.Module):
        def __init__(self, esm_model, batch_converter, target_sequences, hidden_dim=128, esm_dim=480):
            super().__init__()
            self.esm_model = esm_model
            self.batch_converter = batch_converter
            self.target_sequences = target_sequences
            self.proj = nn.Sequential(
                nn.Linear(esm_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            )

        def forward(self, target_idx_batch, device):
            unique_idx, inverse = torch.unique(target_idx_batch, return_inverse=True)
            seqs = [(str(i.item()), self.target_sequences[i.item()]) for i in unique_idx]
            _, _, tokens = self.batch_converter(seqs)
            tokens = tokens.to(device)
            results = self.esm_model(tokens, repr_layers=[12], return_contacts=False)
            reps = results["representations"][12]
            pooled = torch.stack([reps[i, 1:len(seq) + 1].mean(0) for i, (_, seq) in enumerate(seqs)])
            per_example = pooled[inverse]
            return self.proj(per_example)

    class DTIModelV3(nn.Module):
        def __init__(self, drug_in_channels, esm_model, batch_converter, target_sequences, hidden_dim=128):
            super().__init__()
            self.drug_encoder = DrugEncoder(drug_in_channels, hidden_dim, hidden_dim)
            self.target_encoder = TargetEncoder(esm_model, batch_converter, target_sequences, hidden_dim)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_dim, 64), nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x, edge_index, batch, target_idx, device):
            drug_vec = self.drug_encoder(x, edge_index, batch)
            target_vec = self.target_encoder(target_idx, device)
            combined = torch.cat([drug_vec, target_vec], dim=1)
            return self.fusion(combined).squeeze(-1)

    model = DTIModelV3(
        drug_in_channels=18, esm_model=esm_model, batch_converter=batch_converter,
        target_sequences=target_sequences, hidden_dim=128
    ).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # --------------------------------------------------------
    # Resume from the latest checkpoint if one exists
    # --------------------------------------------------------
    start_epoch = 1
    existing = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("dti_model_v3_epoch")],
        key=lambda f: int(f.split("epoch")[1].split(".")[0])
    )
    if existing:
        last_ckpt = existing[-1]
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, last_ckpt), weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        log.info(f"Resuming from {last_ckpt} -> starting at epoch {start_epoch}")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        batches = build_batches(seed=epoch)
        n_batches = len(batches)
        log.info(f"Epoch {epoch}/{NUM_EPOCHS} starting — {n_batches} batches")

        epoch_start = time.time()
        total_loss, total_examples = 0.0, 0

        for b_i, idxs in enumerate(batches, 1):
            batch_list = [train_data[i] for i in idxs]
            batch = DataLoader(batch_list, batch_size=len(batch_list), shuffle=False)
            batch = next(iter(batch)).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred = model(batch.x, batch.edge_index, batch.batch, batch.target_idx, device)
                loss = weighted_mse_loss(pred, batch.y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * batch.num_graphs
            total_examples += batch.num_graphs

            if b_i % LOG_EVERY_N_BATCHES == 0 or b_i == n_batches:
                elapsed = time.time() - epoch_start
                rate = b_i / elapsed
                eta = (n_batches - b_i) / rate if rate > 0 else float("inf")
                log.info(
                    f"  epoch {epoch} | batch {b_i}/{n_batches} "
                    f"| running loss {total_loss/total_examples:.4f} "
                    f"| {rate:.2f} batch/s | ETA {eta/60:.1f} min"
                )

        avg_loss = total_loss / total_examples
        epoch_time = time.time() - epoch_start
        log.info(f"Epoch {epoch:02d} DONE | Weighted Train Loss: {avg_loss:.4f} | Time: {epoch_time/60:.1f} min")

        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"dti_model_v3_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "bin_edges": bin_edges,
                "bin_weights": bin_weights,
            }, ckpt_path)
            log.info(f"Checkpoint saved: {ckpt_path}")

    log.info("Training completed.")


if __name__ == "__main__":
    main()