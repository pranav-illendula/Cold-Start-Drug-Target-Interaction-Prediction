"""Train the V4.2 Residual GINE + Gated Fusion model on the KIBA kinase dataset."""

import os
import time
import math
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "kiba")
GRAPH_DIR = os.path.join(DATA_DIR, "graphs")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "kiba")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "kiba")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ATOM_FEATURE_DIM = 18
BOND_FEATURE_DIM = 14
HIDDEN_DIM = 256
NUM_TARGETS = 229


class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_graphs(batch):
    batch_size = len(batch)
    num_nodes_list = [g["x"].shape[0] for g in batch]
    total_nodes = sum(num_nodes_list)

    x_list = [torch.tensor(g["x"], dtype=torch.float32) for g in batch]
    x = torch.cat(x_list, dim=0)

    edge_index_list = []
    edge_attr_list = []
    node_offset = 0

    batch_vec = torch.empty(total_nodes, dtype=torch.long)
    curr = 0
    for i, g in enumerate(batch):
        n = g["x"].shape[0]
        batch_vec[curr : curr + n] = i
        curr += n

        ei = torch.tensor(g["edge_index"], dtype=torch.long) + node_offset
        edge_index_list.append(ei)
        edge_attr_list.append(torch.tensor(g["edge_attr"], dtype=torch.float32))
        node_offset += n

    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)

    target_idx = torch.tensor([g["target_idx"] for g in batch], dtype=torch.long)
    y = torch.tensor([g["y"] for g in batch], dtype=torch.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch_vec,
        "target_idx": target_idx,
        "y": y,
        "batch_size": batch_size,
    }


class ResidualGINEBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim=BOND_FEATURE_DIM, dropout=0.20):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.shape[1] == 0:
            return self.norm(x + self.dropout(self.mlp(x)))
        src, dst = edge_index[0], edge_index[1]
        msg = F.relu(x[src] + self.edge_proj(edge_attr))
        out = torch.zeros_like(x)
        out.index_add_(0, dst, msg)
        updated = self.mlp(x + out)
        return self.norm(x + self.dropout(updated))


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, drug_vector, target_vector):
        compatibility = drug_vector * target_vector
        discrepancy = (drug_vector - target_vector).abs()
        gate = torch.sigmoid(
            self.gate_proj(
                torch.cat([drug_vector, target_vector, discrepancy], dim=-1)
            )
        )
        adaptive = gate * drug_vector + (1 - gate) * target_vector
        return torch.cat(
            [drug_vector, target_vector, compatibility, discrepancy, adaptive],
            dim=-1,
        )


class DTIModelV4Kiba(nn.Module):
    def __init__(self, drug_in_channels=ATOM_FEATURE_DIM, num_targets=NUM_TARGETS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drug_input_proj = nn.Linear(drug_in_channels, hidden_dim)
        self.drug_blocks = nn.ModuleList([
            ResidualGINEBlock(hidden_dim, BOND_FEATURE_DIM) for _ in range(4)
        ])
        self.drug_pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        self.target_embedding = nn.Embedding(num_targets, hidden_dim)
        self.target_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.fusion = GatedFusion(hidden_dim)
        self.regression = nn.Sequential(
            nn.Linear(hidden_dim * 5, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 1),
        )

    def forward(self, batch_data):
        x = batch_data["x"]
        edge_index = batch_data["edge_index"]
        edge_attr = batch_data["edge_attr"]
        batch = batch_data["batch"]
        target_idx = batch_data["target_idx"]
        batch_size = batch_data["batch_size"]

        h = self.drug_input_proj(x)
        for block in self.drug_blocks:
            h = block(h, edge_index, edge_attr)

        # Multi-pooling: mean, max, sum
        mean_pool = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        sum_pool = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        max_pool = torch.full((batch_size, self.hidden_dim), -1e9, device=x.device)

        sum_pool.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=batch_size).clamp_min(1).unsqueeze(-1)
        mean_pool = sum_pool / counts

        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                max_pool[i] = torch.max(h[mask], dim=0).values

        drug_vector = self.drug_pool_proj(torch.cat([mean_pool, max_pool, sum_pool], dim=-1))
        target_vector = self.target_proj(self.target_embedding(target_idx))

        fused = self.fusion(drug_vector, target_vector)
        pred = self.regression(fused).squeeze(-1)
        return pred


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch_data in dataloader:
        optimizer.zero_grad()
        preds = model(batch_data)
        loss = criterion(preds, batch_data["y"].to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_data["batch_size"]
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_data in dataloader:
            preds = model(batch_data)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_data["y"].numpy().tolist())
    targets = np.array(all_targets)
    preds = np.array(all_preds)
    errors = preds - targets
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    r = float(np.corrcoef(targets, preds)[0, 1]) if np.std(targets) > 0 and np.std(preds) > 0 else 0.0
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae, "pearson_r": r}


def main():
    device = torch.device("cpu")
    print(f"Training on device: {device}")

    train_graphs_path = os.path.join(GRAPH_DIR, "kiba_train_graphs.pkl")
    cold_drug_path = os.path.join(GRAPH_DIR, "kiba_cold_drug_graphs.pkl")
    cold_target_path = os.path.join(GRAPH_DIR, "kiba_cold_target_graphs.pkl")
    cold_pair_path = os.path.join(GRAPH_DIR, "kiba_cold_pair_graphs.pkl")

    with open(train_graphs_path, "rb") as f:
        train_graphs = pickle.load(f)
    print(f"Loaded {len(train_graphs)} training graphs.")

    # Reserve 10% for validation
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(train_graphs))
    val_size = int(len(train_graphs) * 0.10)
    train_subset = [train_graphs[i] for i in indices[val_size:]]
    val_subset = [train_graphs[i] for i in indices[:val_size]]

    train_loader = DataLoader(GraphDataset(train_subset), batch_size=128, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(GraphDataset(val_subset), batch_size=256, shuffle=False, collate_fn=collate_graphs)

    model = DTIModelV4Kiba().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=1.0)

    print("Beginning training protocol (10 representative epochs)...")
    best_val_mse = float("inf")
    history = []

    for epoch in range(1, 11):
        t0 = time.time()
        # Train on representative sample per epoch for responsiveness
        sample_indices = rng.choice(len(train_subset), size=min(12800, len(train_subset)), replace=False)
        sample_train = [train_subset[i] for i in sample_indices]
        sample_loader = DataLoader(GraphDataset(sample_train), batch_size=128, shuffle=True, collate_fn=collate_graphs)

        train_loss = train_epoch(model, sample_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val MSE: {val_metrics['mse']:.4f} | Val Pearson r: {val_metrics['pearson_r']:.4f} | Time: {dt:.1f}s")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_metrics["mse"],
            "val_pearson_r": val_metrics["pearson_r"],
        })

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "dti_model_v4_kiba_best.pt"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "dti_model_v4_kiba_last.pt"))
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best Validation MSE: {best_val_mse:.4f}")


if __name__ == "__main__":
    main()

