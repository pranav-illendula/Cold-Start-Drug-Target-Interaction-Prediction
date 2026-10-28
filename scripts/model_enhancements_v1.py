"""Model and data helpers for the isolated Section 4 enhancement experiment."""

import math
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.nn import (
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch

try:
    from .bond_features_v4_2 import BOND_FEATURE_DIM
except ImportError:
    from bond_features_v4_2 import BOND_FEATURE_DIM


ATOM_FEATURE_DIM = 18
ESM_WINDOW_DIM = 1280
ESM_RESIDUE_DIM = 640
HIDDEN_DIM = 256
WINDOW_ATTENTION_DIM = 256
RANKING_MARGIN = 0.5
RANKING_WEIGHT = 0.2
PIC50_BIN_EDGES = (4.5, 6.0, 7.5, 9.0, 10.5, 12.0)
MAX_PIC50_BIN_WEIGHT = 5.0


class ResidualGINEBlock(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, edge_dim=BOND_FEATURE_DIM):
        super().__init__()
        self.conv = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            ),
            edge_dim=edge_dim,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, edge_index, edge_attr):
        return self.norm(x + self.dropout(self.conv(x, edge_index, edge_attr)))


class WindowAttentionPool(nn.Module):
    """Learned attention pooling across the ESM representations of sequence windows."""

    def __init__(self, input_dim=ESM_WINDOW_DIM, attention_dim=WINDOW_ATTENTION_DIM):
        super().__init__()
        self.projection = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, windows, mask):
        if windows.ndim != 3 or mask.shape != windows.shape[:2]:
            raise ValueError("Window embeddings and mask have incompatible shapes.")
        if windows.shape[-1] != ESM_WINDOW_DIM:
            raise ValueError(f"Expected {ESM_WINDOW_DIM}-dimensional window vectors.")
        if not mask.any(dim=1).all():
            raise ValueError("Every target must contain at least one valid window.")
        scores = self.score(torch.tanh(self.projection(windows))).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(windows * weights.unsqueeze(-1), dim=1), weights


class AtomResidueCrossAttention(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, residue_dim=ESM_RESIDUE_DIM, heads=8):
        super().__init__()
        self.residue_projection = nn.Linear(residue_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atoms, residues, residue_mask, return_attention=False):
        if atoms.ndim != 3 or residues.ndim != 3:
            raise ValueError("Atom and residue representations must be rank-three.")
        if residue_mask.shape != residues.shape[:2]:
            raise ValueError("Residue embeddings and mask have incompatible shapes.")
        if not residue_mask.any(dim=1).all():
            raise ValueError("Every graph must contain at least one valid residue.")
        projected = self.residue_projection(residues)
        attended, weights = self.attention(
            atoms,
            projected,
            projected,
            key_padding_mask=~residue_mask,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        conditioned = self.norm(atoms + attended)
        if return_attention:
            return conditioned, weights
        return conditioned, None


class DTIEnhancementModel(nn.Module):
    """GINE DTI model with window pooling and atom-to-residue cross-attention."""

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        residue_dim=ESM_RESIDUE_DIM,
        window_dim=ESM_WINDOW_DIM,
    ):
        super().__init__()
        if atom_feature_dim != ATOM_FEATURE_DIM or bond_feature_dim != BOND_FEATURE_DIM:
            raise ValueError("Enhancement model expects v4.2 atom/bond feature widths.")
        if residue_dim != ESM_RESIDUE_DIM or window_dim != ESM_WINDOW_DIM:
            raise ValueError("Enhancement model expects configured ESM feature widths.")
        self.hidden_dim = hidden_dim
        self.atom_projection = nn.Linear(atom_feature_dim, hidden_dim)
        self.drug_blocks = nn.ModuleList(
            ResidualGINEBlock(hidden_dim, bond_feature_dim) for _ in range(4)
        )
        self.drug_pool = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.window_pool = WindowAttentionPool(window_dim)
        self.target_projection = nn.Sequential(
            nn.Linear(window_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cross_attention = AtomResidueCrossAttention(
            hidden_dim,
            residue_dim,
        )
        self.cross_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gate_projection = nn.Linear(hidden_dim * 3, hidden_dim)
        self.regression = nn.Sequential(
            nn.Linear(hidden_dim * 6, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        graph_batch,
        window_embeddings,
        window_mask,
        residue_embeddings,
        residue_mask,
        return_attention=False,
    ):
        edge_attr = getattr(graph_batch, "edge_attr", None)
        if edge_attr is None or edge_attr.ndim != 2 or edge_attr.shape[1] != BOND_FEATURE_DIM:
            raise ValueError(f"Graphs must contain {BOND_FEATURE_DIM}-dimensional edge_attr.")
        if edge_attr.shape[0] != graph_batch.edge_index.shape[1]:
            raise ValueError("edge_attr rows must align with directed edges.")
        graph_count = int(graph_batch.num_graphs)
        if window_embeddings.shape[0] != graph_count:
            raise ValueError("One window embedding sequence is required per graph.")
        if residue_embeddings.shape[0] != graph_count:
            raise ValueError("One residue embedding sequence is required per graph.")

        atom_states = F.gelu(self.atom_projection(graph_batch.x))
        for block in self.drug_blocks:
            atom_states = block(
                atom_states,
                graph_batch.edge_index,
                edge_attr,
            )
        drug_vector = self.drug_pool(
            torch.cat(
                [
                    global_mean_pool(atom_states, graph_batch.batch),
                    global_max_pool(atom_states, graph_batch.batch),
                    global_add_pool(atom_states, graph_batch.batch),
                ],
                dim=-1,
            )
        )

        pooled_target, window_weights = self.window_pool(
            window_embeddings,
            window_mask,
        )
        target_vector = self.target_projection(pooled_target)

        dense_atoms, atom_mask = to_dense_batch(atom_states, graph_batch.batch)
        conditioned, attention = self.cross_attention(
            dense_atoms,
            residue_embeddings,
            residue_mask,
            return_attention,
        )
        atom_mask_expanded = atom_mask.unsqueeze(-1)
        conditioned_mean = (
            conditioned.masked_fill(~atom_mask_expanded, 0).sum(dim=1)
            / atom_mask.sum(dim=1, keepdim=True).clamp_min(1)
        )
        conditioned_max = conditioned.masked_fill(
            ~atom_mask_expanded,
            torch.finfo(conditioned.dtype).min,
        ).max(dim=1).values
        cross_vector = self.cross_pool(
            torch.cat([conditioned_mean, conditioned_max], dim=-1)
        )
        if return_attention:
            attention_maps = [
                attention[index][atom_mask[index]][:, residue_mask[index]]
                for index in range(graph_count)
            ]

        discrepancy = (drug_vector - target_vector).abs()
        gate = torch.sigmoid(
            self.gate_projection(
                torch.cat([drug_vector, target_vector, discrepancy], dim=-1)
            )
        )
        adaptive = gate * drug_vector + (1 - gate) * target_vector
        fused = torch.cat(
            [
                drug_vector,
                target_vector,
                drug_vector * target_vector,
                discrepancy,
                adaptive,
                cross_vector,
            ],
            dim=-1,
        )
        prediction = self.regression(fused).squeeze(-1)
        if return_attention:
            return prediction, {
                "window_weights": window_weights,
                "atom_residue_weights": attention_maps,
            }
        return prediction


def build_pic50_bin_weights(labels):
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 1 or not labels.size or not np.isfinite(labels).all():
        raise ValueError("Training labels must be a nonempty finite 1D array.")
    bins = np.searchsorted(PIC50_BIN_EDGES, labels, side="right")
    counts = np.bincount(bins, minlength=len(PIC50_BIN_EDGES) + 1)
    weights = np.ones_like(counts, dtype=np.float64)
    populated = counts > 0
    weights[populated] = np.minimum(
        np.sqrt(counts.max() / counts[populated]),
        MAX_PIC50_BIN_WEIGHT,
    )
    weights /= np.dot(weights, counts) / counts.sum()
    return torch.as_tensor(weights, dtype=torch.float32), counts


def hybrid_ranking_regression_loss(
    predictions,
    targets,
    bin_weights,
    ranking_weight=RANKING_WEIGHT,
    ranking_margin=RANKING_MARGIN,
):
    if predictions.ndim != 1 or targets.shape != predictions.shape:
        raise ValueError("Predictions and targets must be equal-length 1D tensors.")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("Predictions and targets must be finite.")
    boundaries = torch.as_tensor(
        PIC50_BIN_EDGES,
        dtype=targets.dtype,
        device=targets.device,
    )
    bin_indices = torch.bucketize(targets, boundaries, right=True)
    weights = bin_weights.to(device=targets.device, dtype=predictions.dtype)[
        bin_indices
    ]
    regression = F.smooth_l1_loss(
        predictions,
        targets,
        reduction="none",
    )
    weighted_regression = (regression * weights).mean()

    target_differences = targets[:, None] - targets[None, :]
    prediction_differences = predictions[:, None] - predictions[None, :]
    pair_mask = target_differences != 0
    pair_losses = F.relu(
        -torch.sign(target_differences) * prediction_differences + ranking_margin
    )
    if pair_mask.any():
        ranking = pair_losses[pair_mask].mean()
    else:
        ranking = predictions.new_zeros(())
    total = weighted_regression + ranking_weight * ranking
    return total, {
        "regression": weighted_regression.detach(),
        "ranking": ranking.detach(),
    }


class TargetEmbeddingStore:
    """Load per-target ESM tensors on demand from an indexed directory."""

    def __init__(self, directory, cache_size=16):
        self.directory = Path(directory)
        self.cache_size = cache_size
        self.cache = OrderedDict()
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ESM representation manifest missing: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
        if (
            manifest.get("dataset_version") != "recovered_baseline_v1"
            or manifest.get("model_name") != "esm2_t30_150M_UR50D"
        ):
            raise ValueError(f"Incompatible ESM representation manifest: {manifest_path}")
        self.manifest_targets = {
            int(item["target_idx"]): item
            for item in manifest.get("targets", [])
        }

    def get(self, target_index):
        target_index = int(target_index)
        if target_index in self.cache:
            self.cache.move_to_end(target_index)
            return self.cache[target_index]
        path = self.directory / f"{target_index}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"ESM target representation missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        windows = payload.get("window_embeddings")
        residues = payload.get("residue_embeddings")
        manifest_item = self.manifest_targets.get(target_index)
        if (
            manifest_item is None
            or payload.get("target_idx") != target_index
            or payload.get("target_id") != manifest_item.get("target_id")
            or payload.get("sequence_sha256") != manifest_item.get("sequence_sha256")
            or not isinstance(windows, torch.Tensor)
            or windows.ndim != 2
            or windows.shape[1] != ESM_WINDOW_DIM
            or windows.shape[0] != manifest_item.get("window_count")
            or not isinstance(residues, torch.Tensor)
            or residues.ndim != 2
            or residues.shape[1] != ESM_RESIDUE_DIM
            or residues.shape[0] != manifest_item.get("sequence_length")
        ):
            raise ValueError(f"Invalid ESM tensor shapes in {path}")
        result = (windows.float(), residues.float())
        self.cache[target_index] = result
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result


def collate_graphs(graphs, embedding_store):
    if not graphs:
        raise ValueError("Cannot collate an empty graph batch.")
    graph_batch = Batch.from_data_list(graphs)
    window_rows = []
    residue_rows = []
    for graph in graphs:
        windows, residues = embedding_store.get(graph.target_idx.item())
        window_rows.append(windows)
        residue_rows.append(residues)
    window_batch = pad_sequence(window_rows, batch_first=True)
    residue_batch = pad_sequence(residue_rows, batch_first=True)
    window_lengths = torch.as_tensor(
        [row.shape[0] for row in window_rows],
        dtype=torch.long,
    )
    residue_lengths = torch.as_tensor(
        [row.shape[0] for row in residue_rows],
        dtype=torch.long,
    )
    window_mask = torch.arange(window_batch.shape[1])[None, :] < window_lengths[:, None]
    residue_mask = torch.arange(residue_batch.shape[1])[None, :] < residue_lengths[:, None]
    return graph_batch, window_batch, window_mask, residue_batch, residue_mask


def conformal_scale_quantile(targets, means, standard_deviations, coverage=0.90):
    targets = np.asarray(targets, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    standard_deviations = np.asarray(standard_deviations, dtype=np.float64)
    if targets.ndim != 1 or means.shape != targets.shape or standard_deviations.shape != targets.shape:
        raise ValueError("Conformal inputs must be equally sized 1D arrays.")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one.")
    if not np.isfinite(targets).all() or not np.isfinite(means).all():
        raise ValueError("Conformal targets and means must be finite.")
    if not np.isfinite(standard_deviations).all() or (standard_deviations < 0).any():
        raise ValueError("Predictive standard deviations must be finite and nonnegative.")
    scores = np.abs(targets - means) / np.maximum(standard_deviations, 1e-6)
    rank = min(len(scores), math.ceil((len(scores) + 1) * coverage))
    if rank < 1:
        raise ValueError("At least one calibration observation is required.")
    return float(np.partition(scores, rank - 1)[rank - 1])


def target_group_conformal_quantile(
    targets,
    means,
    standard_deviations,
    target_indices,
    coverage=0.90,
):
    """Calibrate worst normalized error per held-out target group."""
    targets = np.asarray(targets, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    standard_deviations = np.asarray(standard_deviations, dtype=np.float64)
    target_indices = np.asarray(target_indices)
    if (
        targets.ndim != 1
        or means.shape != targets.shape
        or standard_deviations.shape != targets.shape
        or target_indices.shape != targets.shape
    ):
        raise ValueError("Target-group conformal inputs must be equally sized 1D arrays.")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one.")
    if not len(targets):
        raise ValueError("At least one calibration example is required.")
    if (
        not np.isfinite(targets).all()
        or not np.isfinite(means).all()
        or not np.isfinite(standard_deviations).all()
        or (standard_deviations < 0).any()
    ):
        raise ValueError("Conformal inputs must be finite with nonnegative scale.")
    scores = np.abs(targets - means) / np.maximum(standard_deviations, 1e-6)
    grouped_scores = [
        float(scores[target_indices == target_id].max())
        for target_id in np.unique(target_indices)
    ]
    rank = min(
        len(grouped_scores),
        math.ceil((len(grouped_scores) + 1) * coverage),
    )
    return float(np.partition(grouped_scores, rank - 1)[rank - 1])
