import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from bond_features_v4.2 import BOND_FEATURE_DIM


ARCHITECTURE_VERSION = 7
ESM_MODEL_NAME = "esm2_t30_150M_UR50D"
ESM_EMBEDDING_DIM = 1280


class ResidualGINEBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim=BOND_FEATURE_DIM, dropout=0.25):
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        return self.norm(x + self.dropout(self.conv(x, edge_index, edge_attr)))


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
            [
                drug_vector,
                target_vector,
                compatibility,
                discrepancy,
                adaptive,
            ],
            dim=-1,
        )


class DTIModelV4(nn.Module):
    def __init__(
        self,
        drug_in_channels,
        target_embeddings,
        hidden_dim=256,
        edge_dim=BOND_FEATURE_DIM,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.drug_input_proj = nn.Linear(drug_in_channels, hidden_dim)
        self.drug_blocks = nn.ModuleList(
            ResidualGINEBlock(hidden_dim, edge_dim=edge_dim)
            for _ in range(4)
        )
        self.drug_pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.register_buffer(
            "target_embeddings",
            target_embeddings.float(),
            persistent=False,
        )
        self.target_proj = nn.Sequential(
            nn.Linear(ESM_EMBEDDING_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gated_fusion = GatedFusion(hidden_dim)
        self.regression = nn.Sequential(
            nn.Linear(hidden_dim * 5, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, target_idx):
        if edge_attr.ndim != 2 or edge_attr.shape[1] != self.edge_dim:
            raise ValueError(
                f"Expected edge_attr with shape [num_bonds, {self.edge_dim}]; "
                f"received {tuple(edge_attr.shape)}."
            )
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_attr must contain one feature row for every directed edge."
            )

        x = F.gelu(self.drug_input_proj(x))
        for block in self.drug_blocks:
            x = block(x, edge_index, edge_attr)
        pooled = torch.cat(
            [
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch),
            ],
            dim=-1,
        )
        drug_vector = self.drug_pool_proj(pooled)
        target_vector = self.target_proj(self.target_embeddings[target_idx])
        fused = self.gated_fusion(drug_vector, target_vector)
        return self.regression(fused).squeeze(-1)
