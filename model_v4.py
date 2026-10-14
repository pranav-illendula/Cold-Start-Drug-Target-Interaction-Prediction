import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


ARCHITECTURE_VERSION = 4
ESM_MODEL_NAME = "esm2_t30_150M_UR50D"
ESM_EMBEDDING_DIM = 1280


class DTIModelV4(nn.Module):
    def __init__(self, drug_in_channels, target_embeddings, hidden_dim=128):
        super().__init__()
        self.drug_conv1 = GCNConv(drug_in_channels, hidden_dim)
        self.drug_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.drug_conv3 = GCNConv(hidden_dim, hidden_dim)
        self.register_buffer(
            "target_embeddings",
            target_embeddings.float(),
            persistent=False,
        )
        self.target_proj = nn.Sequential(
            nn.Linear(ESM_EMBEDDING_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, batch, target_idx):
        x = F.relu(self.drug_conv1(x, edge_index))
        x = F.relu(self.drug_conv2(x, edge_index))
        x = self.drug_conv3(x, edge_index)
        drug_vector = global_mean_pool(x, batch)
        target_vector = self.target_proj(self.target_embeddings[target_idx])
        combined = torch.cat([drug_vector, target_vector], dim=1)
        return self.fusion(combined).squeeze(-1)
