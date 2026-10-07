import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


ARCHITECTURE_VERSION = 2


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
    def __init__(
        self,
        esm_model,
        batch_converter,
        target_sequences,
        hidden_dim=128,
        esm_dim=480,
    ):
        super().__init__()
        self.esm_model = esm_model
        self.batch_converter = batch_converter
        self.target_sequences = target_sequences
        self.proj = nn.Sequential(
            nn.Linear(esm_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, target_idx_batch, device):
        unique_idx, inverse = torch.unique(
            target_idx_batch,
            return_inverse=True,
        )
        sequences = [
            (str(index.item()), self.target_sequences[index.item()])
            for index in unique_idx
        ]
        _, _, tokens = self.batch_converter(sequences)
        tokens = tokens.to(device)

        results = self.esm_model(
            tokens,
            repr_layers=[12],
            return_contacts=False,
        )
        representations = results["representations"][12]
        pooled = torch.stack(
            [
                representations[i, 1 : len(sequence) + 1].mean(0)
                for i, (_, sequence) in enumerate(sequences)
            ]
        )

        projected_unique = self.proj(pooled)
        return projected_unique[inverse]


class DTIModelV3(nn.Module):
    def __init__(
        self,
        drug_in_channels,
        esm_model,
        batch_converter,
        target_sequences,
        hidden_dim=128,
    ):
        super().__init__()
        self.drug_encoder = DrugEncoder(
            drug_in_channels,
            hidden_dim,
            hidden_dim,
        )
        self.target_encoder = TargetEncoder(
            esm_model,
            batch_converter,
            target_sequences,
            hidden_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, batch, target_idx, device):
        drug_vector = self.drug_encoder(x, edge_index, batch)
        target_vector = self.target_encoder(target_idx, device)
        combined = torch.cat([drug_vector, target_vector], dim=1)
        return self.fusion(combined).squeeze(-1)
