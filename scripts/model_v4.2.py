
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


# Identifies the architecture version and the protein embedding configuration.
ARCHITECTURE_VERSION = 7
ESM_MODEL_NAME = "esm2_t30_150M_UR50D"
ESM_EMBEDDING_DIM = 1280


# Applies GINE message passing with a residual connection, dropout, and layer normalization.
# This allows atoms to learn from neighboring atoms and their connecting bond features.
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

    # Updates atom representations while preserving the original information through a residual connection.
    def forward(self, x, edge_index, edge_attr):
        return self.norm(x + self.dropout(self.conv(x, edge_index, edge_attr)))


# Combines drug and protein representations using their product, difference, and a learned gate.
# Returns five 256-dimensional vectors concatenated into one fused representation.
class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim * 3, hidden_dim)

    # Creates interaction features and learns how to weight drug and protein information.
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


# Defines the complete DTI model, including drug processing, protein processing,
# gated fusion, and the final pIC50 regression head.
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

        # Projects each atom's input features into the model's hidden dimension.
        self.drug_input_proj = nn.Linear(drug_in_channels, hidden_dim)

        # Creates four GINE blocks to learn increasingly rich molecular representations.
        self.drug_blocks = nn.ModuleList(
            ResidualGINEBlock(hidden_dim, edge_dim=edge_dim)
            for _ in range(4)
        )

        # Compresses mean, maximum, and sum pooled atom features into one drug vector.
        self.drug_pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )

        # Stores precomputed protein embeddings as a non-trainable model buffer.
        # persistent=False means they are not included in the model's state_dict.
        self.register_buffer(
            "target_embeddings",
            target_embeddings.float(),
            persistent=False,
        )

        # Projects the 1280-dimensional ESM-2 embedding into the 256-dimensional hidden space.
        self.target_proj = nn.Sequential(
            nn.Linear(ESM_EMBEDDING_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Combines the drug and protein vectors into a fused feature representation.
        self.gated_fusion = GatedFusion(hidden_dim)

        # Predicts one continuous pIC50 value from the fused drug-protein features.
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

    # Performs a complete forward pass: validates graph data, processes the drug,
    # retrieves the protein embedding, fuses both representations, and predicts pIC50.
    def forward(self, x, edge_index, edge_attr, batch, target_idx):

        # Ensures bond features have the expected dimensions.
        if edge_attr.ndim != 2 or edge_attr.shape[1] != self.edge_dim:
            raise ValueError(
                f"Expected edge_attr with shape [num_bonds, {self.edge_dim}]; "
                f"received {tuple(edge_attr.shape)}."
            )

        # Ensures every directed edge has a matching bond-feature row.
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_attr must contain one feature row for every directed edge."
            )

        # Converts the initial atom features into hidden representations.
        x = F.gelu(self.drug_input_proj(x))

        # Applies four residual GINE blocks to learn molecular structure features.
        for block in self.drug_blocks:
            x = block(x, edge_index, edge_attr)

        # Summarizes the atom representations using mean, max, and sum pooling.
        # The three 256-dimensional vectors produce a 768-dimensional pooled vector.
        pooled = torch.cat(
            [
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch),
            ],
            dim=-1,
        )

        # Projects the pooled molecular representation from 768 to 256 dimensions.
        drug_vector = self.drug_pool_proj(pooled)

        # Retrieves each protein's precomputed ESM-2 embedding and projects it to 256 dimensions.
        target_vector = self.target_proj(self.target_embeddings[target_idx])

        # Combines drug and protein information into a 1280-dimensional fused vector.
        fused = self.gated_fusion(drug_vector, target_vector)

        # Predicts pIC50 and removes the final singleton output dimension.
        return self.regression(fused).squeeze(-1)
