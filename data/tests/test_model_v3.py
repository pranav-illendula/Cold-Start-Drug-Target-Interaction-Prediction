import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from model_v3 import DTIModelV3, TargetEncoder


class FakeBatchConverter:
    def __call__(self, sequences):
        max_length = max(len(sequence) for _, sequence in sequences) + 2
        tokens = torch.zeros((len(sequences), max_length), dtype=torch.long)
        for row, (_, sequence) in enumerate(sequences):
            tokens[row, 1 : len(sequence) + 1] = torch.arange(
                1,
                len(sequence) + 1,
            )
        return None, None, tokens


class FakeESM(nn.Module):
    def __init__(self, embedding_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(16, embedding_dim)

    def forward(self, tokens, repr_layers, return_contacts):
        return {"representations": {12: self.embedding(tokens)}}


class TargetEncoderTests(unittest.TestCase):
    def test_repeated_targets_share_a_stable_embedding(self):
        encoder = TargetEncoder(
            FakeESM(),
            FakeBatchConverter(),
            ["AC", "DEF"],
            hidden_dim=8,
            esm_dim=4,
        )
        encoder.train()

        output = encoder(torch.tensor([0, 1, 0]), torch.device("cpu"))

        self.assertEqual(output.shape, (3, 8))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output[0], output[2])
        self.assertFalse(
            any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
                for layer in encoder.modules())
        )

    def test_gnn_forward_and_backward_with_repeated_target(self):
        model = DTIModelV3(
            drug_in_channels=3,
            esm_model=FakeESM(480),
            batch_converter=FakeBatchConverter(),
            target_sequences=["AC", "DEF"],
            hidden_dim=8,
        )
        model.eval()
        first_graph = Data(
            x=torch.randn(2, 3),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            target_idx=torch.tensor([0]),
        )
        second_graph = Data(
            x=torch.randn(3, 3),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            target_idx=torch.tensor([1]),
        )
        batch = Batch.from_data_list([first_graph, second_graph])

        predictions = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.target_idx,
            torch.device("cpu"),
        )
        predictions.sum().backward()

        single_batch = Batch.from_data_list([first_graph])
        single_prediction = model(
            single_batch.x,
            single_batch.edge_index,
            single_batch.batch,
            single_batch.target_idx,
            torch.device("cpu"),
        )

        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(torch.isfinite(predictions).all())
        self.assertLessEqual(predictions.abs().max().item(), 20)
        torch.testing.assert_close(predictions[0], single_prediction[0])
        self.assertIsNotNone(model.fusion[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
