import unittest

import torch
from torch_geometric.data import Batch, Data

from model_v4 import DTIModelV4


class CachedEmbeddingModelTests(unittest.TestCase):
    def test_cached_target_embeddings_support_forward_and_backward(self):
        model = DTIModelV4(
            drug_in_channels=3,
            target_embeddings=torch.randn(2, 1280),
            hidden_dim=8,
        )
        batch = Batch.from_data_list([
            Data(
                x=torch.randn(2, 3),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                target_idx=torch.tensor([0]),
                y=torch.tensor([6.0]),
            ),
            Data(
                x=torch.randn(3, 3),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                target_idx=torch.tensor([1]),
                y=torch.tensor([7.0]),
            ),
        ])

        prediction = model(batch.x, batch.edge_index, batch.batch, batch.target_idx)
        prediction.sum().backward()

        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertNotIn("target_embeddings", model.state_dict())
        self.assertIsNotNone(model.target_proj[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
