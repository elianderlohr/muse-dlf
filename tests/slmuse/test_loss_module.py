import os
import sys
import unittest
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, src_dir)

from src.model.slmuse_dlf.loss_module import LossModule


class TestLossModule(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLossModule, self).__init__(*args, **kwargs)

        # Initialize parameters for testing
        self.lambda_orthogonality = 0.1
        self.M = 1.0
        self.t = 5
        self._debug = False

        # Initialize model
        self.model = LossModule(
            lambda_orthogonality=self.lambda_orthogonality,
            M=self.M,
            t=self.t,
            _debug=self._debug,
        )

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_contrastive_loss(self):
        # Example inputs
        batch_size = 32
        embedding_dim = 128
        v = torch.randn(batch_size, embedding_dim).to(self.device)
        vhat = torch.randn(batch_size, embedding_dim).to(self.device)
        negatives = torch.randn(10, embedding_dim).to(
            self.device
        )  # Example: 10 negatives
        mask = torch.ones(batch_size).to(self.device)  # Non-zero mask

        loss = self.model.contrastive_loss(v, vhat, negatives, mask)

        # Ensure loss tensor shape and values are as expected
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        self.assertTrue(loss >= 0)

    def test_focal_triplet_loss(self):
        # Example inputs
        batch_size = 32
        embedding_dim = 128
        num_classes = 10
        v = torch.randn(batch_size, embedding_dim).to(self.device)
        vhat_z = torch.randn(batch_size, embedding_dim).to(self.device)
        g = torch.randn(batch_size, num_classes).to(self.device)
        F = torch.randn(num_classes, embedding_dim).to(self.device)
        mask = torch.ones(batch_size).to(self.device)  # Non-zero mask

        loss = self.model.focal_triplet_loss(v, vhat_z, g, F, mask)

        # Ensure loss tensor shape and values are as expected
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        self.assertTrue(loss >= 0)

    def test_orthogonality_term(self):
        # Example input
        num_classes = 10
        F = torch.randn(num_classes, num_classes).to(self.device)

        loss = self.model.orthogonality_term(F)

        # Ensure loss value is non-negative
        self.assertTrue(loss >= 0)

    def test_forward(self):
        # Example inputs
        batch_size = 32
        embedding_dim = 128
        num_classes = 10
        v = torch.randn(batch_size, embedding_dim).to(self.device)
        vhat = torch.randn(batch_size, embedding_dim).to(self.device)
        d = torch.randn(batch_size, num_classes).to(self.device)
        g = torch.randn(batch_size, num_classes).to(self.device)
        F = torch.randn(num_classes, embedding_dim).to(self.device)
        negatives = torch.randn(10, embedding_dim).to(
            self.device
        )  # Example: 10 negatives
        mask = torch.ones(batch_size).to(self.device)  # Non-zero mask

        # Construct input dictionary as expected by forward method
        c = {
            "v": v,
            "vhat": vhat,
            "d": d,
            "g": g,
            "F": F,
        }

        loss = self.model.forward(c, negatives, mask)

        # Ensure loss value is non-negative
        self.assertTrue(loss >= 0)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

    def test_half_empty_batch(self):
        # Example inputs with half of the embeddings being zero
        batch_size = 32
        embedding_dim = 128
        num_classes = 10
        v = torch.cat(
            [
                torch.randn(batch_size // 2, embedding_dim),
                torch.zeros(batch_size // 2, embedding_dim),
            ],
            dim=0,
        ).to(self.device)
        vhat = torch.cat(
            [
                torch.randn(batch_size // 2, embedding_dim),
                torch.zeros(batch_size // 2, embedding_dim),
            ],
            dim=0,
        ).to(self.device)
        d = torch.randn(batch_size, num_classes).to(self.device)
        g = torch.randn(batch_size, num_classes).to(self.device)
        F = torch.randn(num_classes, embedding_dim).to(self.device)
        negatives = torch.randn(10, embedding_dim).to(self.device)
        mask = torch.cat(
            [torch.ones(batch_size // 2), torch.zeros(batch_size // 2)]
        ).to(self.device)

        # Construct input dictionary as expected by forward method
        c = {
            "v": v,
            "vhat": vhat,
            "d": d,
            "g": g,
            "F": F,
        }

        loss = self.model.forward(c, negatives, mask)

        # Ensure loss value is non-negative and the correct shape
        self.assertTrue(loss >= 0)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

    def test_full_empty_batch(self):
        # Example inputs with all embeddings being zero
        batch_size = 32
        embedding_dim = 128
        num_classes = 10
        v = torch.zeros(batch_size, embedding_dim).to(self.device)
        vhat = torch.zeros(batch_size, embedding_dim).to(self.device)
        d = torch.randn(batch_size, num_classes).to(self.device)
        g = torch.randn(batch_size, num_classes).to(self.device)
        F = torch.randn(num_classes, embedding_dim).to(self.device)
        negatives = torch.randn(10, embedding_dim).to(self.device)
        mask = torch.zeros(batch_size).to(self.device)  # Fully zero mask

        # Construct input dictionary as expected by forward method
        c = {
            "v": v,
            "vhat": vhat,
            "d": d,
            "g": g,
            "F": F,
        }

        loss = self.model.forward(c, negatives, mask)

        # Ensure loss is zero and the correct shape
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())


if __name__ == "__main__":
    unittest.main()
