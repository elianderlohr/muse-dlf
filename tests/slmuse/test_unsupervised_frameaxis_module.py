import os
import sys
import unittest
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, src_dir)

from src.model.slmuse_dlf.frameaxis_autoencoder import FrameAxisAutoencoder
from src.model.slmuse_dlf.loss_module import LossModule
from src.model.slmuse_dlf.unsupervised_frameaxis_module import MUSEFrameAxisUnsupervised


class TestMUSEFrameAxisUnsupervised(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMUSEFrameAxisUnsupervised, self).__init__(*args, **kwargs)

        # Initialize parameters for testing
        self.embedding_dim = 768
        self.frameaxis_dim = 64
        self.hidden_dim = 256
        self.num_classes = 10
        self.lambda_orthogonality = 0.1
        self.M = 1.0
        self.t = 5
        self.num_layers = 2
        self.dropout_prob = 0.3
        self.activation = "relu"
        self.use_batch_norm = True
        self.matmul_input = "g"
        self.gumbel_softmax_log = False
        self._debug = False

        # Initialize model
        self.model = MUSEFrameAxisUnsupervised(
            embedding_dim=self.embedding_dim,
            frameaxis_dim=self.frameaxis_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            lambda_orthogonality=self.lambda_orthogonality,
            M=self.M,
            t=self.t,
            num_layers=self.num_layers,
            dropout_prob=self.dropout_prob,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            matmul_input=self.matmul_input,
            gumbel_softmax_log=self.gumbel_softmax_log,
            _debug=self._debug,
        )

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward_pass(self):
        batch_size = 32
        v_sentence = torch.randn(batch_size, self.embedding_dim).to(self.device)
        v_fx = torch.randn(batch_size, self.frameaxis_dim).to(self.device)
        fx_negatives = torch.randn(10, self.frameaxis_dim).to(
            self.device
        )  # Example: 10 negatives
        tau = 0.5
        mask = torch.ones(batch_size, dtype=torch.float32).to(self.device)

        # Run the forward pass
        results = self.model(
            v_fx, mask, v_sentence, fx_negatives, tau, mixed_precision="fp16"
        )

        # Check if results contain expected keys
        self.assertIn("loss", results)
        self.assertIn("fx", results)

        # Ensure loss tensor shape and values are as expected
        loss = results["loss"]
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        self.assertTrue((loss > 0).all())

    def test_forward_pass_cpu(self):
        batch_size = 32
        v_sentence = torch.randn(batch_size, self.embedding_dim)
        v_fx = torch.randn(batch_size, self.frameaxis_dim)
        fx_negatives = torch.randn(10, self.frameaxis_dim)  # Example: 10 negatives
        tau = 0.5
        mask = torch.ones(batch_size, dtype=torch.float32)

        # Move model to CPU
        self.model.to("cpu")

        # Run the forward pass
        results = self.model(
            v_fx, mask, v_sentence, fx_negatives, tau, mixed_precision="fp32"
        )

        # Check if results contain expected keys
        self.assertIn("loss", results)
        self.assertIn("fx", results)

        # Ensure loss tensor shape and values are as expected
        loss = results["loss"]
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        self.assertTrue((loss > 0).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_forward_pass_mixed_precision(self):
        batch_size = 32
        v_sentence = torch.randn(batch_size, self.embedding_dim).to(self.device)
        v_fx = torch.randn(batch_size, self.frameaxis_dim).to(self.device)
        fx_negatives = torch.randn(10, self.frameaxis_dim).to(
            self.device
        )  # Example: 10 negatives
        tau = 0.5

        # Test for different mixed precision settings
        for precision in ["fp16", "bf16", "fp32"]:
            with self.subTest(precision=precision):
                results = self.model(
                    v_sentence, v_fx, fx_negatives, tau, mixed_precision=precision
                )

                # Check if results contain expected keys
                self.assertIn("loss", results)
                self.assertIn("fx", results)

                # Ensure loss tensor shape and values are as expected
                loss = results["loss"]
                self.assertFalse(torch.isnan(loss).any())
                self.assertFalse(torch.isinf(loss).any())
                self.assertTrue((loss > 0).all())

    def test_half_empty_batch(self):
        # Example inputs with half of the embeddings being zero
        batch_size = 32
        v_sentence = torch.cat(
            [
                torch.randn(batch_size // 2, self.embedding_dim),
                torch.zeros(batch_size // 2, self.embedding_dim),
            ],
            dim=0,
        ).to(self.device)
        v_fx = torch.cat(
            [
                torch.randn(batch_size // 2, self.frameaxis_dim),
                torch.zeros(batch_size // 2, self.frameaxis_dim),
            ],
            dim=0,
        ).to(self.device)
        fx_negatives = torch.randn(10, self.frameaxis_dim).to(self.device)
        tau = 0.5
        mask = torch.cat(
            [torch.ones(batch_size // 2), torch.zeros(batch_size // 2)]
        ).to(self.device)

        # Run the forward pass
        results = self.model(
            v_fx, mask, v_sentence, fx_negatives, tau, mixed_precision="fp16"
        )

        # Check if results contain expected keys
        self.assertIn("loss", results)
        self.assertIn("fx", results)

        # Ensure loss tensor shape and values are as expected
        loss = results["loss"]
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        self.assertTrue((loss >= 0).all())

    def test_full_empty_batch(self):
        # Example inputs with all embeddings being zero
        batch_size = 32
        v_sentence = torch.zeros(batch_size, self.embedding_dim).to(self.device)
        v_fx = torch.zeros(batch_size, self.frameaxis_dim).to(self.device)
        fx_negatives = torch.randn(10, self.frameaxis_dim).to(self.device)
        tau = 0.5
        mask = torch.zeros(batch_size).to(self.device)

        # Run the forward pass
        results = self.model(
            v_fx, mask, v_sentence, fx_negatives, tau, mixed_precision="fp16"
        )

        # Check if results contain expected keys
        self.assertIn("loss", results)
        self.assertIn("fx", results)

        # Ensure loss tensor shape and values are as expected
        loss = results["loss"]
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())


if __name__ == "__main__":
    unittest.main()
