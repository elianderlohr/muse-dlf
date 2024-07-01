import os
import sys
import unittest
import torch
import torch.nn as nn

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, src_dir)

from src.model.slmuse_dlf.supervised_module import MUSESupervised


class TestMUSESupervised(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMUSESupervised, self).__init__(*args, **kwargs)

        # Initialize parameters for testing
        self.embedding_dim = 768
        self.num_classes = 10
        self.frameaxis_dim = 64
        self.num_sentences = 16
        self.dropout_prob = 0.3
        self.concat_frameaxis = True
        self.num_layers = 3
        self.activation_function = "relu"
        self._debug = False

        # Initialize model
        self.model = MUSESupervised(
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            frameaxis_dim=self.frameaxis_dim,
            num_sentences=self.num_sentences,
            dropout_prob=self.dropout_prob,
            concat_frameaxis=self.concat_frameaxis,
            num_layers=self.num_layers,
            activation_function=self.activation_function,
            _debug=self._debug,
        )

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_forward_pass(self):
        batch_size = 32
        num_args = 3  # number of arguments (predicate, arg0, arg1)
        d_p = torch.randn(
            batch_size, self.num_sentences, num_args, self.num_classes
        ).to(self.device)
        d_a0 = torch.randn(
            batch_size, self.num_sentences, num_args, self.num_classes
        ).to(self.device)
        d_a1 = torch.randn(
            batch_size, self.num_sentences, num_args, self.num_classes
        ).to(self.device)
        d_fx = torch.randn(batch_size, self.num_sentences, self.num_classes).to(
            self.device
        )
        vs = torch.randn(batch_size, self.num_sentences, self.embedding_dim).to(
            self.device
        )
        frameaxis_data = torch.randn(
            batch_size, self.num_sentences, self.frameaxis_dim
        ).to(self.device)

        # Run the forward pass
        y_hat_u, y_hat_s, combined, other = self.model(
            d_p, d_a0, d_a1, d_fx, vs, frameaxis_data, mixed_precision="fp32"
        )

        # Ensure outputs have the correct shape and values are as expected
        self.assertEqual(y_hat_u.shape, (batch_size, self.num_classes))
        self.assertEqual(y_hat_s.shape, (batch_size, self.num_classes))
        self.assertEqual(combined.shape, (batch_size, self.num_classes))

        for key in ["predicate", "arg0", "arg1", "frameaxis"]:
            self.assertIn(key, other)
            self.assertEqual(other[key].shape, (batch_size, self.num_classes))

        # Ensure no NaNs or infinities in the outputs
        self.assertFalse(torch.isnan(y_hat_u).any())
        self.assertFalse(torch.isinf(y_hat_u).any())
        self.assertFalse(torch.isnan(y_hat_s).any())
        self.assertFalse(torch.isinf(y_hat_s).any())
        self.assertFalse(torch.isnan(combined).any())
        self.assertFalse(torch.isinf(combined).any())

    def test_forward_pass_cpu(self):
        batch_size = 32
        num_args = 3  # number of arguments (predicate, arg0, arg1)
        d_p = torch.randn(batch_size, self.num_sentences, num_args, self.num_classes)
        d_a0 = torch.randn(batch_size, self.num_sentences, num_args, self.num_classes)
        d_a1 = torch.randn(batch_size, self.num_sentences, num_args, self.num_classes)
        d_fx = torch.randn(batch_size, self.num_sentences, self.num_classes)
        vs = torch.randn(batch_size, self.num_sentences, self.embedding_dim)
        frameaxis_data = torch.randn(batch_size, self.num_sentences, self.frameaxis_dim)

        # Move model to CPU
        self.model.to("cpu")

        # Run the forward pass
        y_hat_u, y_hat_s, combined, other = self.model(
            d_p, d_a0, d_a1, d_fx, vs, frameaxis_data, mixed_precision="fp32"
        )

        # Ensure outputs have the correct shape and values are as expected
        self.assertEqual(y_hat_u.shape, (batch_size, self.num_classes))
        self.assertEqual(y_hat_s.shape, (batch_size, self.num_classes))
        self.assertEqual(combined.shape, (batch_size, self.num_classes))

        for key in ["predicate", "arg0", "arg1", "frameaxis"]:
            self.assertIn(key, other)
            self.assertEqual(other[key].shape, (batch_size, self.num_classes))

        # Ensure no NaNs or infinities in the outputs
        self.assertFalse(torch.isnan(y_hat_u).any())
        self.assertFalse(torch.isinf(y_hat_u).any())
        self.assertFalse(torch.isnan(y_hat_s).any())
        self.assertFalse(torch.isinf(y_hat_s).any())
        self.assertFalse(torch.isnan(combined).any())
        self.assertFalse(torch.isinf(combined).any())

    def test_padded_sentence(self):
        batch_size = 8
        num_args = 3
        d_p = torch.randn(batch_size, self.num_sentences, num_args, self.num_classes)
        d_a0 = torch.randn(batch_size, self.num_sentences, num_args, self.num_classes)
        d_a1 = torch.randn(batch_size, self.num_sentences, num_args, self.num_classes)
        d_fx = torch.randn(batch_size, self.num_sentences, self.num_classes)
        frameaxis_data = torch.randn(batch_size, self.num_sentences, self.frameaxis_dim)
        vs = torch.randn(batch_size, self.num_sentences, self.embedding_dim)

        # Set the last 5 sentences to be padded
        d_p[:, -5:, :, :] = 0
        d_a0[:, -5:, :, :] = 0
        d_a1[:, -5:, :, :] = 0
        d_fx[:, -5:, :] = 0
        vs[:, -5:, :] = 0
        frameaxis_data[:, -5:, :] = 0

        # Set for the 2 batch the last -7 sentences to be padded
        d_p[2, -7:, :, :] = 0
        d_a0[2, -7:, :, :] = 0
        d_a1[2, -7:, :, :] = 0
        d_fx[2, -7:, :] = 0
        vs[2, -7:, :] = 0
        frameaxis_data[2, -7:, :] = 0

        # Run the forward pass
        y_hat_u, y_hat_s, combined, other = self.model(
            d_p, d_a0, d_a1, d_fx, vs, frameaxis_data, mixed_precision="fp32"
        )

        # Ensure outputs have the correct shape and values are as expected
        self.assertEqual(y_hat_u.shape, (batch_size, self.num_classes))
        self.assertEqual(y_hat_s.shape, (batch_size, self.num_classes))
        self.assertEqual(combined.shape, (batch_size, self.num_classes))

        for key in ["predicate", "arg0", "arg1", "frameaxis"]:
            self.assertIn(key, other)
            self.assertEqual(other[key].shape, (batch_size, self.num_classes))


if __name__ == "__main__":
    unittest.main()
