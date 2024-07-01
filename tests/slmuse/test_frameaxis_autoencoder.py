import os
import sys
import unittest
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, softmax

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, src_dir)


from src.model.slmuse_dlf.frameaxis_autoencoder import FrameAxisAutoencoder


class TestFrameAxisAutoencoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFrameAxisAutoencoder, self).__init__(*args, **kwargs)

        # Initialize parameters for testing
        self.embedding_dim = 768
        self.frameaxis_dim = 20
        self.hidden_dim = 256
        self.num_classes = 10
        self.num_layers = 2
        self.dropout_prob = 0.3
        self.activation = "relu"
        self.use_batch_norm = True
        self.matmul_input = "g"
        self.log = False
        self._debug = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = FrameAxisAutoencoder(
            embedding_dim=self.embedding_dim,
            frameaxis_dim=self.frameaxis_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout_prob=self.dropout_prob,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            matmul_input=self.matmul_input,
            log=self.log,
            _debug=self._debug,
        ).to(self.device)

    def test_foward_pass_shapes(self):
        # Define batch size for this test
        batch_size = 32

        # Example input tensors
        v_frameaxis = torch.randn(batch_size, self.frameaxis_dim).to(self.device)
        v_sentence = torch.randn(batch_size, self.embedding_dim).to(self.device)
        mask = torch.ones(batch_size, dtype=torch.float32).to(self.device)
        tau = 0.5

        output = self.model(
            v_frameaxis=v_frameaxis,
            mask=mask,
            v_sentence=v_sentence,
            tau=tau,
            mixed_precision="fp32",
        )

        # Example assertions on output shapes
        self.assertIn("vhat", output)
        self.assertIn("d", output)
        self.assertIn("g", output)
        self.assertIn("F", output)

        # Check shapes
        self.assertEqual(output["vhat"].size(), (batch_size, self.frameaxis_dim))
        self.assertEqual(output["d"].size(), (batch_size, self.num_classes))
        self.assertEqual(output["g"].size(), (batch_size, self.num_classes))
        self.assertEqual(output["F"].size(), (self.num_classes, self.frameaxis_dim))

    def test_no_nans_or_zeros(self):
        # Define batch size for this test
        batch_size = 32

        # Example input tensors
        v_frameaxis = torch.randn(batch_size, self.frameaxis_dim).to(self.device)
        v_sentence = torch.randn(batch_size, self.embedding_dim).to(self.device)
        mask = torch.ones(batch_size, dtype=torch.float32).to(self.device)

        # Forward pass
        output = self.model(
            v_frameaxis=v_frameaxis,
            mask=mask,
            v_sentence=v_sentence,
            tau=0.5,
            mixed_precision="fp32",
        )

        # Check for NaNs
        for tensor_name, tensor_data in output.items():
            self.assertFalse(
                torch.isnan(tensor_data).any(),
                f"NaNs found in {tensor_name}",
            )

            if tensor_data.numel() > 0:
                self.assertFalse(
                    (tensor_data == 0).all(),
                    f"All zeros found in /{tensor_name}",
                )

    def test_activation_function_selection(self):
        # Test if the correct activation function is selected based on the input parameter
        activation_functions = {
            "relu": torch.nn.ReLU,
            "leaky_relu": torch.nn.LeakyReLU,
            "gelu": torch.nn.GELU,
            "elu": torch.nn.ELU,
        }

        for activation_name, activation_class in activation_functions.items():
            model = FrameAxisAutoencoder(
                embedding_dim=self.embedding_dim,
                frameaxis_dim=self.frameaxis_dim,
                hidden_dim=self.hidden_dim,
                num_classes=self.num_classes,
                num_layers=self.num_layers,
                dropout_prob=self.dropout_prob,
                activation=activation_name,
                use_batch_norm=self.use_batch_norm,
                matmul_input=self.matmul_input,
                log=self.log,
                _debug=self._debug,
            ).to(self.device)

            # Ensure the activation function in the model matches the expected class
            self.assertIsInstance(model.activation_func, activation_class)


if __name__ == "__main__":
    unittest.main()
