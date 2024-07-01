import os
import sys
import unittest
import torch
from torch.nn.functional import log_softmax, softmax

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, src_dir)

from src.model.slmuse_dlf.combined_autoencoder import CombinedAutoencoder


class TestCombinedAutoencoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCombinedAutoencoder, self).__init__(*args, **kwargs)

        # Initialize parameters for testing
        self.embedding_dim = 768
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
        self.model = CombinedAutoencoder(
            embedding_dim=self.embedding_dim,
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
        v_p = torch.randn(batch_size, self.embedding_dim).to(self.device)
        v_a0 = torch.randn(batch_size, self.embedding_dim).to(self.device)
        v_a1 = torch.randn(batch_size, self.embedding_dim).to(self.device)
        v_sentence = torch.randn(batch_size, self.embedding_dim).to(self.device)
        tau = 0.5
        mask_p = torch.ones(batch_size, dtype=torch.float32).to(self.device)
        mask_a0 = torch.ones(batch_size, dtype=torch.float32).to(self.device)
        mask_a1 = torch.ones(batch_size, dtype=torch.float32).to(self.device)

        output = self.model(
            v_p=v_p,
            v_a0=v_a0,
            v_a1=v_a1,
            mask_p=mask_p,
            mask_a0=mask_a0,
            mask_a1=mask_a1,
            v_sentence=v_sentence,
            tau=tau,
            mixed_precision="fp32",
        )

        # Assertions based on expected output structure
        self.assertIn("p", output)
        self.assertIn("a0", output)
        self.assertIn("a1", output)

        # Example assertions on output shapes
        for view in ["p", "a0", "a1"]:
            self.assertIn("vhat", output[view])
            self.assertIn("d", output[view])
            self.assertIn("g", output[view])
            self.assertIn("F", output[view])

            # Check shapes
            self.assertEqual(
                output[view]["vhat"].size(), (batch_size, self.embedding_dim)
            )
            self.assertEqual(output[view]["d"].size(), (batch_size, self.num_classes))
            self.assertEqual(output[view]["g"].size(), (batch_size, self.num_classes))
            self.assertEqual(
                output[view]["F"].size(), (self.num_classes, self.embedding_dim)
            )

    def test_no_nans_or_zeros(self):
        # Define batch size for this test
        batch_size = 32

        # Example input tensors
        v_p = torch.randn(batch_size, self.embedding_dim).to(self.device)
        v_a0 = torch.randn(batch_size, self.embedding_dim).to(self.device)
        v_a1 = torch.randn(batch_size, self.embedding_dim).to(self.device)
        v_sentence = torch.randn(batch_size, self.embedding_dim).to(self.device)
        tau = 0.5
        mask_p = torch.ones(batch_size, dtype=torch.float32).to(self.device)
        mask_a0 = torch.ones(batch_size, dtype=torch.float32).to(self.device)
        mask_a1 = torch.ones(batch_size, dtype=torch.float32).to(self.device)

        output = self.model(
            v_p=v_p,
            v_a0=v_a0,
            v_a1=v_a1,
            mask_p=mask_p,
            mask_a0=mask_a0,
            mask_a1=mask_a1,
            v_sentence=v_sentence,
            tau=tau,
            mixed_precision="fp32",
        )

        # Check for NaNs in each output tensor
        for view in ["p", "a0", "a1"]:
            for tensor_name, tensor_data in output[view].items():
                self.assertFalse(
                    torch.isnan(tensor_data).any(),
                    f"NaNs found in {view}/{tensor_name}",
                )

                if tensor_data.numel() > 0:
                    self.assertFalse(
                        (tensor_data == 0).all(),
                        f"All zeros found in {view}/{tensor_name}",
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
            model = CombinedAutoencoder(
                embedding_dim=self.embedding_dim,
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

    def test_half_empty_batch(self):
        # Example inputs with half of the embeddings being zero
        batch_size = 32
        v_p = torch.cat(
            [
                torch.randn(batch_size // 2, self.embedding_dim),
                torch.zeros(batch_size // 2, self.embedding_dim),
            ],
            dim=0,
        ).to(self.device)
        v_a0 = torch.cat(
            [
                torch.randn(batch_size // 2, self.embedding_dim),
                torch.zeros(batch_size // 2, self.embedding_dim),
            ],
            dim=0,
        ).to(self.device)
        v_a1 = torch.cat(
            [
                torch.randn(batch_size // 2, self.embedding_dim),
                torch.zeros(batch_size // 2, self.embedding_dim),
            ],
            dim=0,
        ).to(self.device)
        v_sentence = torch.cat(
            [
                torch.randn(batch_size // 2, self.embedding_dim),
                torch.zeros(batch_size // 2, self.embedding_dim),
            ],
            dim=0,
        ).to(self.device)
        tau = 0.5
        mask_p = torch.cat(
            [
                torch.ones(batch_size // 2, dtype=torch.float32),
                torch.zeros(batch_size // 2, dtype=torch.float32),
            ],
            dim=0,
        ).to(self.device)
        mask_a0 = torch.cat(
            [
                torch.ones(batch_size // 2, dtype=torch.float32),
                torch.zeros(batch_size // 2, dtype=torch.float32),
            ],
            dim=0,
        ).to(self.device)
        mask_a1 = torch.cat(
            [
                torch.ones(batch_size // 2, dtype=torch.float32),
                torch.zeros(batch_size // 2, dtype=torch.float32),
            ],
            dim=0,
        ).to(self.device)

        output = self.model(
            v_p=v_p,
            v_a0=v_a0,
            v_a1=v_a1,
            mask_p=mask_p,
            mask_a0=mask_a0,
            mask_a1=mask_a1,
            v_sentence=v_sentence,
            tau=tau,
            mixed_precision="fp32",
        )

        # Check for NaNs in each output tensor
        for view in ["p", "a0", "a1"]:
            for tensor_name, tensor_data in output[view].items():
                self.assertFalse(
                    torch.isnan(tensor_data).any(),
                    f"NaNs found in {view}/{tensor_name}",
                )

                if tensor_data.numel() > 0:
                    self.assertFalse(
                        (tensor_data == 0).all(),
                        f"All zeros found in {view}/{tensor_name}",
                    )

    def test_full_empty_batch(self):
        # Example inputs with all embeddings being zero
        batch_size = 32
        v_p = torch.zeros(batch_size, self.embedding_dim).to(self.device)
        v_a0 = torch.zeros(batch_size, self.embedding_dim).to(self.device)
        v_a1 = torch.zeros(batch_size, self.embedding_dim).to(self.device)
        v_sentence = torch.zeros(batch_size, self.embedding_dim).to(self.device)
        tau = 0.5
        mask_p = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
        mask_a0 = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
        mask_a1 = torch.zeros(batch_size, dtype=torch.float32).to(self.device)

        output = self.model(
            v_p=v_p,
            v_a0=v_a0,
            v_a1=v_a1,
            mask_p=mask_p,
            mask_a0=mask_a0,
            mask_a1=mask_a1,
            v_sentence=v_sentence,
            tau=tau,
            mixed_precision="fp32",
        )

        # Check for NaNs in each output tensor
        for view in ["p", "a0", "a1"]:
            for tensor_name, tensor_data in output[view].items():
                self.assertFalse(
                    torch.isnan(tensor_data).any(),
                    f"NaNs found in {view}/{tensor_name}",
                )

                if tensor_data.numel() > 0:
                    self.assertTrue(
                        (tensor_data == 0).all(),
                        f"Not all zeros found in {view}/{tensor_name}",
                    )


if __name__ == "__main__":
    unittest.main()
