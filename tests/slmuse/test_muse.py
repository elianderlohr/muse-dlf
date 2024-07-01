import os
import sys
import unittest
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, src_dir)

from src.model.slmuse_dlf.muse import MUSEDLF


class TestMUSEDLF(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMUSEDLF, self).__init__(*args, **kwargs)

        # Initialize parameters for testing
        self.embedding_dim = 768
        self.frameaxis_dim = 10
        self.hidden_dim = 256
        self.num_classes = 15
        self.num_sentences = 16
        self.dropout_prob = 0.3
        self.bert_model_name = "roberta-base"
        self.bert_model_name_or_path = "roberta-base"
        self.srl_embeddings_pooling = "mean"
        self.lambda_orthogonality = 1e-3
        self.M = 8
        self.t = 8
        self.muse_unsupervised_num_layers = 2
        self.muse_unsupervised_activation = "relu"
        self.muse_unsupervised_use_batch_norm = True
        self.muse_unsupervised_matmul_input = "g"
        self.muse_unsupervised_gumbel_softmax_log = False
        self.muse_frameaxis_unsupervised_num_layers = 2
        self.muse_frameaxis_unsupervised_activation = "relu"
        self.muse_frameaxis_unsupervised_use_batch_norm = True
        self.muse_frameaxis_unsupervised_matmul_input = "g"
        self.muse_frameaxis_unsupervised_gumbel_softmax_log = False
        self.num_negatives = -1
        self.supervised_concat_frameaxis = True
        self.supervised_num_layers = 2
        self.supervised_activation = "relu"
        self._debug = False

        self.num_args = 8
        self.max_args_length = 8

        # Initialize model
        self.model = MUSEDLF(
            embedding_dim=self.embedding_dim,
            frameaxis_dim=self.frameaxis_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_sentences=self.num_sentences,
            dropout_prob=self.dropout_prob,
            bert_model_name=self.bert_model_name,
            bert_model_name_or_path=self.bert_model_name_or_path,
            srl_embeddings_pooling=self.srl_embeddings_pooling,
            lambda_orthogonality=self.lambda_orthogonality,
            M=self.M,
            t=self.t,
            muse_unsupervised_num_layers=self.muse_unsupervised_num_layers,
            muse_unsupervised_activation=self.muse_unsupervised_activation,
            muse_unsupervised_use_batch_norm=self.muse_unsupervised_use_batch_norm,
            muse_unsupervised_matmul_input=self.muse_unsupervised_matmul_input,
            muse_unsupervised_gumbel_softmax_log=self.muse_unsupervised_gumbel_softmax_log,
            muse_frameaxis_unsupervised_num_layers=self.muse_frameaxis_unsupervised_num_layers,
            muse_frameaxis_unsupervised_activation=self.muse_frameaxis_unsupervised_activation,
            muse_frameaxis_unsupervised_use_batch_norm=self.muse_frameaxis_unsupervised_use_batch_norm,
            muse_frameaxis_unsupervised_matmul_input=self.muse_frameaxis_unsupervised_matmul_input,
            muse_frameaxis_unsupervised_gumbel_softmax_log=self.muse_frameaxis_unsupervised_gumbel_softmax_log,
            num_negatives=self.num_negatives,
            supervised_concat_frameaxis=self.supervised_concat_frameaxis,
            supervised_num_layers=self.supervised_num_layers,
            supervised_activation=self.supervised_activation,
            _debug=self._debug,
        )

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_forward(self):
        # Generate random input data
        batch_size = 8
        seq_length = 128
        sentence_ids = torch.randint(
            0, 30522, (batch_size, self.num_sentences, seq_length)
        ).to(self.device)
        sentence_attention_masks = torch.ones(
            (batch_size, self.num_sentences, seq_length)
        ).to(self.device)
        predicate_ids = torch.randint(
            0,
            30522,
            (batch_size, self.num_sentences, self.num_args, self.max_args_length),
        ).to(self.device)
        arg0_ids = torch.randint(
            0,
            30522,
            (batch_size, self.num_sentences, self.num_args, self.max_args_length),
        ).to(self.device)
        arg1_ids = torch.randint(
            0,
            30522,
            (batch_size, self.num_sentences, self.num_args, self.max_args_length),
        ).to(self.device)
        frameaxis_data = torch.randn(
            (batch_size, self.num_sentences, self.frameaxis_dim)
        ).to(self.device)
        tau = 0.1

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            unsupervised_loss, span_pred, sentence_pred, combined_pred, other = (
                self.model(
                    sentence_ids,
                    sentence_attention_masks,
                    predicate_ids,
                    arg0_ids,
                    arg1_ids,
                    frameaxis_data,
                    tau,
                    mixed_precision="fp16",
                )
            )

        # Check the output types and shapes
        self.assertIsInstance(unsupervised_loss, torch.Tensor)
        self.assertIsInstance(span_pred, torch.Tensor)
        self.assertIsInstance(sentence_pred, torch.Tensor)
        self.assertIsInstance(combined_pred, torch.Tensor)
        self.assertIsInstance(other, dict)
        self.assertEqual(unsupervised_loss.shape, torch.Size([]))


if __name__ == "__main__":
    unittest.main()
