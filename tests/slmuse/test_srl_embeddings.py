import os
import sys
import unittest
import torch
from transformers import RobertaTokenizerFast

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, src_dir)

from src.model.slmuse_dlf.srl_embeddings import (
    SRLEmbeddings,
)  # Replace with correct import path


class TestSRLEmbeddings(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSRLEmbeddings, self).__init__(*args, **kwargs)

        # Initialize parameters for testing
        self.model_name_or_path = "roberta-base"
        self.model_type = "roberta-base"
        self.pooling = "mean"
        self._debug = False

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Example input shapes
        self.batch_size = 2
        self.num_sentences = 3
        self.sentence_len = 10
        self.num_args = 2
        self.arg_len = 5

    def setUp(self):
        # Initialize model
        self.srl = SRLEmbeddings(
            model_name_or_path=self.model_name_or_path,
            model_type=self.model_type,
            pooling=self.pooling,
            _debug=self._debug,
        ).to(self.device)

    def test_initialization_bert(self):
        # Test initialization with BERT model
        self.assertIsInstance(self.srl, SRLEmbeddings)
        self.assertEqual(self.srl.model_type, "roberta-base")
        self.assertEqual(self.srl.pooling, "mean")

    def test_initialization_roberta(self):
        # Test initialization with RoBERTa model
        model_name_or_path = "roberta-base"
        srl = SRLEmbeddings(
            model_name_or_path=model_name_or_path,
            model_type="roberta-base",
            pooling="cls",
            _debug=self._debug,
        ).to(self.device)
        self.assertIsInstance(srl, SRLEmbeddings)
        self.assertEqual(srl.model_type, "roberta-base")
        self.assertEqual(srl.pooling, "cls")

    def test_model_loading(self):
        # Test model loading and verification
        self.srl.verify_model_loading()

    def test_get_sentence_embedding(self):
        # Test get_sentence_embedding method
        # Example inputs
        ids = torch.randint(
            0, 100, (self.batch_size, self.num_sentences, self.sentence_len)
        ).to(self.device)
        attention_masks = torch.ones_like(ids).to(self.device)

        embeddings, embeddings_mean = self.srl.get_sentence_embedding(
            ids, attention_masks
        )

        # Check shapes
        self.assertEqual(
            embeddings.shape,
            (
                self.batch_size,
                self.num_sentences,
                self.sentence_len,
                self.srl.embedding_dim,
            ),
        )
        self.assertEqual(
            embeddings_mean.shape,
            (self.batch_size, self.num_sentences, self.srl.embedding_dim),
        )

    def test_get_arg_embedding(self):
        # Test get_arg_embedding method
        # Example inputs
        arg_ids = torch.randint(
            0, 100, (self.batch_size, self.num_sentences, self.num_args, self.arg_len)
        ).to(self.device)
        sentence_ids = torch.randint(
            0, 100, (self.batch_size, self.num_sentences, self.sentence_len)
        ).to(self.device)
        sentence_embeddings = torch.randn(
            self.batch_size,
            self.num_sentences,
            self.sentence_len,
            self.srl.embedding_dim,
        ).to(self.device)

        arg_embeddings = self.srl.get_arg_embedding(
            arg_ids, sentence_ids, sentence_embeddings
        )

        # Check shapes
        self.assertEqual(
            arg_embeddings.shape,
            (
                self.batch_size,
                self.num_sentences,
                self.num_args,
                self.srl.embedding_dim,
            ),
        )

    def test_forward(self):
        # Test forward method
        # Example inputs
        sentence_ids = torch.randint(
            0, 100, (self.batch_size, self.num_sentences, self.sentence_len)
        ).to(self.device)
        sentence_attention_masks = torch.ones_like(sentence_ids).to(self.device)
        predicate_ids = torch.randint(
            0, 100, (self.batch_size, self.num_sentences, self.num_args, self.arg_len)
        ).to(self.device)
        arg0_ids = torch.randint(
            0, 100, (self.batch_size, self.num_sentences, self.num_args, self.arg_len)
        ).to(self.device)
        arg1_ids = torch.randint(
            0, 100, (self.batch_size, self.num_sentences, self.num_args, self.arg_len)
        ).to(self.device)

        embeddings_avg, predicate_embeddings, arg0_embeddings, arg1_embeddings = (
            self.srl.forward(
                sentence_ids,
                sentence_attention_masks,
                predicate_ids,
                arg0_ids,
                arg1_ids,
            )
        )

        # Check shapes
        self.assertEqual(
            embeddings_avg.shape,
            (self.batch_size, self.num_sentences, self.srl.embedding_dim),
        )
        self.assertEqual(
            predicate_embeddings.shape,
            (
                self.batch_size,
                self.num_sentences,
                self.num_args,
                self.srl.embedding_dim,
            ),
        )
        self.assertEqual(
            arg0_embeddings.shape,
            (
                self.batch_size,
                self.num_sentences,
                self.num_args,
                self.srl.embedding_dim,
            ),
        )
        self.assertEqual(
            arg1_embeddings.shape,
            (
                self.batch_size,
                self.num_sentences,
                self.num_args,
                self.srl.embedding_dim,
            ),
        )

    def test_sentence_embeddings_avg_mean_pooling(self):
        # Test if sentence_embeddings_avg is exactly the average embedding with mean pooling
        # Example inputs
        ids = torch.randint(
            0, 100, (self.batch_size, self.num_sentences, self.sentence_len)
        ).to(self.device)
        attention_masks = torch.ones_like(ids).to(self.device)

        embeddings, embeddings_mean = self.srl.get_sentence_embedding(
            ids, attention_masks
        )

        # Get the first sentence embeddings (assuming batch_size=2, num_sentences=3 for simplicity)
        sentence_embeddings_avg = embeddings_mean[0, 0]  # First sentence, first token

        # Calculate the expected average embedding manually
        expected_avg_embedding = torch.mean(embeddings[0, 0], dim=0)

        # Compare the calculated average embedding with the returned sentence_embeddings_avg
        self.assertTrue(torch.equal(sentence_embeddings_avg, expected_avg_embedding))

    def test_sentence_embeddings_avg_cls_pooling(self):
        # Test if sentence_embeddings_avg is exactly the CLS token embedding with cls pooling
        # Change pooling to 'cls'
        self.srl.pooling = "cls"

        # Example inputs
        ids = torch.randint(
            0, 100, (self.batch_size, self.num_sentences, self.sentence_len)
        ).to(self.device)
        attention_masks = torch.ones_like(ids).to(self.device)

        embeddings, embeddings_mean = self.srl.get_sentence_embedding(
            ids, attention_masks
        )

        # Get the CLS token embedding
        sentence_embeddings_avg = embeddings_mean[0, 0]  # First sentence, CLS token

        # CLS token embedding should be the first token's embedding
        cls_token_embedding = embeddings[0, 0, 0]

        # Compare the CLS token embedding with the returned sentence_embeddings_avg
        self.assertTrue(torch.equal(sentence_embeddings_avg, cls_token_embedding))

    def test_get_sentence_embedding(self):
        # Initialize parameters for testing
        model_name_or_path = "roberta-base"
        model_type = "roberta-base"
        pooling = "mean"
        _debug = False

        # Example input
        batch_size = 2
        num_sentences = 3
        sentence_len = 10

        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        srl = SRLEmbeddings(
            model_name_or_path=model_name_or_path,
            model_type=model_type,
            pooling=pooling,
            _debug=_debug,
        ).to(device)

        # Example input tensors
        ids = torch.randint(0, 100, (batch_size, num_sentences, sentence_len)).to(
            device
        )
        attention_masks = torch.ones_like(ids).to(device)

        # Get embeddings
        embeddings, embeddings_mean = srl.get_sentence_embedding(ids, attention_masks)

        # Assertions
        assert embeddings.shape == (
            batch_size,
            num_sentences,
            sentence_len,
            srl.embedding_dim,
        )
        assert embeddings_mean.shape == (batch_size, num_sentences, srl.embedding_dim)

        # Print some debug info if needed
        if _debug:
            print("Example input ids:", ids)
            print("Example embeddings shape:", embeddings.shape)
            print("Example embeddings mean shape:", embeddings_mean.shape)

    def test_get_sentence_embedding(self):
        # Test get_sentence_embedding method
        # Example inputs
        ids = torch.randint(
            0, 100, (self.batch_size, self.num_sentences, self.sentence_len)
        ).to(self.device)
        attention_masks = torch.ones_like(ids).to(self.device)

        embeddings, embeddings_mean = self.srl.get_sentence_embedding(
            ids, attention_masks
        )

        # Check shapes
        self.assertEqual(
            embeddings.shape,
            (
                self.batch_size,
                self.num_sentences,
                self.sentence_len,
                self.srl.embedding_dim,
            ),
        )
        self.assertEqual(
            embeddings_mean.shape,
            (self.batch_size, self.num_sentences, self.srl.embedding_dim),
        )

    def test_get_arg_embedding(self):
        # Test get_arg_embedding method

        # Example sentence and argument ids
        sentence_ids = torch.tensor([[[1, 51, 515, 212, 121]]]).to(
            self.device
        )  # shape: (1, 1, 5)
        attention_masks = torch.ones_like(sentence_ids).to(self.device)
        arg_ids = torch.tensor([[[[212, 121]]]]).to(self.device)  # shape: (1, 1, 1, 2)

        # Get sentence embeddings
        sentence_embeddings, _ = self.srl.get_sentence_embedding(
            sentence_ids, attention_masks
        )

        # Get argument embeddings
        arg_embeddings = self.srl.get_arg_embedding(
            arg_ids, sentence_ids, sentence_embeddings
        )

        # Check shapes
        self.assertEqual(
            arg_embeddings.shape,
            (
                1,
                1,
                1,
                self.srl.embedding_dim,
            ),
        )

        # Expected embeddings
        expected_arg_embedding = sentence_embeddings[0, 0, 3:5].mean(dim=0)

        # Assert that the embeddings are correctly retrieved
        self.assertTrue(
            torch.equal(arg_embeddings[0, 0, 0], expected_arg_embedding),
            "Embedding for arg is incorrect",
        )


if __name__ == "__main__":
    unittest.main()
