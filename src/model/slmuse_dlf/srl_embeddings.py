import pandas as pd
from transformers import BertModel, RobertaModel
import torch.nn as nn
import torch

from utils.logging_manager import LoggerManager


class SRLEmbeddings(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        model_type: str = "bert-base-uncased",
        pooling: str = "mean",
        _debug=False,
    ):
        super(SRLEmbeddings, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.model_name_or_path = model_name_or_path
        self.model_type = model_type

        if model_type == "bert-base-uncased":
            self.model = BertModel.from_pretrained(model_name_or_path)
        elif model_type == "roberta-base":
            self.model = RobertaModel.from_pretrained(model_name_or_path)
        else:
            raise ValueError(
                f"Unsupported model_type. Choose either 'bert-base-uncased' or 'roberta-base'. Found: {model_type}"
            )

        # Move model to CUDA if available
        if torch.cuda.is_available():
            self.model.cuda()

        if pooling not in ["mean", "cls"]:
            raise ValueError(
                f"Unsupported pooling type. Choose either 'mean' or 'cls'. Found: {pooling}"
            )

        self.pooling = pooling

        self.embedding_dim = self.model.config.hidden_size

        self._debug = _debug

        # Debugging:
        self.logger.info(f"✅ SRLEmbeddings successfully initialized")

        if self._debug:
            self.verify_model_loading()

    def verify_model_loading(self):
        if self.model_type == "bert-base-uncased":
            model = BertModel.from_pretrained(self.model_name_or_path)
        elif self.model_type == "roberta-base":
            model = RobertaModel.from_pretrained(self.model_name_or_path)
        else:
            raise ValueError(
                f"Unsupported model_type. Choose either 'bert-base-uncased' or 'roberta-base'. Found: {self.model_type}"
            )

        model.eval()

        # Test with some random input
        test_ids = torch.randint(
            0, 100, (1, 1, 10)
        )  # Adjust based on your vocab size and input shape
        test_attention_masks = torch.ones_like(test_ids)

        with torch.no_grad():
            outputs = model(input_ids=test_ids, attention_mask=test_attention_masks)
            embeddings = outputs.last_hidden_state

        if torch.isnan(embeddings).any():
            raise ValueError("NaNs found in test embeddings after loading the model")

        self.logger.info("No NaNs found in test embeddings. Model loading seems fine.")

    def check_for_nans(self, tensor, tensor_name):
        if torch.isnan(tensor).any():
            self.logger.error(f"NaN values detected in {tensor_name}")

    def get_sentence_embedding(self, ids: torch.Tensor, attention_masks: torch.Tensor):
        batch_size, num_sentences, max_sentence_length = ids.shape

        # Reshape the input tensors to combine batch_size and num_sentences dimensions
        ids_flat = ids.view(batch_size * num_sentences, max_sentence_length)
        attention_masks_flat = attention_masks.view(
            batch_size * num_sentences, max_sentence_length
        )

        with torch.no_grad():
            # Obtain the embeddings from the BERT model
            outputs = self.model(
                input_ids=ids_flat, attention_mask=attention_masks_flat
            )
            embeddings = (
                outputs.last_hidden_state
            )  # Assuming the embeddings are in the last_hidden_state

        if torch.isnan(embeddings).any():
            raise ValueError("NaNs found in embeddings after model forward pass")

        # Reshape the embeddings to the desired output shape
        embeddings = embeddings.view(batch_size, num_sentences, max_sentence_length, -1)

        if torch.isnan(embeddings).any():
            raise ValueError("NaNs found in embeddings after reshaping")

        if torch.isnan(embeddings).any():
            # Moving tensor to CPU before performing operations
            embeddings_cpu = embeddings.cpu()

            # Check for NaN values in the tensor on CPU
            nan_mask_cpu = torch.isnan(embeddings_cpu)

            # Count total NaN values
            total_nan_values_cpu = nan_mask_cpu.sum().item()

            # Check if any sentences are completely NaN
            nan_sentences_mask_cpu = nan_mask_cpu.all(dim=-1)
            nan_sentences_cpu = nan_sentences_mask_cpu.sum(dim=-1).numpy()

            # Prepare data for visualization
            nan_details_cpu = {
                "Batch Index": list(range(batch_size)),
                "Total NaN Sentences": nan_sentences_cpu.tolist(),
                "Total NaN Values": [
                    nan_mask_cpu[i].sum().item() for i in range(batch_size)
                ],
            }

            self.logger.error(
                f"Total NaN values in sentence embeddings_reshaped: {total_nan_values_cpu}"
            )

            nan_df_cpu = pd.DataFrame(nan_details_cpu)

            # save df to file as csv
            nan_df_cpu.to_csv("nan_values.csv", index=False)

            # Log NaN values
            self.logger.error(nan_df_cpu.head(32))

        if self.pooling == "mean":
            # Calculate mean embeddings across the token dimension while ignoring padded tokens
            attention_masks_expanded = attention_masks_flat.unsqueeze(-1).expand(
                embeddings.size()
            )
            embeddings_masked = embeddings * attention_masks_expanded
            sum_embeddings = torch.sum(embeddings_masked, dim=2)
            token_counts = (
                attention_masks_flat.sum(dim=1, keepdim=True)
                .clamp(min=1)
                .view(batch_size, num_sentences, 1)
            )
            embeddings_mean = sum_embeddings / token_counts
        elif self.pooling == "cls":
            # Use the [CLS] token representation for the sentence embedding
            embeddings_mean = embeddings[:, :, 0, :]

        # check for NaN values in embeddings_mean
        self.check_for_nans(embeddings_mean, "embeddings_mean")

        return embeddings, embeddings_mean

    def get_arg_embedding(
        self,
        arg_ids: torch.Tensor,
        sentence_ids: torch.Tensor,
        sentence_embeddings: torch.Tensor,
    ):
        batch_size, num_sentences, max_sentence_length = sentence_ids.shape
        _, _, num_args, max_arg_length = arg_ids.shape

        arg_embeddings = torch.zeros(
            batch_size,
            num_sentences,
            num_args,
            self.embedding_dim,
            device=sentence_embeddings.device,
        )

        for batch_idx in range(batch_size):
            for sent_idx in range(num_sentences):
                for arg_idx in range(num_args):
                    for token_idx in range(max_arg_length):
                        arg_token_id = arg_ids[
                            batch_idx, sent_idx, arg_idx, token_idx
                        ].item()
                        if arg_token_id == 0:  # Skip padding tokens
                            continue
                        match_indices = (
                            sentence_ids[batch_idx, sent_idx] == arg_token_id
                        ).nonzero(as_tuple=False)
                        if match_indices.nelement() == 0:
                            continue
                        flat_indices = match_indices[:, 0]
                        selected_embeddings = sentence_embeddings[
                            batch_idx, sent_idx, flat_indices
                        ]
                        avg_embedding = selected_embeddings.mean(dim=0)
                        arg_embeddings[batch_idx, sent_idx, arg_idx] = avg_embedding

                        # Check for NaN values in arg_embeddings
                        self.check_for_nans(
                            avg_embedding,
                            f"arg_embeddings at batch {batch_idx}, sentence {sent_idx}, arg {arg_idx}",
                        )

        return arg_embeddings

    def forward(
        self,
        sentence_ids: torch.Tensor,
        sentence_attention_masks: torch.Tensor,
        predicate_ids: torch.Tensor,
        arg0_ids: torch.Tensor,
        arg1_ids: torch.Tensor,
    ):
        with torch.no_grad():
            sentence_embeddings, sentence_embeddings_avg = self.get_sentence_embedding(
                sentence_ids, sentence_attention_masks
            )

            # Check for NaNs in sentence_embeddings_avg or sentence_embeddings
            self.check_for_nans(sentence_embeddings_avg, "sentence_embeddings_avg")
            self.check_for_nans(sentence_embeddings, "sentence_embeddings")

            predicate_embeddings = self.get_arg_embedding(
                predicate_ids, sentence_ids, sentence_embeddings
            )
            arg0_embeddings = self.get_arg_embedding(
                arg0_ids, sentence_ids, sentence_embeddings
            )
            arg1_embeddings = self.get_arg_embedding(
                arg1_ids, sentence_ids, sentence_embeddings
            )

            # Final check for NaN values in output embeddings
            self.check_for_nans(predicate_embeddings, "predicate_embeddings")
            self.check_for_nans(arg0_embeddings, "arg0_embeddings")
            self.check_for_nans(arg1_embeddings, "arg1_embeddings")

        return (
            sentence_embeddings_avg,
            predicate_embeddings,
            arg0_embeddings,
            arg1_embeddings,
        )
