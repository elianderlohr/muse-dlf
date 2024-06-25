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

        # Set model to evaluation mode
        self.model.eval()

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
        test_ids = torch.randint(0, 100, (1, 10))  # Corrected shape
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

        # Define a maximum chunk size to avoid exceeding model limits
        max_chunk_size = 512  # Adjust based on your model's capabilities
        all_embeddings = []
        all_embeddings_mean = []

        for start_idx in range(0, batch_size * num_sentences, max_chunk_size):
            end_idx = min(start_idx + max_chunk_size, batch_size * num_sentences)
            ids_chunk = ids.view(batch_size * num_sentences, max_sentence_length)[
                start_idx:end_idx
            ]
            attention_masks_chunk = attention_masks.view(
                batch_size * num_sentences, max_sentence_length
            )[start_idx:end_idx]

            with torch.no_grad():
                # Obtain the embeddings from the BERT model
                outputs = self.model(
                    input_ids=ids_chunk,
                    attention_mask=attention_masks_chunk,
                    output_hidden_states=True,
                )
                embeddings_chunk = outputs.last_hidden_state

                # Check for NaN values in embeddings_chunk
                self.check_for_nans(
                    embeddings_chunk, f"raw model output chunk {start_idx}-{end_idx}"
                )

                # Compute the sum of the last 4 layers to get the new token embeddings
                last_4_layers = outputs.hidden_states[-4:]  # Last 4 layers
                summed_embeddings_chunk = torch.stack(last_4_layers, dim=0).sum(0)

                # Check for NaN values in summed_embeddings_chunk
                self.check_for_nans(
                    summed_embeddings_chunk,
                    f"summed embeddings chunk {start_idx}-{end_idx}",
                )

                # Reshape the embeddings to the desired output shape
                summed_embeddings_chunk = summed_embeddings_chunk.view(
                    -1, max_sentence_length, self.embedding_dim
                )

                # Calculate mean embeddings across the token dimension while ignoring padded tokens
                if self.pooling == "mean":
                    attention_masks_expanded = attention_masks_chunk.unsqueeze(
                        -1
                    ).expand(summed_embeddings_chunk.size())
                    embeddings_masked = (
                        summed_embeddings_chunk * attention_masks_expanded
                    )
                    sum_embeddings = torch.sum(embeddings_masked, dim=1)
                    token_counts = attention_masks_chunk.sum(dim=1, keepdim=True).clamp(
                        min=1
                    )
                    embeddings_mean_chunk = sum_embeddings / token_counts
                elif self.pooling == "cls":
                    embeddings_mean_chunk = summed_embeddings_chunk[:, 0, :]

                # Append the processed chunks to the final embeddings lists
                all_embeddings.append(summed_embeddings_chunk)
                all_embeddings_mean.append(embeddings_mean_chunk)

        # Concatenate all chunks to form the final embeddings tensors
        embeddings = torch.cat(all_embeddings, dim=0)
        embeddings_mean = torch.cat(all_embeddings_mean, dim=0)

        # Reshape to match the original batch size and number of sentences
        embeddings = embeddings.view(
            batch_size, num_sentences, max_sentence_length, self.embedding_dim
        )
        embeddings_mean = embeddings_mean.view(
            batch_size, num_sentences, self.embedding_dim
        )

        # Check for NaN values in the final mean embeddings
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

            predicate_embeddings = self.get_arg_embedding(
                predicate_ids, sentence_ids, sentence_embeddings
            )
            arg0_embeddings = self.get_arg_embedding(
                arg0_ids, sentence_ids, sentence_embeddings
            )
            arg1_embeddings = self.get_arg_embedding(
                arg1_ids, sentence_ids, sentence_embeddings
            )

        return (
            sentence_embeddings_avg,
            predicate_embeddings,
            arg0_embeddings,
            arg1_embeddings,
        )
