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

    def check_for_nans(self, tensor, tensor_name):
        if torch.isnan(tensor).any():
            self.logger.error(f"NaN values detected in {tensor_name}")
            raise ValueError(f"NaN values detected in {tensor_name}")

    def get_sentence_embedding(self, ids: torch.Tensor, attention_masks: torch.Tensor):
        # Assume ids and attention_masks shapes are [batch_size, num_sentences, max_sentence_length]
        batch_size, num_sentences, max_sentence_length = ids.size()

        # Check for NaNs in inputs
        self.check_for_nans(ids, "input ids")
        self.check_for_nans(attention_masks, "attention masks")

        # Flatten ids and attention_masks to 2D tensors
        ids_flat = ids.view(-1, max_sentence_length)
        attention_masks_flat = attention_masks.view(-1, max_sentence_length)

        # Ensure ids_flat and attention_masks_flat contain valid values
        assert not torch.isnan(
            ids_flat
        ).any(), "NaNs found in ids_flat before model input"
        assert not torch.isinf(
            ids_flat
        ).any(), "Infs found in ids_flat before model input"
        assert not torch.isnan(
            attention_masks_flat
        ).any(), "NaNs found in attention_masks_flat before model input"
        assert not torch.isinf(
            attention_masks_flat
        ).any(), "Infs found in attention_masks_flat before model input"

        with torch.no_grad():
            # Obtain the embeddings from the BERT model
            embeddings = self.model(
                input_ids=ids_flat, attention_mask=attention_masks_flat
            )[0]

        # if embeddings is NaN, log the input ids and attention_masks
        if torch.isnan(embeddings).any():
            self.logger.error(f"ERROR IN EMBEDDINGS FROM BERT")

            # check if the input ids and attention_masks have NaN values
            if torch.isnan(ids_flat).any() or torch.isnan(attention_masks_flat).any():
                self.logger.error(f"ERROR IN INPUT IDS AND ATTENTION MASKS")
                # print the nan rows in ids_flat and attention_masks_flat which are causing the issue
                nan_indices = torch.isnan(ids_flat).any(dim=1) | torch.isnan(
                    attention_masks_flat
                ).any(dim=1)
                self.logger.error(f"nan_indices: {nan_indices}")
                self.logger.error(f"ids_flat[nan_indices]: {ids_flat[nan_indices]}")
                self.logger.error(
                    f"attention_masks_flat[nan_indices]: {attention_masks_flat[nan_indices]}"
                )

        # Reshape back to original batch and sentence dimensions
        embeddings_reshaped = embeddings.view(
            batch_size, num_sentences, max_sentence_length, -1
        )

        if self.pooling == "mean":
            # Calculate mean embeddings across the token dimension while ignoring padded tokens
            attention_masks_expanded = attention_masks_flat.unsqueeze(-1).expand(
                embeddings.size()
            )
            embeddings_masked = embeddings * attention_masks_expanded
            sum_embeddings = torch.sum(embeddings_masked, dim=1)
            token_counts = attention_masks_flat.sum(dim=1, keepdim=True).clamp(min=1)
            embeddings_mean = sum_embeddings / token_counts
            embeddings_mean_reshaped = embeddings_mean.view(
                batch_size, num_sentences, -1
            )
        elif self.pooling == "cls":
            # Use the [CLS] token representation for the sentence embedding
            embeddings_mean_reshaped = embeddings[:, 0, :].view(
                batch_size, num_sentences, -1
            )

        # Check for NaNs in mean or CLS embeddings
        self.check_for_nans(
            embeddings_mean_reshaped, "sentence embeddings (mean or CLS)"
        )

        return embeddings_reshaped, embeddings_mean_reshaped

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
