import torch
from torch.cuda.amp import autocast
from transformers import BertModel, RobertaModel
import torch.nn as nn
from utils.logging_manager import LoggerManager


class SLMUSEEmbeddings(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        model_type: str = "bert-base-uncased",
        pooling: str = "mean",
        _debug=False,
    ):
        super(SLMUSEEmbeddings, self).__init__()

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if pooling not in ["mean", "cls"]:
            raise ValueError(
                f"Unsupported pooling type. Choose either 'mean' or 'cls'. Found: {pooling}"
            )

        self.pooling = pooling

        self.embedding_dim = self.model.config.hidden_size

        self._debug = _debug

        if self._debug:
            self.verify_model_loading()

        # Debugging:
        self.logger.debug(f"✅ SRLEmbeddings successfully initialized")

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
            with autocast():
                outputs = model(input_ids=test_ids, attention_mask=test_attention_masks)
                embeddings = outputs.last_hidden_state

        if torch.isnan(embeddings).any():
            raise ValueError("NaNs found in test embeddings after loading the model")

        self.logger.info("No NaNs found in test embeddings. Model loading seems fine.")

    def check_for_nans(self, tensor, tensor_name):
        if torch.isnan(tensor).any():
            self.logger.error(f"NaN values detected in {tensor_name}")

    def get_sentence_embedding(
        self, ids: torch.Tensor, attention_masks: torch.Tensor, mixed_precision="fp16"
    ):
        ids, attention_masks = ids.to(self.device), attention_masks.to(self.device)
        batch_size, num_sentences, max_sentence_length = ids.shape

        ids_flat = ids.view(batch_size * num_sentences, max_sentence_length)
        attention_masks_flat = attention_masks.view(
            batch_size * num_sentences, max_sentence_length
        )

        precision_dtype = (
            torch.float16
            if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
        )

        with torch.no_grad():
            with autocast(
                enabled=mixed_precision in ["fp16", "bf16", "fp32"],
                dtype=precision_dtype,
            ):
                outputs = self.model(
                    input_ids=ids_flat,
                    attention_mask=attention_masks_flat,
                    output_hidden_states=True,
                )

                second_to_last_hidden_state = outputs.hidden_states[-2]

            self.check_for_nans(
                second_to_last_hidden_state, "second to last hidden state"
            )

            second_to_last_hidden_state = second_to_last_hidden_state.view(
                batch_size, num_sentences, max_sentence_length, self.embedding_dim
            )

            if self.pooling == "mean":
                attention_masks_expanded = attention_masks.view(
                    batch_size, num_sentences, max_sentence_length, 1
                )
                attention_masks_expanded = attention_masks_expanded.expand(
                    second_to_last_hidden_state.size()
                )
                embeddings_masked = (
                    second_to_last_hidden_state * attention_masks_expanded
                )
                sum_embeddings = torch.sum(embeddings_masked, dim=2)
                token_counts = (
                    attention_masks.view(batch_size, num_sentences, max_sentence_length)
                    .sum(dim=2, keepdim=True)
                    .clamp(min=1)
                )
                embeddings_mean = sum_embeddings / token_counts
            elif self.pooling == "cls":
                embeddings_mean = second_to_last_hidden_state[:, :, 0, :]

            self.check_for_nans(embeddings_mean, "embeddings_mean")

            # Delete intermediate tensors
            del (
                ids_flat,
                attention_masks_flat,
                outputs,
                attention_masks_expanded,
                embeddings_masked,
                sum_embeddings,
                token_counts,
            )
            torch.cuda.empty_cache()

        return second_to_last_hidden_state, embeddings_mean

    def get_arg_embedding(
        self,
        arg_ids: torch.Tensor,
        sentence_ids: torch.Tensor,
        sentence_embeddings: torch.Tensor,
    ):
        arg_ids, sentence_ids, sentence_embeddings = (
            arg_ids.to(self.device),
            sentence_ids.to(self.device),
            sentence_embeddings.to(self.device),
        )

        batch_size, num_sentences, max_sentence_length = sentence_ids.shape
        _, _, num_args, max_arg_length = arg_ids.shape

        arg_embeddings = torch.zeros(
            batch_size,
            num_sentences,
            num_args,
            self.embedding_dim,
            device=sentence_embeddings.device,
            dtype=sentence_embeddings.dtype,
        )

        for batch_idx in range(batch_size):
            for sent_idx in range(num_sentences):
                for arg_idx in range(num_args):
                    selected_embeddings = []
                    for token_idx in range(max_arg_length):
                        arg_token_id = arg_ids[
                            batch_idx, sent_idx, arg_idx, token_idx
                        ].item()
                        if arg_token_id == 0:
                            continue
                        match_indices = (
                            sentence_ids[batch_idx, sent_idx] == arg_token_id
                        ).nonzero(as_tuple=False)
                        if match_indices.nelement() == 0:
                            continue
                        flat_indices = match_indices[:, 0]
                        selected_embeddings.append(
                            sentence_embeddings[batch_idx, sent_idx, flat_indices]
                        )
                    if selected_embeddings:
                        selected_embeddings = torch.cat(selected_embeddings, dim=0)
                        avg_embedding = selected_embeddings.mean(dim=0)
                        arg_embeddings[batch_idx, sent_idx, arg_idx] = avg_embedding

                    # Delete intermediate tensors
                    del selected_embeddings, match_indices, flat_indices
                    torch.cuda.empty_cache()

        return arg_embeddings

    def forward(
        self,
        sentence_ids: torch.Tensor,
        sentence_attention_masks: torch.Tensor,
        predicate_ids: torch.Tensor,
        arg0_ids: torch.Tensor,
        arg1_ids: torch.Tensor,
        mixed_precision="fp16",
    ):
        sentence_ids, sentence_attention_masks, predicate_ids, arg0_ids, arg1_ids = (
            sentence_ids.to(self.device),
            sentence_attention_masks.to(self.device),
            predicate_ids.to(self.device),
            arg0_ids.to(self.device),
            arg1_ids.to(self.device),
        )

        with torch.no_grad():

            precision_dtype = (
                torch.float16
                if mixed_precision == "fp16"
                else torch.bfloat16 if mixed_precision == "bf16" else None
            )

            with autocast(
                enabled=mixed_precision in ["fp16", "bf16", "fp32"],
                dtype=precision_dtype,
            ):  # Use autocast for mixed precision
                sentence_embeddings, sentence_embeddings_avg = (
                    self.get_sentence_embedding(
                        sentence_ids,
                        sentence_attention_masks,
                        mixed_precision=mixed_precision,
                    )
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

                # Delete intermediate tensors
                del (
                    sentence_ids,
                    sentence_attention_masks,
                    predicate_ids,
                    arg0_ids,
                    arg1_ids,
                    sentence_embeddings,
                )
                torch.cuda.empty_cache()

        return (
            sentence_embeddings_avg,
            predicate_embeddings,
            arg0_embeddings,
            arg1_embeddings,
        )
