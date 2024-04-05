from transformers import BertModel, RobertaModel
import torch.nn as nn
import torch

from utils.logging_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class SRLEmbeddings(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", bert_model_name_or_path=""):
        super(SRLEmbeddings, self).__init__()

        if bert_model_name == "bert-base-uncased":
            self.bert_model = BertModel.from_pretrained(bert_model_name_or_path)
        elif bert_model_name == "roberta-base":
            self.bert_model = RobertaModel.from_pretrained(bert_model_name_or_path)

        # move to cuda if available
        if torch.cuda.is_available():
            self.bert_model.cuda()

        self.embedding_dim = 768

    def get_sentence_embedding(self, ids, attention_masks):
        # Assume ids and attention_masks shapes are [batch_size, num_sentences, max_sentence_length]
        batch_size, num_sentences, max_sentence_length = ids.size()

        # Flatten ids and attention_masks to 2D tensors
        ids_flat = ids.view(-1, max_sentence_length)
        attention_masks_flat = attention_masks.view(-1, max_sentence_length)

        with torch.no_grad():
            # Obtain the embeddings from the BERT model
            embeddings = self.bert_model(
                input_ids=ids_flat, attention_mask=attention_masks_flat
            )[0]

            # Use attention masks to ignore padded tokens
            # Expand attention masks for element-wise multiplication with embeddings
            attention_masks_expanded = attention_masks_flat.unsqueeze(-1).expand(
                embeddings.size()
            )

            # Multiply embeddings with the expanded attention mask
            # This zeroes out padded token embeddings so they don't contribute to the sum
            embeddings_masked = embeddings * attention_masks_expanded

            # Sum embeddings across the token dimension
            sum_embeddings = torch.sum(embeddings_masked, dim=1)

            # Count real tokens (sum across attention_masks_flat)
            token_counts = attention_masks_flat.sum(dim=1, keepdim=True)

            # Avoid division by zero for sentences that are entirely padding
            token_counts = token_counts.clamp(min=1)

            # Calculate mean by dividing sum of embeddings by number of real tokens
            embeddings_mean = sum_embeddings / token_counts

        # Reshape back to original batch and sentence dimensions
        embeddings_mean_reshaped = embeddings_mean.view(batch_size, num_sentences, -1)
        embeddings_reshaped = embeddings.view(
            batch_size, num_sentences, max_sentence_length, -1
        )

        return embeddings_reshaped, embeddings_mean_reshaped

    def get_arg_embedding(
        self, predicate_ids, sentence_ids, sentence_attention_masks, sentence_embeddings
    ):
        batch_size, num_sentences, seq_len = sentence_ids.shape
        _, _, num_predicates, pred_token_len = predicate_ids.shape
        embedding_dim = sentence_embeddings.size(-1)

        # Extend sentence_ids and attention masks to match predicate_ids for broadcasting
        extended_sentence_ids = (
            sentence_ids.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, num_predicates, pred_token_len, -1)
        )
        extended_attention_masks = (
            sentence_attention_masks.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, num_predicates, pred_token_len, -1)
        )

        # Extend predicate_ids for comparison
        extended_predicate_ids = predicate_ids.unsqueeze(4).expand(
            -1, -1, -1, -1, seq_len
        )

        # Create mask for valid predicate tokens (non-padding) and for matching sentence tokens
        valid_pred_mask = extended_predicate_ids != 1  # Assuming 1 is padding
        matches = (extended_sentence_ids == extended_predicate_ids) & valid_pred_mask

        # Apply sentence attention mask
        matches &= extended_attention_masks.bool()

        # Prepare embeddings tensor, initially filled with zeros
        arg_embeddings = torch.zeros(
            batch_size,
            num_sentences,
            num_predicates,
            embedding_dim,
            device=sentence_embeddings.device,
        )

        for batch_idx in range(batch_size):
            for sent_idx in range(num_sentences):
                for pred_idx in range(num_predicates):
                    # Gather all embeddings corresponding to the matches, then average them
                    match_indices = matches[batch_idx, sent_idx, pred_idx].nonzero()
                    if match_indices.nelement() == 0:
                        continue  # Skip if no matches

                    # Flatten indices to use for gathering embeddings across seq_len dimension
                    flat_indices = match_indices[:, -1]
                    selected_embeddings = sentence_embeddings[
                        batch_idx, sent_idx, flat_indices
                    ]

                    # Average the selected embeddings, ensuring not to divide by zero
                    if selected_embeddings.nelement() > 0:
                        avg_embedding = selected_embeddings.mean(dim=0)
                        arg_embeddings[batch_idx, sent_idx, pred_idx] = avg_embedding

        return arg_embeddings

    def forward(
        self,
        sentence_ids,
        sentence_attention_masks,
        predicate_ids,
        predicate_attention_masks,
        arg0_ids,
        arg0_attention_masks,
        arg1_ids,
        arg1_attention_masks,
    ):
        with torch.no_grad():
            sentence_embeddings, sentence_embeddings_avg = self.get_sentence_embedding(
                sentence_ids, sentence_attention_masks
            )

            predicate_embeddings = self.get_arg_embedding(
                predicate_ids,
                sentence_ids,
                sentence_attention_masks,
                sentence_embeddings,
            )
            arg0_embeddings = self.get_arg_embedding(
                arg0_ids, sentence_ids, sentence_attention_masks, sentence_embeddings
            )
            arg1_embeddings = self.get_arg_embedding(
                arg1_ids, sentence_ids, sentence_attention_masks, sentence_embeddings
            )

        return (
            sentence_embeddings_avg,
            predicate_embeddings,
            arg0_embeddings,
            arg1_embeddings,
        )
