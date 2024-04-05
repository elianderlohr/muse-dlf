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

    def get_arg_embedding(self, predicate_ids, sentence_ids, sentence_embeddings):
        # Initialize a tensor to store the averaged embeddings
        arg_embeddings = torch.zeros(
            predicate_ids.size(0),
            predicate_ids.size(1),
            predicate_ids.size(2),
            sentence_embeddings.size(-1),
            device=sentence_embeddings.device,
        )

        # Loop over batches, sentences, and arguments
        for batch_idx in range(predicate_ids.size(0)):
            for sentence_idx in range(predicate_ids.size(1)):
                for arg_idx in range(predicate_ids.size(2)):
                    # Get the current predicate IDs for the argument
                    current_predicate_ids = predicate_ids[
                        batch_idx, sentence_idx, arg_idx
                    ]

                    # Initialize a list to store the embeddings for this argument
                    embeddings_list = []

                    for token_idx in current_predicate_ids:
                        if (
                            token_idx.item() != 1
                        ):  # Assuming 1 is a padding value in predicate_ids
                            # Find the index/indices in sentence_ids where this token_id matches
                            match_indices = torch.where(
                                sentence_ids[batch_idx, sentence_idx] == token_idx
                            )

                            embeddings = []

                            # Gather the embeddings for these indices
                            for idx in match_indices:
                                embeddings.append(
                                    sentence_embeddings[batch_idx, sentence_idx, idx]
                                )

                            # mean over embeddings and add to embeddings_list
                            embeddings_list.append(torch.stack(embeddings).mean(dim=0))

                    # If we have collected any embeddings, average them
                    if embeddings_list:
                        # Stack the collected embeddings and compute the average
                        avg_embedding = torch.stack(embeddings_list).mean(dim=0)
                        # Store the averaged embedding
                        arg_embeddings[batch_idx, sentence_idx, arg_idx] = avg_embedding

        # Return the averaged argument embeddings
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
