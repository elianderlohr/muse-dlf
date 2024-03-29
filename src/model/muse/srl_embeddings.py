from transformers import BertModel, RobertaModel
import torch.nn as nn
import torch


class SRLEmbeddings(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", bert_model_name_or_path=""):
        super(SRLEmbeddings, self).__init__()

        if bert_model_name == "bert-base-uncased":
            self.bert_model = BertModel.from_pretrained(bert_model_name_or_path)
        elif bert_model_name == "roberta-base":
            self.bert_model = RobertaModel.from_pretrained(bert_model_name_or_path)

        self.embedding_dim = 768  # for bert-base-uncased

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
            sentence_outputs = self.bert_model(
                input_ids=sentence_ids, attention_mask=sentence_attention_masks
            )
            sentence_embeddings = sentence_outputs.last_hidden_state

        sentence_average_embeddings = sentence_embeddings.mean(dim=1)

        predicate_embeddings = self.extract_embeddings_from_ids(
            sentence_embeddings, predicate_ids
        )
        arg0_embeddings = self.extract_embeddings_from_ids(
            sentence_embeddings, arg0_ids
        )
        arg1_embeddings = self.extract_embeddings_from_ids(
            sentence_embeddings, arg1_ids
        )

        return (
            sentence_average_embeddings,
            predicate_embeddings,
            arg0_embeddings,
            arg1_embeddings,
        )

    def extract_embeddings_from_ids(self, sentence_embeddings, token_ids):
        batch_size, seq_length, _ = sentence_embeddings.shape
        num_tokens = token_ids.shape[1]
        _, _, emb_dim = sentence_embeddings.shape
        averaged_embeddings = torch.zeros(
            (batch_size, num_tokens, emb_dim), device=sentence_embeddings.device
        )

        for i in range(batch_size):
            for j in range(num_tokens):
                # Ensuring IDs are within sequence length bounds
                current_ids = token_ids[i, j]
                mask = (current_ids != 0) & (
                    current_ids < seq_length
                )  # Additional check for bounds
                valid_ids = current_ids[mask]

                if len(valid_ids) > 0:  # If there are valid IDs
                    # Ensure safe indexing
                    embeddings = sentence_embeddings[i][valid_ids]  # Fetch embeddings
                    averaged_embeddings[i, j] = embeddings.mean(dim=0)
                # If no valid IDs, the embedding remains zero, which could be handled differently if needed

        return averaged_embeddings
