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
            # Encode the entire sentence
            sentence_outputs = self.bert_model(
                input_ids=sentence_ids.view(-1, sentence_ids.size(-1)),
                attention_mask=sentence_attention_masks.view(
                    -1, sentence_attention_masks.size(-1)
                ),
            )
            sentence_embeddings = sentence_outputs[
                0
            ]  # Assuming [0] gets the last hidden state

        # Now extract embeddings for predicate, ARG0, and ARG1 based on their token IDs
        predicate_embeddings = self.extract_embeddings_from_ids(
            sentence_embeddings, predicate_ids, sentence_ids
        )
        arg0_embeddings = self.extract_embeddings_from_ids(
            sentence_embeddings, arg0_ids, sentence_ids
        )
        arg1_embeddings = self.extract_embeddings_from_ids(
            sentence_embeddings, arg1_ids, sentence_ids
        )

        return (
            sentence_embeddings,
            predicate_embeddings,
            arg0_embeddings,
            arg1_embeddings,
        )

    def extract_embeddings_from_ids(self, sentence_embeddings, token_ids, sentence_ids):
        embeddings = []
        # loop over sentence_ids and add to embeddings if token_id is in token_ids
        for sentence_id, sentence_embedding in zip(sentence_ids, sentence_embeddings):
            if sentence_id in token_ids:
                embeddings.append(sentence_embedding)

        # mean over the embeddings
        embeddings = torch.stack(embeddings).mean(dim=0)

        return embeddings
