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
            # Sentence embeddings
            sentence_embeddings = self.bert_model(
                input_ids=sentence_ids.view(-1, sentence_ids.size(-1)),
                attention_mask=sentence_attention_masks.view(
                    -1, sentence_attention_masks.size(-1)
                ),
            )[0]
            sentence_embeddings = sentence_embeddings.view(
                sentence_ids.size(0), sentence_ids.size(1), -1, self.embedding_dim
            )
            sentence_embeddings = sentence_embeddings.mean(dim=2)

            # Predicate embeddings
            predicate_embeddings = self.bert_model(
                input_ids=predicate_ids.view(-1, predicate_ids.size(-1)),
                attention_mask=predicate_attention_masks.view(
                    -1, predicate_attention_masks.size(-1)
                ),
            )[0]
            predicate_embeddings = predicate_embeddings.view(
                predicate_ids.size(0),
                predicate_ids.size(1),
                predicate_ids.size(2),
                -1,
                self.embedding_dim,
            )
            predicate_embeddings = predicate_embeddings.mean(dim=3)

            # ARG0 embeddings
            arg0_embeddings = self.bert_model(
                input_ids=arg0_ids.view(-1, arg0_ids.size(-1)),
                attention_mask=arg0_attention_masks.view(
                    -1, arg0_attention_masks.size(-1)
                ),
            )[0]
            arg0_embeddings = arg0_embeddings.view(
                arg0_ids.size(0),
                arg0_ids.size(1),
                arg0_ids.size(2),
                -1,
                self.embedding_dim,
            )
            arg0_embeddings = arg0_embeddings.mean(dim=3)

            # ARG1 embeddings
            arg1_embeddings = self.bert_model(
                input_ids=arg1_ids.view(-1, arg1_ids.size(-1)),
                attention_mask=arg1_attention_masks.view(
                    -1, arg1_attention_masks.size(-1)
                ),
            )[0]
            arg1_embeddings = arg1_embeddings.view(
                arg1_ids.size(0),
                arg1_ids.size(1),
                arg1_ids.size(2),
                -1,
                self.embedding_dim,
            )
            arg1_embeddings = arg1_embeddings.mean(dim=3)

        return (
            sentence_embeddings,
            predicate_embeddings,
            arg0_embeddings,
            arg1_embeddings,
        )
