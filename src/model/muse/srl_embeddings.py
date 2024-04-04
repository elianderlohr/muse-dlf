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

    def get_embeddings(self, ids, attention_masks, reduce_dim):
        # Utilizing the BERT/Roberta model to get embeddings and then averaging them over the specified dimension.
        embeddings = self.bert_model(input_ids=ids, attention_mask=attention_masks)[0]
        # Reshape for averaging
        shape = ids.size()[:-1] + (self.embedding_dim,)
        embeddings = embeddings.view(*shape)
        # Reduce/average over the specified dimension
        averaged_embeddings = embeddings.mean(dim=reduce_dim)
        return averaged_embeddings

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
            # Process sentences - average over the last dimension
            sentence_embeddings = self.get_embeddings(
                sentence_ids, sentence_attention_masks, reduce_dim=-2
            )

            # Process predicates, ARG0, ARG1 - reshape and then average over the last dimension
            predicate_embeddings = self.get_embeddings(
                predicate_ids, predicate_attention_masks, reduce_dim=-1
            )
            arg0_embeddings = self.get_embeddings(
                arg0_ids, arg0_attention_masks, reduce_dim=-1
            )
            arg1_embeddings = self.get_embeddings(
                arg1_ids, arg1_attention_masks, reduce_dim=-1
            )

        return (
            sentence_embeddings,
            predicate_embeddings,
            arg0_embeddings,
            arg1_embeddings,
        )
