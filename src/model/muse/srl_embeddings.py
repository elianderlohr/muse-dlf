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

        ids_flat = ids.view(-1, max_sentence_length)
        attention_masks_flat = attention_masks.view(-1, max_sentence_length)

        with torch.no_grad():
            embeddings = self.bert_model(
                input_ids=ids_flat, attention_mask=attention_masks_flat
            )[0]

            embeddings_mean = embeddings.mean(dim=1)

        embeddings_reshaped = embeddings_mean.view(batch_size, num_sentences, -1)

        return embeddings_reshaped

    def get_arg_embedding(self, ids, attention_mask):
        # Flatten the batch, num_sentences, and num_args dimensions
        batch_size, num_sentences, num_args, max_arg_length = ids.size()

        ids_flat = ids.view(-1, max_arg_length)
        attention_mask_flat = attention_mask.view(-1, max_arg_length)

        with torch.no_grad():
            embeddings = self.bert_model(
                input_ids=ids_flat, attention_mask=attention_mask_flat
            )[0]

            embeddings_mean = embeddings.mean(dim=1)

        embeddings_reshaped = embeddings_mean.view(
            batch_size, num_sentences, num_args, -1
        )

        return embeddings_reshaped

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
            sentence_embeddings = self.get_sentence_embedding(
                sentence_ids, sentence_attention_masks
            )
            predicate_embeddings = self.get_arg_embedding(
                predicate_ids, predicate_attention_masks
            )
            arg0_embeddings = self.get_arg_embedding(arg0_ids, arg0_attention_masks)
            arg1_embeddings = self.get_arg_embedding(arg1_ids, arg1_attention_masks)

        return (
            sentence_embeddings,
            predicate_embeddings,
            arg0_embeddings,
            arg1_embeddings,
        )
