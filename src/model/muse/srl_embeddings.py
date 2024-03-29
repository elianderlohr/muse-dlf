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
        # Assuming the first dimension for attention masks aligns with sentence_ids for simplicity
        with torch.no_grad():
            sentence_embeddings = self.bert_model(
                input_ids=sentence_ids.view(-1, sentence_ids.size(-1)),
                attention_mask=sentence_attention_masks.view(
                    -1, sentence_attention_masks.size(-1)
                ),
            )[0]
            sentence_embeddings = sentence_embeddings.view(
                sentence_ids.size(0), sentence_ids.size(1), -1, self.embedding_dim
            )

        sentence_average_embeddings = sentence_embeddings.mean(dim=2)

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
            sentence_average_embeddings,
            predicate_embeddings,
            arg0_embeddings,
            arg1_embeddings,
        )

    def extract_embeddings_from_ids(self, sentence_embeddings, token_ids, sentence_ids):
        print("sentence_embeddings", sentence_embeddings.shape)
        print("sentence_ids", sentence_ids.shape)
        print("token_ids", token_ids.shape)

        batch_size, num_sentences, seq_length, emb_dim = sentence_embeddings.shape
        _, _, num_args, arg_length = token_ids.shape

        averaged_embeddings = []
        # batch_size
        for i in range(batch_size):
            batches = []
            # num_sentences
            for j in range(num_sentences):
                sentences = []
                # num_args
                for k in range(num_args):
                    embedding = []

                    # arg_length
                    for l in range(arg_length):
                        token_id = token_ids[i, j, k, l]

                        # skip 0
                        if token_id == 0:
                            continue

                        sentence_ids_to_find = sentence_ids[i, j]

                        # find index of token_id in sentence_ids_to_find
                        idx = (sentence_ids_to_find == token_id).nonzero(as_tuple=True)[
                            0
                        ]

                        print(
                            "try to find token_id",
                            token_id,
                            "in",
                            sentence_ids_to_find,
                            "found at",
                            idx,
                        )

                        # append embedding
                        embedding.append(sentence_embeddings[i, j, idx])

                    # average embeddings
                    averaged_embedding = torch.mean(torch.stack(embedding), dim=0)
                    sentences.append(averaged_embedding)
                batches.append(sentences)
            averaged_embeddings.append(batches)

        averaged_embeddings = torch.stack(averaged_embeddings)

        # if not shape is (batch_size, num_sentences, num_args, emb_dim) create empty tensor
        if averaged_embeddings.shape != (batch_size, num_sentences, num_args, emb_dim):
            logger.warning(
                f"Shape of averaged_embeddings is {averaged_embeddings.shape}, expected {(batch_size, num_sentences, num_args, emb_dim)}"
            )
            averaged_embeddings = torch.zeros(
                batch_size, num_sentences, num_args, emb_dim
            )

        return averaged_embeddings
