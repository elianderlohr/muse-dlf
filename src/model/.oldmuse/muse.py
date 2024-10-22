import torch
import torch.nn as nn

from model.oldmuse.srl_embeddings import SRLEmbeddings
from model.oldmuse.supervised_module import MuSESupervised
from model.oldmuse.unsupervised_module import MuSEUnsupervised
from model.oldmuse.unsupervised_frameaxis_module import MuSEFrameAxisUnsupervised
from utils.logging_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class MuSE(nn.Module):
    def __init__(
        self,
        embedding_dim,
        D_h,
        lambda_orthogonality,
        M,
        t,
        num_sentences,
        K,
        num_frames,
        frameaxis_dim,
        dropout_prob=0.3,
        bert_model_name="bert-base-uncased",
        bert_model_name_or_path="",
        supervised_sentence_prediction_method="friss",  # friss or custom
    ):
        super(MuSE, self).__init__()

        # Aggregation layer replaced with SRL_Embeddings
        self.aggregation = SRLEmbeddings(bert_model_name, bert_model_name_or_path)

        # Unsupervised training module
        self.unsupervised = MuSEUnsupervised(
            embedding_dim,
            D_h,
            K,
            lambda_orthogonality=lambda_orthogonality,
            M=M,
            t=t,
            dropout_prob=dropout_prob,
        )

        self.unsupervised_fx = MuSEFrameAxisUnsupervised(
            embedding_dim,
            D_h,
            K,
            frameaxis_dim=frameaxis_dim,
            lambda_orthogonality=lambda_orthogonality,
            M=M,
            t=t,
            dropout_prob=dropout_prob,
        )

        # Supervised training module
        self.supervised = MuSESupervised(
            embedding_dim,
            K,
            num_frames,
            frameaxis_dim=frameaxis_dim,
            num_sentences=num_sentences,
            dropout_prob=dropout_prob,
            sentence_prediction_method=supervised_sentence_prediction_method,
        )

    def negative_sampling(self, embeddings, num_negatives=-1):
        if num_negatives == -1:
            num_negatives = embeddings.size(0)

        batch_size, num_sentences, num_args, embedding_dim = embeddings.size()
        all_negatives = []

        for i in range(batch_size):
            for j in range(num_sentences):
                # Flatten the arguments dimension to sample across all arguments in the sentence
                flattened_embeddings = embeddings[i, j].view(-1, embedding_dim)

                # Get indices of non-padded embeddings (assuming padding is represented by all-zero vectors)
                non_padded_indices = torch.where(
                    torch.any(flattened_embeddings != 0, dim=1)
                )[0]

                # Randomly sample negative indices from non-padded embeddings
                if len(non_padded_indices) > 0:
                    negative_indices = non_padded_indices[
                        torch.randint(0, len(non_padded_indices), (num_negatives,))
                    ]
                else:
                    # If no non-padded embeddings, use zeros
                    negative_indices = torch.zeros(num_negatives, dtype=torch.long)

                negative_samples = flattened_embeddings[negative_indices, :]
                all_negatives.append(negative_samples)

        # Concatenate all negative samples into a single tensor
        all_negatives = torch.cat(all_negatives, dim=0)

        # If more samples than required, randomly select 'num_negatives' samples
        if all_negatives.size(0) > num_negatives:
            indices = torch.randperm(all_negatives.size(0))[:num_negatives]
            all_negatives = all_negatives[indices]

        return all_negatives

    def negative_fx_sampling(self, fxs, num_negatives=8):
        batch_size, num_sentences, frameaxis_dim = fxs.size()
        all_negatives = []

        for i in range(batch_size):
            for j in range(num_sentences):
                # Flatten the arguments dimension to sample across all arguments in the sentence
                flattened_fxs = fxs[i, j].view(-1, frameaxis_dim)

                # Get indices of non-padded embeddings (assuming padding is represented by all-zero vectors)
                non_padded_indices = torch.where(torch.any(flattened_fxs != 0, dim=1))[
                    0
                ]

                # Randomly sample negative indices from non-padded embeddings
                if len(non_padded_indices) > 0:
                    negative_indices = non_padded_indices[
                        torch.randint(0, len(non_padded_indices), (num_negatives,))
                    ]
                else:
                    # If no non-padded embeddings, use zeros
                    negative_indices = torch.zeros(num_negatives, dtype=torch.long)

                negative_samples = flattened_fxs[negative_indices, :]
                all_negatives.append(negative_samples)

        # Concatenate all negative samples into a single tensor
        all_negatives = torch.cat(all_negatives, dim=0)

        # If more samples than required, randomly select 'num_negatives' samples
        if all_negatives.size(0) > num_negatives:
            indices = torch.randperm(all_negatives.size(0))[:num_negatives]
            all_negatives = all_negatives[indices]

        return all_negatives

    def forward(
        self,
        sentence_ids,
        sentence_attention_masks,
        predicate_ids,
        arg0_ids,
        arg1_ids,
        frameaxis_data,
        tau,
    ):
        # Convert input IDs to embeddings
        sentence_embeddings, predicate_embeddings, arg0_embeddings, arg1_embeddings = (
            self.aggregation(
                sentence_ids,
                sentence_attention_masks,
                predicate_ids,
                arg0_ids,
                arg1_ids,
            )
        )

        # Handle multiple spans by averaging predictions
        unsupervised_losses = torch.zeros(
            (sentence_embeddings.size(0),), device=sentence_embeddings.device
        )

        # Creating storage for aggregated d tensors
        d_p_list, d_a0_list, d_a1_list, d_fx_list = [], [], [], []

        negatives_p = self.negative_sampling(predicate_embeddings)
        negatives_a0 = self.negative_sampling(arg0_embeddings)
        negatives_a1 = self.negative_sampling(arg1_embeddings)

        negatives_fx = self.negative_fx_sampling(frameaxis_data)

        # Process each sentence
        for sentence_idx in range(sentence_embeddings.size(1)):
            s_sentence_span = sentence_embeddings[:, sentence_idx, :]
            v_fx = frameaxis_data[:, sentence_idx, :]

            d_p_sentence_list = []
            d_a0_sentence_list = []
            d_a1_sentence_list = []

            # Process each span
            for span_idx in range(predicate_embeddings.size(2)):
                v_p_span = predicate_embeddings[:, sentence_idx, span_idx, :]
                v_a0_span = arg0_embeddings[:, sentence_idx, span_idx, :]
                v_a1_span = arg1_embeddings[:, sentence_idx, span_idx, :]

                # Feed the embeddings to the unsupervised module
                unsupervised_results = self.unsupervised(
                    v_p_span,
                    v_a0_span,
                    v_a1_span,
                    s_sentence_span,
                    negatives_p,
                    negatives_a0,
                    negatives_a1,
                    tau,
                )
                unsupervised_losses += unsupervised_results["loss"]

                # Use the vhat (reconstructed embeddings) for supervised predictions
                d_p_sentence_list.append(unsupervised_results["p"]["d"])
                d_a0_sentence_list.append(unsupervised_results["a0"]["d"])
                d_a1_sentence_list.append(unsupervised_results["a1"]["d"])

            # Aggregating across all spans
            d_p_sentence = torch.stack(d_p_sentence_list, dim=1)
            d_a0_sentence = torch.stack(d_a0_sentence_list, dim=1)
            d_a1_sentence = torch.stack(d_a1_sentence_list, dim=1)

            d_p_list.append(d_p_sentence)
            d_a0_list.append(d_a0_sentence)
            d_a1_list.append(d_a1_sentence)

            # As per sentence only one frameaxis data set is present calculate only once
            unsupervised_fx_results = self.unsupervised_fx(
                s_sentence_span,
                v_fx,
                negatives_fx,
                tau,
            )

            d_fx_list.append(unsupervised_fx_results["fx"]["d"])

            # add the loss to the unsupervised losses
            unsupervised_losses += unsupervised_fx_results["loss"]

        # Aggregating across all spans
        d_p_aggregated = torch.stack(d_p_list, dim=1)
        d_a0_aggregated = torch.stack(d_a0_list, dim=1)
        d_a1_aggregated = torch.stack(d_a1_list, dim=1)
        d_fx_aggregated = torch.stack(d_fx_list, dim=1)

        # Supervised predictions
        span_pred, sentence_pred, combined_pred, other = self.supervised(
            d_p_aggregated,
            d_a0_aggregated,
            d_a1_aggregated,
            d_fx_aggregated,
            sentence_embeddings,
            frameaxis_data,
        )

        # Identify valid (non-nan) losses
        valid_losses = ~torch.isnan(unsupervised_losses)

        # Take average by summing the valid losses and dividing by num sentences so that padded sentences are also taken in equation
        unsupervised_loss = unsupervised_losses[valid_losses].sum() / (
            sentence_embeddings.shape[0]
            * sentence_embeddings.shape[1]
            * sentence_embeddings.shape[2]
        )

        return unsupervised_loss, span_pred, sentence_pred, combined_pred, other
