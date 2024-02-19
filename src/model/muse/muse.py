import torch
import torch.nn as nn

from model.muse.srl_embeddings import SRLEmbeddings
from model.muse.supervised_module import MUSESupervised
from model.muse.unsupervised_module import MUSEUnsupervised


class MUSE(nn.Module):
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
    ):
        super(MUSE, self).__init__()

        # Aggregation layer replaced with SRL_Embeddings
        self.aggregation = SRLEmbeddings(bert_model_name)

        # Unsupervised training module
        self.unsupervised = MUSEUnsupervised(
            embedding_dim,
            D_h,
            K,
            frameaxis_dim=frameaxis_dim,
            num_frames=num_frames,
            lambda_orthogonality=lambda_orthogonality,
            M=M,
            t=t,
            dropout_prob=dropout_prob,
        )

        # Supervised training module
        self.supervised = MUSESupervised(
            embedding_dim,
            K,
            num_frames,
            frameaxis_dim=frameaxis_dim,
            num_sentences=num_sentences,
            dropout_prob=dropout_prob,
        )

    def negative_sampling(self, embeddings, num_negatives=8):
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
        predicate_attention_masks,
        arg0_ids,
        arg0_attention_masks,
        arg1_ids,
        arg1_attention_masks,
        frameaxis_data,
        tau,
    ):
        # Convert input IDs to embeddings
        sentence_embeddings, predicate_embeddings, arg0_embeddings, arg1_embeddings = (
            self.aggregation(
                sentence_ids,
                sentence_attention_masks,
                predicate_ids,
                predicate_attention_masks,
                arg0_ids,
                arg0_attention_masks,
                arg1_ids,
                arg1_attention_masks,
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
            d_fx_sentence_list = []

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
                    v_fx,
                    negatives_p,
                    negatives_a0,
                    negatives_a1,
                    negatives_fx,
                    tau,
                )
                unsupervised_losses += unsupervised_results["loss"]

                if torch.isnan(unsupervised_results["loss"]).any():
                    print("loss is nan")

                # Use the vhat (reconstructed embeddings) for supervised predictions
                d_p_sentence_list.append(unsupervised_results["p"]["d"])
                d_a0_sentence_list.append(unsupervised_results["a0"]["d"])
                d_a1_sentence_list.append(unsupervised_results["a1"]["d"])
                d_fx_sentence_list.append(unsupervised_results["fx"]["d"])

            # Aggregating across all spans
            d_p_sentence = torch.stack(d_p_sentence_list, dim=1)
            d_a0_sentence = torch.stack(d_a0_sentence_list, dim=1)
            d_a1_sentence = torch.stack(d_a1_sentence_list, dim=1)
            d_fx_sentence = torch.stack(d_fx_sentence_list, dim=1)

            d_p_list.append(d_p_sentence)
            d_a0_list.append(d_a0_sentence)
            d_a1_list.append(d_a1_sentence)
            d_fx_list.append(d_fx_sentence)

        # Aggregating across all spans
        d_p_aggregated = torch.stack(d_p_list, dim=1)
        d_a0_aggregated = torch.stack(d_a0_list, dim=1)
        d_a1_aggregated = torch.stack(d_a1_list, dim=1)
        d_fx_aggregated = torch.stack(d_fx_list, dim=1)

        # Supervised predictions
        span_pred, sentence_pred, combined_pred = self.supervised(
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
            sentence_embeddings.shape[1] * sentence_embeddings.shape[2]
        )

        return unsupervised_loss, span_pred, sentence_pred, combined_pred
