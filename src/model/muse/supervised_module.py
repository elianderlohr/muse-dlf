import torch
import torch.nn as nn
import torch.nn.functional as F


class MUSESupervised(nn.Module):
    def __init__(
        self,
        D_w,
        K,
        num_frames,
        frameaxis_dim,
        num_sentences,
        dropout_prob=0.3,
        sentence_prediction_method="friss",
        combine_method="sum",
    ):
        super(MUSESupervised, self).__init__()

        self.D_w = D_w
        self.frameaxis_dim = frameaxis_dim

        self.dropout_1 = nn.Dropout(dropout_prob)
        self.dropout_2 = nn.Dropout(dropout_prob)

        self.Wr = nn.Linear(
            D_w + (frameaxis_dim if sentence_prediction_method == "custom" else 0), D_w
        )
        self.Wt = nn.Linear(D_w, K)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.sentence_prediction_method = sentence_prediction_method
        self.combine_method = combine_method

    def forward(
        self,
        d_p,
        d_a0,
        d_a1,
        d_fx,
        vs,
        frameaxis_data,
        sentence_attention_mask,
        args_mask,
    ):
        # print shapes of all inputs
        print("d_p shape:", d_p.shape)
        print("d_a0 shape:", d_a0.shape)
        print("d_a1 shape:", d_a1.shape)
        print("d_fx shape:", d_fx.shape)
        print("vs shape:", vs.shape)
        print("frameaxis_data shape:", frameaxis_data.shape)
        print("sentence_attention_mask shape:", sentence_attention_mask.shape)
        print("args_mask shape:", args_mask.shape)

        # Adjust mask for use with d_p, d_a0, d_a1, and d_fx
        valid_sentences_mask = sentence_attention_mask.unsqueeze(-1).unsqueeze(
            -1
        )  # [B, S, 1, 1]
        print("valid_sentences_mask shape:", valid_sentences_mask.shape)

        def masked_mean(tensor, mask):
            # Ensure mask shape matches tensor
            mask = mask.expand_as(tensor)
            masked_tensor = tensor * mask.float()
            sum_tensor = masked_tensor.sum(dim=2)  # Sum along the third dim (num_args)
            count_tensor = mask.sum(dim=2)
            mean_tensor = sum_tensor / count_tensor.clamp(min=1)  # Safe division
            return mean_tensor

        # Apply masked mean
        d_p_mean = masked_mean(d_p, valid_sentences_mask.squeeze(-1))
        d_a0_mean = masked_mean(d_a0, valid_sentences_mask.squeeze(-1))
        d_a1_mean = masked_mean(d_a1, valid_sentences_mask.squeeze(-1))
        d_fx_mean = masked_mean(
            d_fx, valid_sentences_mask.squeeze(-1).squeeze(-1)
        )  # For d_fx, adjust mask as needed

        def mean_across_sentences(tensor):
            valid_sentences_tensor = (
                tensor * valid_sentences_mask.squeeze(-1).float()
            )  # Adjust for single feature dimension
            sum_tensor = valid_sentences_tensor.sum(dim=1)
            count_tensor = valid_sentences_mask.sum(dim=1)
            mean_tensor = sum_tensor / count_tensor.clamp(min=1)
            return mean_tensor

        # Aggregate means across sentences
        d_p_final = mean_across_sentences(d_p_mean)
        d_a0_final = mean_across_sentences(d_a0_mean)
        d_a1_final = mean_across_sentences(d_a1_mean)
        d_fx_final = mean_across_sentences(d_fx_mean)

        # Combine and normalize the final descriptor
        w_u = (d_p_final + d_a0_final + d_a1_final + d_fx_final) / 4
        y_hat_u = w_u.sum(dim=1)

        if self.sentence_prediction_method == "friss":
            # Concatenate vs with frameaxis_data if sentence_prediction_method is False
            vs = torch.cat([vs, frameaxis_data], dim=-1)

        ws = self.dropout_1(vs)
        ws = self.relu(self.Wr(vs))  # vs should have shape [B, S, D_w]

        # Apply attention mask to ws before aggregation
        mask = valid_sentences_mask.unsqueeze(-1).unsqueeze(-1)
        num_spans = ws.size(1)
        mask = mask.expand(-1, -1, num_spans, 1)

        ws *= mask  # Mask out padded sentences

        # Summing masked embeddings and computing mean across sentences
        summed_embeddings = ws.sum(dim=1)
        valid_sentences_count = mask.sum(dim=1)
        document_representation = summed_embeddings / valid_sentences_count.clamp(
            min=1
        )  # Avoid division by zero

        # Apply second dropout and output layer
        document_representation = self.dropout_2(document_representation)
        logits = self.Wt(document_representation)
        y_hat_s = self.softmax(logits)

        if self.combine_method == "sum":
            # Sum the two predictions
            combined = y_hat_u + y_hat_s
        elif self.combine_method == "avg":
            # Average the two predictions
            combined = (y_hat_u + y_hat_s) / 2

        return y_hat_u, y_hat_s, combined
