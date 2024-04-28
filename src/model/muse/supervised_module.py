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
        batch_size, num_sentences, num_args, max_arg_length = args_mask.size()

        # Initialize masks
        args_sentence_mask = args_mask.any(dim=3)
        args_frame_mask = args_mask.view(batch_size, num_sentences, -1).any(dim=2)

        # mean to dim 8, 32, ignore where mask is False
        d_p_sentence_masked = d_p * args_sentence_mask.unsqueeze(-1)
        d_p_sentence_masked = d_p_sentence_masked.mean(dim=2)

        # mean to dim 8, ignore where mask is False
        d_p_masked = d_p_sentence_masked * args_frame_mask.unsqueeze(-1)
        d_p_masked = d_p_masked.mean(dim=1)

        d_a0_sentence_masked = d_a0 * args_sentence_mask.unsqueeze(-1)
        d_a0_sentence_masked = d_a0_sentence_masked.mean(dim=2)

        d_a0_masked = d_a0_sentence_masked * args_frame_mask.unsqueeze(-1)
        d_a0_masked = d_a0_masked.mean(dim=1)

        d_a1_sentence_masked = d_a1 * args_sentence_mask.unsqueeze(-1)
        d_a1_sentence_masked = d_a1_sentence_masked.mean(dim=2)

        d_a1_masked = d_a1_sentence_masked * args_frame_mask.unsqueeze(-1)
        d_a1_masked = d_a1_masked.mean(dim=1)

        d_p_mean = d_p_masked
        d_a0_mean = d_a0_masked
        d_a1_mean = d_a1_masked

        d_fx_sentence_masked = d_fx * args_frame_mask.unsqueeze(-1)
        d_fx_mean = d_fx_sentence_masked.mean(dim=1)

        # Combine and normalize the final descriptor
        w_u = (d_p_mean + d_a0_mean + d_a1_mean + d_fx_mean) / 4
        y_hat_u = w_u.sum(dim=1)

        if self.sentence_prediction_method == "friss":
            # Concatenate vs with frameaxis_data if sentence_prediction_method is False
            vs = torch.cat([vs, frameaxis_data], dim=-1)

        ws = self.dropout_1(vs)
        ws = self.relu(self.Wr(vs))  # vs should have shape [B, S, D_w]

        # attention maks
        sentence_mask = sentence_attention_mask.any(dim=2)

        ws = ws * sentence_mask.unsqueeze(-1)
        ws = ws.mean(dim=1)

        # Apply second dropout and output layer
        document_representation = self.dropout_2(ws)
        logits = self.Wt(document_representation)
        y_hat_s = self.softmax(logits)

        if self.combine_method == "sum":
            # Sum the two predictions
            combined = y_hat_u + y_hat_s
        elif self.combine_method == "avg":
            # Average the two predictions
            combined = (y_hat_u + y_hat_s) / 2

        return y_hat_u, y_hat_s, combined
