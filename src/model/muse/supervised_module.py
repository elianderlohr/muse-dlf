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
        # Print shapes of all inputs for debugging
        print("d_p shape:", d_p.shape)
        print("d_a0 shape:", d_a0.shape)
        print("d_a1 shape:", d_a1.shape)
        print("d_fx shape:", d_fx.shape)
        print("vs shape:", vs.shape)
        print("frameaxis_data shape:", frameaxis_data.shape)
        print("sentence_attention_mask shape:", sentence_attention_mask.shape)
        print("args_mask shape:", args_mask.shape)

        masked_d_p = d_p * args_mask.unsqueeze(-1)
        masked_d_a0 = d_a0 * args_mask.unsqueeze(-1)
        masked_d_a1 = d_a1 * args_mask.unsqueeze(-1)

        print("masked_d_p shape:", masked_d_p.shape)
        print("masked_d_a0 shape:", masked_d_a0.shape)
        print("masked_d_a1 shape:", masked_d_a1.shape)

        d_p_mean = masked_d_p.sum(dim=[2, 3]) / args_mask.sum(dim=[2, 3]).unsqueeze(
            -1
        ).clamp(min=1)
        d_a0_mean = masked_d_a0.sum(dim=[2, 3]) / args_mask.sum(dim=[2, 3]).unsqueeze(
            -1
        ).clamp(min=1)
        d_a1_mean = masked_d_a1.sum(dim=[2, 3]) / args_mask.sum(dim=[2, 3]).unsqueeze(
            -1
        ).clamp(min=1)

        print("d_p_mean shape:", d_p_mean.shape)
        print("d_a0_mean shape:", d_a0_mean.shape)
        print("d_a1_mean shape:", d_a1_mean.shape)

        frame_level_mask = args_mask.any(dim=3)
        print("frame_level_mask shape:", frame_level_mask.shape)

        frame_level_mask_d_fx = frame_level_mask.unsqueeze(-1).repeat(
            1, 1, 1, 15 // 10 + 1
        )[:, :, :, :15]

        print("frame_level_mask_d_fx shape:", frame_level_mask_d_fx.shape)

        masked_d_fx = d_fx * frame_level_mask_d_fx.float()

        print("masked_d_fx shape:", masked_d_fx.shape)

        d_fx_mean = masked_d_fx.sum(dim=2) / frame_level_mask_d_fx.sum(dim=2).clamp(
            min=1
        )

        print("d_fx_mean shape:", d_fx_mean.shape)

        # Combine and normalize the final descriptor
        w_u = (d_p_mean + d_a0_mean + d_a1_mean + d_fx_mean) / 4
        y_hat_u = w_u.sum(dim=1)

        if self.sentence_prediction_method == "friss":
            # Concatenate vs with frameaxis_data if sentence_prediction_method is False
            vs = torch.cat([vs, frameaxis_data], dim=-1)

        ws = self.dropout_1(vs)
        ws = self.relu(self.Wr(vs))  # vs should have shape [B, S, D_w]

        ws *= sentence_attention_mask

        # Summing masked embeddings and computing mean across sentences
        summed_embeddings = ws.sum(dim=1)
        valid_sentences_count = sentence_attention_mask.sum(dim=1)
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
