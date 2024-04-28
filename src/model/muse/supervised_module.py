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
        sentence_prediction_method="friss",  # friss or custom
    ):
        super(MUSESupervised, self).__init__()

        self.D_w = D_w
        self.frameaxis_dim = frameaxis_dim

        self.dropout_1 = nn.Dropout(dropout_prob)
        self.dropout_2 = nn.Dropout(dropout_prob)

        wr_shape = (
            D_w + (frameaxis_dim if sentence_prediction_method == "custom" else 0)
        ) * num_sentences

        self.Wr = nn.Linear(wr_shape, D_w * num_sentences)
        self.Wt = nn.Linear(D_w * num_sentences, K * num_sentences)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten(start_dim=1)

        self.sentence_prediction_method = sentence_prediction_method

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
        batch_size, num_sentences, _, _ = args_mask.size()

        # Initialize masks
        args_sentence_mask = args_mask.any(dim=3)

        # mean to dim 8, 32, ignore where mask is False
        d_p_sentence_masked = d_p * args_sentence_mask.unsqueeze(-1)
        d_p_sentence = d_p_sentence_masked.mean(dim=2)

        d_a0_sentence_masked = d_a0 * args_sentence_mask.unsqueeze(-1)
        d_a0_sentence = d_a0_sentence_masked.mean(dim=2)

        d_a1_sentence_masked = d_a1 * args_sentence_mask.unsqueeze(-1)
        d_a1_sentence = d_a1_sentence_masked.mean(dim=2)

        # Combine and normalize the final descriptor
        w_u = (d_p_sentence + d_a0_sentence + d_a1_sentence + d_fx) / 4
        y_hat_u = w_u.sum(dim=1)

        if self.sentence_prediction_method == "custom":
            vs = torch.cat([vs, frameaxis_data], dim=-1)

        ws_flattened = self.flatten(vs)

        ws = self.dropout_1(ws_flattened)
        ws = self.relu(self.Wr(ws))

        ws = self.dropout_2(ws)
        ws = self.Wt(ws)

        ws_unflatten = ws.view(batch_size, num_sentences, -1)

        # attention maks
        sentence_mask = sentence_attention_mask.any(dim=2)

        ws_unflatten = ws_unflatten * sentence_mask.unsqueeze(-1)
        y_hat_s = ws_unflatten.sum(dim=1) / sentence_mask.sum(dim=1).unsqueeze(-1)

        # Sum the two predictions
        combined = y_hat_u + y_hat_s

        return y_hat_u, y_hat_s, combined
