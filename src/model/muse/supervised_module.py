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

        wr_shape = D_w + (
            frameaxis_dim if sentence_prediction_method == "custom" else 0
        )

        self.feed_forward_sentence = nn.Sequential(
            nn.Linear(wr_shape, D_w),
            nn.BatchNorm1d(D_w),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(D_w, D_w),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(D_w, num_frames),
        )

        self.sentence_prediction_method = sentence_prediction_method

    def forward(
        self,
        d_p,
        d_a0,
        d_a1,
        d_fx,
        vs,
        frameaxis_data,
    ):
        batch_size, num_sentences, _ = vs.size()

        d_p_mean = d_p.mean(dim=2).mean(dim=1)
        d_a0_mean = d_a0.mean(dim=2).mean(dim=1)
        d_a1_mean = d_a1.mean(dim=2).mean(dim=1)

        d_fx_mean = d_fx.mean(dim=1)

        # Combine and normalize the final descriptor
        y_hat_u = (d_p_mean + d_a0_mean + d_a1_mean + d_fx_mean) / 4

        if self.sentence_prediction_method == "custom":
            vs = torch.cat([vs, frameaxis_data], dim=-1)

        # reshape vs from [batch_size, num_sentences, D_w] to [batch_size * num_sentences, D_w]
        ws_flattened = vs.view(-1, vs.size(-1))

        ws = self.feed_forward_sentence(ws_flattened)

        ws_unflatten = ws.view(batch_size, num_sentences, -1)

        y_hat_s = ws_unflatten.mean(dim=1)

        # Sum the two predictions
        combined = y_hat_u + y_hat_s

        return y_hat_u, y_hat_s, combined
