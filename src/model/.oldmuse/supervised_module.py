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

        D_h = D_w + (frameaxis_dim if sentence_prediction_method == "custom" else 0)

        # Feed-forward networks for sentence embeddings
        self.feed_forward_sentence = nn.Sequential(
            nn.Linear(
                D_h * num_sentences,
                D_h * num_sentences,
            ),
            nn.BatchNorm1d(D_h * num_sentences),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(
                D_h * num_sentences,
                D_w,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(D_w, num_frames),
        )

        # Flatten
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
    ):
        d_p_mean = torch.mean(d_p, dim=2).mean(dim=1)
        d_a0_mean = torch.mean(d_a0, dim=2).mean(dim=1)
        d_a1_mean = torch.mean(d_a1, dim=2).mean(dim=1)

        d_fx_mean = torch.mean(d_fx, dim=1)

        # Combine and normalize the final descriptor
        y_hat_u = (d_p_mean + d_a0_mean + d_a1_mean + d_fx_mean) / 4
        # y_hat_u = w_u.sum(dim=1)

        if self.sentence_prediction_method == "custom":
            vs = torch.cat([vs, frameaxis_data], dim=-1)

        # reshape vs from [batch_size, num_sentences, D_w] to [batch_size * num_sentences, D_w]
        ws_flattened = self.flatten(vs)

        y_hat_s = self.feed_forward_sentence(ws_flattened)

        # Sum the two predictions
        combined = y_hat_u + y_hat_s

        other = {
            "predicate": d_p_mean,
            "arg0": d_a0_mean,
            "arg1": d_a1_mean,
            "frameaxis": d_fx_mean,
        }

        return y_hat_u, y_hat_s, combined, other
