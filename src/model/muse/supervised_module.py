import torch
import torch.nn as nn
import torch.nn.functional as F


class MUSESupervised(nn.Module):
    def __init__(
        self, D_w, K, num_frames, frameaxis_dim, num_sentences, dropout_prob=0.3
    ):
        super(MUSESupervised, self).__init__()

        self.D_w = D_w
        self.frameaxis_dim = frameaxis_dim

        # Feed-forward networks for sentence embeddings
        self.feed_forward_sentence = nn.Sequential(
            nn.Linear((D_w + frameaxis_dim) * num_sentences, D_w * num_sentences),
            nn.BatchNorm1d(D_w * num_sentences),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(D_w * num_sentences, D_w),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(D_w, num_frames),
        )

        # Flatten
        self.flatten = nn.Flatten(start_dim=1)

        # ReLU activation
        self.relu = nn.ReLU()

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(D_w)

    def forward(self, d_p, d_a0, d_a1, d_fx, vs, frameaxis_data):
        # Aggregate the SRL descriptors to have one descriptor per sentence
        d_p = d_p.mean(dim=2)
        d_a0 = d_a0.mean(dim=2)
        d_a1 = d_a1.mean(dim=2)
        # d_fx = d_fx.mean(dim=2) # This is not needed as d_fx is already aggregated

        # Take the mean over descriptors
        w_u = (d_p + d_a0 + d_a1 + d_fx) / 4
        w_u = w_u.sum(dim=1)

        # Sentence-based Classification
        # Concatenate sentence embeddings with frameaxis data
        combined_sentence_input = torch.cat((vs, frameaxis_data), dim=-1)

        combined_sentence_input = self.flatten(combined_sentence_input)

        ws = self.feed_forward_sentence(combined_sentence_input)

        # Combined SRL and sentence prediction
        combined = w_u + ws

        return w_u, ws, combined
