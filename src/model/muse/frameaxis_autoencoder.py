import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, softmax


class FrameAxisAutoencoder(nn.Module):
    def __init__(self, D_w, D_h, frameaxis_dim, K, dropout_prob=0.3):
        super(FrameAxisAutoencoder, self).__init__()

        self.K = K

        print("FrameAxisAutoencoder: ", D_w, D_h, frameaxis_dim, K)

        # Shared feed-forward layer for all views
        self.feed_forward_1 = nn.Linear(D_w + frameaxis_dim, D_h)

        # Unique feed-forward layers for each view
        self.feed_forward_2 = nn.Linear(D_h, K)

        self.F = nn.Parameter(torch.Tensor(K, frameaxis_dim))

        nn.init.xavier_uniform_(self.F.data, gain=nn.init.calculate_gain("relu"))

        # Additional layers and parameters
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.batch_norm = nn.BatchNorm1d(D_h)
        self.activation = nn.ReLU()
        self.activation2 = nn.Sigmoid()

    def sample_gumbel(self, shape, eps=1e-20, device="cpu"):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, t):
        """Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits.size(), device=logits.device)
        return softmax(y / t, dim=-1)

    def gumbel_logsoftmax_sample(self, logits, t):
        """Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits.size(), device=logits.device)
        return log_softmax(y / t, dim=-1)

    def custom_gumbel_softmax(self, logits, tau, hard=False, log=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        tau: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        if log:
            y = self.gumbel_logsoftmax_sample(logits, tau)
        else:
            y = self.gumbel_softmax_sample(logits, tau)
        if hard:
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            # Set gradients w.r.t. y_hard gradients w.r.t. y
            y_hard = (y_hard - y).detach() + y
            return y_hard
        return y

    def forward(self, v_frameaxis, v_sentence, tau):
        h = self.process_through_first(v_frameaxis, v_sentence)

        logits = self.feed_forward_2(h)

        d = torch.softmax(logits, dim=1)

        g = self.custom_gumbel_softmax(d, tau=tau, hard=False, log=False)

        vhat = torch.matmul(g, self.F)

        return {"vhat": vhat, "d": d, "g": g, "F": self.F}

    def process_through_first(self, v_z, v_sentence):
        # Concatenating v_z with the sentence embedding
        concatenated = torch.cat((v_z, v_sentence), dim=-1)

        # Applying dropout
        dropped = self.dropout1(concatenated)

        # Passing through the shared linear layer
        h_shared = self.feed_forward_1(dropped)

        # Applying batch normalization and ReLU activation
        h_shared = self.batch_norm(h_shared)
        h_shared = self.activation(h_shared)

        # Applying dropout again
        h_shared = self.dropout2(h_shared)

        return h_shared
