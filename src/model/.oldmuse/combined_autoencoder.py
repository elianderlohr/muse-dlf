import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, softmax


class CombinedAutoencoder(nn.Module):
    def __init__(self, D_w, D_h, K, dropout_prob=0.3):
        super(CombinedAutoencoder, self).__init__()

        self.D_h = D_h
        self.K = K

        # Shared feed-forward layer for all views
        self.feed_forward_shared = nn.Linear(2 * D_w, D_h)

        # Unique feed-forward layers for each view
        self.feed_forward_unique = nn.ModuleDict(
            {
                "a0": nn.Linear(D_h, K),
                "p": nn.Linear(D_h, K),
                "a1": nn.Linear(D_h, K),
            }
        )

        # Initializing F matrices for each view
        self.F_matrices = nn.ParameterDict(
            {
                "a0": nn.Parameter(torch.Tensor(K, D_w)),
                "p": nn.Parameter(torch.Tensor(K, D_w)),
                "a1": nn.Parameter(torch.Tensor(K, D_w)),
            }
        )

        # init F matrices with xavier_uniform and nn.init.calculate_gain('relu')
        for _, value in self.F_matrices.items():
            nn.init.xavier_uniform_(value.data, gain=nn.init.calculate_gain("relu"))

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

    def forward(self, v_p, v_a0, v_a1, v_sentence, tau):
        h_p = self.process_through_shared(v_p, v_sentence)
        h_a0 = self.process_through_shared(v_a0, v_sentence)
        h_a1 = self.process_through_shared(v_a1, v_sentence)

        logits_p = self.feed_forward_unique["p"](h_p)
        logits_a0 = self.feed_forward_unique["a0"](h_a0)
        logits_a1 = self.feed_forward_unique["a1"](h_a1)

        d_p = torch.softmax(logits_p, dim=1)
        d_a0 = torch.softmax(logits_a0, dim=1)
        d_a1 = torch.softmax(logits_a1, dim=1)

        g_p = self.custom_gumbel_softmax(d_p, tau=tau, hard=False, log=False)
        g_a0 = self.custom_gumbel_softmax(d_a0, tau=tau, hard=False, log=False)
        g_a1 = self.custom_gumbel_softmax(d_a1, tau=tau, hard=False, log=False)

        # g_p = self.custom_gumbel_softmax(logits_p, tau=tau, hard=False, log=False)
        # g_a0 = self.custom_gumbel_softmax(logits_a0, tau=tau, hard=False, log=False)
        # g_a1 = self.custom_gumbel_softmax(logits_a1, tau=tau, hard=False, log=False)

        vhat_p = torch.matmul(g_p, self.F_matrices["p"])
        vhat_a0 = torch.matmul(g_a0, self.F_matrices["a0"])
        vhat_a1 = torch.matmul(g_a1, self.F_matrices["a1"])

        return {
            "p": {"vhat": vhat_p, "d": d_p, "g": g_p, "F": self.F_matrices["p"]},
            "a0": {"vhat": vhat_a0, "d": d_a0, "g": g_a0, "F": self.F_matrices["a0"]},
            "a1": {"vhat": vhat_a1, "d": d_a1, "g": g_a1, "F": self.F_matrices["a1"]},
        }

    def process_through_shared(self, v_z, v_sentence):
        # Concatenating v_z with the sentence embedding
        concatenated = torch.cat((v_z, v_sentence), dim=-1)

        # Applying dropout
        dropped = self.dropout1(concatenated)

        # Passing through the shared linear layer
        h_shared = self.feed_forward_shared(dropped)

        # Applying batch normalization and ReLU activation
        h_shared = self.batch_norm(h_shared)
        h_shared = self.activation(h_shared)

        # Applying dropout again
        h_shared = self.dropout2(h_shared)

        return h_shared
