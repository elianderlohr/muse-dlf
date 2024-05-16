import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, softmax

from utils.logging_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class CombinedAutoencoder(nn.Module):
    def __init__(
        self,
        embedding_dim,  # embedding dimension (e.g. RoBERTa 768)
        hidden_dim,  # hidden dimension
        num_classes,  # number of classes to predict
        num_layers=2,  # number of layers in the encoder
        dropout_prob=0.3,  # dropout probability
        activation="relu",  # activation function (relu, gelu, leaky_relu, elu)
        use_batch_norm=True,  # whether to use batch normalization
        matmul_input="g",  # g or d (g = gumbel-softmax, d = softmax)
        hard=False,  # whether to use hard gumbel softmax
        log=False,  # whether to use log gumbel softmax
        _debug=False,
    ):
        super(CombinedAutoencoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.matmul_input = matmul_input
        self.hard = hard
        self.log = log

        # Initialize activation function
        self.activation_func = self._get_activation(activation)

        # Determine input dimension based on whether to concatenate frameaxis with sentence
        input_dim = 2 * embedding_dim

        # Initialize the layers for the shared encoder
        self.encoder_shared = nn.ModuleList()
        self.batch_norms_shared = nn.ModuleList()

        for i in range(num_layers):
            self.encoder_shared.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms_shared.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim

        # Unique feed-forward layers for each view
        self.feed_forward_unique = nn.ModuleDict(
            {
                "a0": nn.Linear(hidden_dim, num_classes),
                "p": nn.Linear(hidden_dim, num_classes),
                "a1": nn.Linear(hidden_dim, num_classes),
            }
        )

        # Initializing F matrices for each view
        self.F_matrices = nn.ParameterDict(
            {
                "a0": nn.Parameter(torch.Tensor(num_classes, embedding_dim)),
                "p": nn.Parameter(torch.Tensor(num_classes, embedding_dim)),
                "a1": nn.Parameter(torch.Tensor(num_classes, embedding_dim)),
            }
        )

        # Init F matrices with xavier_uniform and nn.init.calculate_gain('relu')
        for _, value in self.F_matrices.items():
            nn.init.xavier_uniform_(value.data, gain=nn.init.calculate_gain("relu"))

        # Additional layers and parameters
        self.dropout = nn.Dropout(dropout_prob)

        self._debug = _debug

        if self._debug:
            logger.debug(
                f"CombinedAutoencoder initialized with parameters: {self.__dict__}"
            )

    def _get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

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

        if self._debug:
            logger.debug(f"h_p: {h_p.shape}")
            logger.debug(f"h_a0: {h_a0.shape}")
            logger.debug(f"h_a1: {h_a1.shape}")

        logits_p = self.feed_forward_unique["p"](h_p)
        logits_a0 = self.feed_forward_unique["a0"](h_a0)
        logits_a1 = self.feed_forward_unique["a1"](h_a1)

        if self._debug:
            logger.debug(f"logits_p: {logits_p.shape}")
            logger.debug(f"logits_a0: {logits_a0.shape}")
            logger.debug(f"logits_a1: {logits_a1.shape}")

        d_p = torch.softmax(logits_p, dim=1)
        d_a0 = torch.softmax(logits_a0, dim=1)
        d_a1 = torch.softmax(logits_a1, dim=1)

        g_p = self.custom_gumbel_softmax(d_p, tau=tau, hard=self.hard, log=self.log)
        g_a0 = self.custom_gumbel_softmax(d_a0, tau=tau, hard=self.hard, log=self.log)
        g_a1 = self.custom_gumbel_softmax(d_a1, tau=tau, hard=self.hard, log=self.log)

        if self.matmul_input == "d":
            vhat_p = torch.matmul(d_p, self.F_matrices["p"])
            vhat_a0 = torch.matmul(d_a0, self.F_matrices["a0"])
            vhat_a1 = torch.matmul(d_a1, self.F_matrices["a1"])
        elif self.matmul_input == "g":
            vhat_p = torch.matmul(g_p, self.F_matrices["p"])
            vhat_a0 = torch.matmul(g_a0, self.F_matrices["a0"])
            vhat_a1 = torch.matmul(g_a1, self.F_matrices["a1"])
        else:
            raise ValueError(
                f"matmul_input must be 'd' or 'g'. Got: {self.matmul_input}"
            )

        if self._debug:
            logger.debug(f"vhat_p: {vhat_p.shape}")
            logger.debug(f"vhat_a0: {vhat_a0.shape}")
            logger.debug(f"vhat_a1: {vhat_a1.shape}")

        return {
            "p": {"vhat": vhat_p, "d": d_p, "g": g_p, "F": self.F_matrices["p"]},
            "a0": {"vhat": vhat_a0, "d": d_a0, "g": g_a0, "F": self.F_matrices["a0"]},
            "a1": {"vhat": vhat_a1, "d": d_a1, "g": g_a1, "F": self.F_matrices["a1"]},
        }

    def process_through_shared(self, v_z, v_sentence):
        x = torch.cat((v_z, v_sentence), dim=-1)

        # Passing through the shared encoder layers
        for i in range(self.num_layers):
            x = self.encoder_shared[i](x)
            x = self.activation_func(x)
            if self.use_batch_norm:
                x = self.batch_norms_shared[i](x)
            x = self.dropout(x)

        return x
