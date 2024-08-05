import torch
import torch.nn as nn
from model.muse_dlf.helper import custom_gumbel_sigmoid
from utils.logging_manager import LoggerManager


class MuSECombinedAutoencoder(nn.Module):
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
        log=False,  # whether to use log gumbel softmax
        _debug=False,
    ):
        super(MuSECombinedAutoencoder, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.matmul_input = matmul_input
        self.log = log

        # Initialize activation function
        self.activation_func = self._get_activation(activation)

        input_dim = 2 * embedding_dim

        # Shared layer components
        self.dropout1 = nn.Dropout(dropout_prob)
        self.feed_forward_shared = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = (
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        )
        self.activation = self.activation_func
        self.dropout2 = nn.Dropout(dropout_prob)

        # Combined individual and unique layers for each view
        self.feed_forward_unique = nn.ModuleDict()
        for view in ["p", "a0", "a1"]:
            layers = []
            for i in range(num_layers):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                        self.activation_func,
                        nn.Dropout(dropout_prob),
                    ]
                )
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.feed_forward_unique[view] = nn.Sequential(*layers)

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

        # Apply weight initialization to feed_forward_unique layers
        self.feed_forward_unique.apply(self.initialize_weights)

        self._debug = _debug

        self.logger.debug(f"✅ CombinedAutoencoder successfully initialized")

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

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

    def forward(
        self,
        v_p,
        v_a0,
        v_a1,
        mask_p,
        mask_a0,
        mask_a1,
        v_sentence,
        tau,
    ):
        h_p = self.process_through_shared(v_p, v_sentence)
        h_a0 = self.process_through_shared(v_a0, v_sentence)
        h_a1 = self.process_through_shared(v_a1, v_sentence)
        logits_p = self.feed_forward_unique["p"](h_p)
        logits_a0 = self.feed_forward_unique["a0"](h_a0)
        logits_a1 = self.feed_forward_unique["a1"](h_a1)

        # Check for NaNs in logits
        if (
            torch.isnan(logits_p).any()
            or torch.isnan(logits_a0).any()
            or torch.isnan(logits_a1).any()
        ):
            self.logger.error("❌ NaNs detected in logits")
            raise ValueError("NaNs detected in logits")

        # Apply masks before softmax to avoid NaNs
        logits_p = logits_p * mask_p.unsqueeze(-1).float()
        logits_a0 = logits_a0 * mask_a0.unsqueeze(-1).float()
        logits_a1 = logits_a1 * mask_a1.unsqueeze(-1).float()

        # Ensure logits are not all zero by adding a small epsilon where mask is zero
        epsilon = 1e-10
        logits_p = logits_p + (1 - mask_p.unsqueeze(-1).float()) * epsilon
        logits_a0 = logits_a0 + (1 - mask_a0.unsqueeze(-1).float()) * epsilon
        logits_a1 = logits_a1 + (1 - mask_a1.unsqueeze(-1).float()) * epsilon

        # Apply sigmoid
        d_p = torch.sigmoid(logits_p) * mask_p.unsqueeze(-1).float()
        d_a0 = torch.sigmoid(logits_a0) * mask_a0.unsqueeze(-1).float()
        d_a1 = torch.sigmoid(logits_a1) * mask_a1.unsqueeze(-1).float()

        # Check for NaNs after sigmoid
        if torch.isnan(d_p).any() or torch.isnan(d_a0).any() or torch.isnan(d_a1).any():
            self.logger.error("❌ NaNs detected in d AFTER sigmoid")
            raise ValueError("NaNs detected in d AFTER sigmoid")

        g_p = (
            custom_gumbel_sigmoid(d_p, tau=tau, hard=False, log=self.log)
            * mask_p.unsqueeze(-1).float()
        )
        g_a0 = (
            custom_gumbel_sigmoid(d_a0, tau=tau, hard=False, log=self.log)
            * mask_a0.unsqueeze(-1).float()
        )
        g_a1 = (
            custom_gumbel_sigmoid(d_a1, tau=tau, hard=False, log=self.log)
            * mask_a1.unsqueeze(-1).float()
        )

        # Check for NaNs after gumbel softmax
        if torch.isnan(g_p).any() or torch.isnan(g_a0).any() or torch.isnan(g_a1).any():
            self.logger.error(
                f"❌ NaNs detected in g AFTER gumbel-softmax, tau: {tau}, hard: {False}, log: {self.log}"
            )
            raise ValueError(
                f"NaNs detected in g AFTER gumbel-softmax, tau: {tau}, hard: {False}, log: {self.log}"
            )

        if self.matmul_input == "d":
            vhat_p = (
                torch.matmul(d_p, self.F_matrices["p"]) * mask_p.unsqueeze(-1).float()
            )
            vhat_a0 = (
                torch.matmul(d_a0, self.F_matrices["a0"])
                * mask_a0.unsqueeze(-1).float()
            )
            vhat_a1 = (
                torch.matmul(d_a1, self.F_matrices["a1"])
                * mask_a1.unsqueeze(-1).float()
            )
        elif self.matmul_input == "g":
            vhat_p = (
                torch.matmul(g_p, self.F_matrices["p"]) * mask_p.unsqueeze(-1).float()
            )
            vhat_a0 = (
                torch.matmul(g_a0, self.F_matrices["a0"])
                * mask_a0.unsqueeze(-1).float()
            )
            vhat_a1 = (
                torch.matmul(g_a1, self.F_matrices["a1"])
                * mask_a1.unsqueeze(-1).float()
            )
        else:
            raise ValueError(
                f"matmul_input must be 'd' or 'g'. Got: {self.matmul_input}"
            )

        # Clear intermediate tensors
        del (
            h_p,
            h_a0,
            h_a1,
            logits_p,
            logits_a0,
            logits_a1,
        )
        torch.cuda.empty_cache()

        return {
            "p": {
                "vhat": vhat_p,
                "d": d_p,
                "g": g_p,
                "F": self.F_matrices["p"],
            },
            "a0": {
                "vhat": vhat_a0,
                "d": d_a0,
                "g": g_a0,
                "F": self.F_matrices["a0"],
            },
            "a1": {
                "vhat": vhat_a1,
                "d": d_a1,
                "g": g_a1,
                "F": self.F_matrices["a1"],
            },
        }
