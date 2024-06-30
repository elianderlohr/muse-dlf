import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, softmax
from utils.logging_manager import LoggerManager
from torch.cuda.amp import autocast


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
        log=False,  # whether to use log gumbel softmax
        _debug=False,
    ):
        super(CombinedAutoencoder, self).__init__()

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

        # Apply weight initialization to feed_forward_unique layers
        self.feed_forward_unique.apply(self.initialize_weights)

        # Additional layers and parameters
        self.dropout = nn.Dropout(dropout_prob)

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

    def forward(
        self,
        v_p,
        v_a0,
        v_a1,
        v_sentence,
        tau,
        mixed_precision="fp16",  # mixed precision as a parameter
    ):
        precision_dtype = (
            torch.float16
            if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
        )

        with autocast(
            enabled=mixed_precision in ["fp16", "bf16", "fp32"], dtype=precision_dtype
        ):
            if (v_p == 0).all() or v_p.std() == 0:
                self.logger.debug(
                    f"❌ v_p has mean {v_p.mean().item()} or std {v_p.std().item()}, {v_p[:, :, :5]}"
                )
            if (v_a0 == 0).all() or v_a0.std() == 0:
                self.logger.debug(
                    f"❌ v_a0 has mean {v_a0.mean().item()} or std {v_a0.std().item()}, {v_a0[:, :, :5]}"
                )
            if (v_a1 == 0).all() or v_a1.std() == 0:
                self.logger.debug(
                    f"❌ v_a1 has mean {v_a1.mean().item()} or std {v_a1.std().item()}, {v_a1[:, :, :5]}"
                )
            if (v_sentence == 0).all() or v_sentence.std() == 0:
                self.logger.debug(
                    f"❌ v_sentence has mean {v_sentence.mean().item()} or std {v_sentence.std().item()}, {v_sentence[:, :, :5]}"
                )

            h_p = self.process_through_shared(v_p, v_sentence)
            h_a0 = self.process_through_shared(v_a0, v_sentence)
            h_a1 = self.process_through_shared(v_a1, v_sentence)

            if (h_p == 0).all() or h_p.std() == 0:
                self.logger.debug(
                    f"❌ h_p has mean {h_p.mean().item()} or std {h_p.std().item()}, {h_p[:, :, :5]}"
                )
            if (h_a0 == 0).all() or h_a0.std() == 0:
                self.logger.debug(
                    f"❌ h_a0 has mean {h_a0.mean().item()} or std {h_a0.std().item()}, {h_a0[:, :, :5]}"
                )
            if (h_a1 == 0).all() or h_a1.std() == 0:
                self.logger.debug(
                    f"❌ h_a1 has mean {h_a1.mean().item()} or std {h_a1.std().item()}, {h_a1[:, :, :5]}"
                )

            if (
                torch.isnan(h_p).any()
                or torch.isnan(h_a0).any()
                or torch.isnan(h_a1).any()
            ):
                self.logger.error("❌ NaNs detected in hidden states")
                raise ValueError("NaNs detected in hidden states")

            logits_p = self.feed_forward_unique["p"](h_p)
            logits_a0 = self.feed_forward_unique["a0"](h_a0)
            logits_a1 = self.feed_forward_unique["a1"](h_a1)

            if (logits_p == 0).all() or logits_p.std() == 0:
                self.logger.debug(
                    f"❌ logits_p has mean {logits_p.mean().item()} or std {logits_p.std().item()}, {logits_p}"
                )
            if (logits_a0 == 0).all() or logits_a0.std() == 0:
                self.logger.debug(
                    f"❌ logits_a0 has mean {logits_a0.mean().item()} or std {logits_a0.std().item()}, {logits_a0}"
                )
            if (logits_a1 == 0).all() or logits_a1.std() == 0:
                self.logger.debug(
                    f"❌ logits_a1 has mean {logits_a1.mean().item()} or std {logits_a1.std().item()}, {logits_a1}"
                )

            d_p = torch.softmax(logits_p, dim=1)
            d_a0 = torch.softmax(logits_a0, dim=1)
            d_a1 = torch.softmax(logits_a1, dim=1)

            # check if d_p has mean 0 or std 0
            if (d_p == 0).all() or d_p.std() == 0:
                self.logger.debug(
                    f"❌ d_p has mean {d_p.mean().item()} or std {d_p.std().item()}, {d_p}"
                )
            if (d_a0 == 0).all() or d_a0.std() == 0:
                self.logger.debug(
                    f"❌ d_a0 has mean {d_a0.mean().item()} or std {d_a0.std().item()}, {d_a0}"
                )
            if (d_a1 == 0).all() or d_a1.std() == 0:
                self.logger.debug(
                    f"❌ d_a1 has mean {d_a1.mean().item()} or std {d_a1.std().item()}, {d_a1}"
                )

            if (
                torch.isnan(d_p).any()
                or torch.isnan(d_a0).any()
                or torch.isnan(d_a1).any()
                or (d_p == 0).all()
                or (d_a0 == 0).all()
                or (d_a1 == 0).all()
            ):
                self.logger.error("❌ NaNs detected in d or d is all 0 AFTER softmax")

            g_p = self.custom_gumbel_softmax(d_p, tau=tau, hard=False, log=self.log)
            g_a0 = self.custom_gumbel_softmax(d_a0, tau=tau, hard=False, log=self.log)
            g_a1 = self.custom_gumbel_softmax(d_a1, tau=tau, hard=False, log=self.log)

            if (g_p == 0).all() or g_p.std() == 0:
                self.logger.debug(
                    f"❌ g_p has mean {g_p.mean().item()} or std {g_p.std().item()}, {g_p}"
                )
            if (g_a0 == 0).all() or g_a0.std() == 0:
                self.logger.debug(
                    f"❌ g_a0 has mean {g_a0.mean().item()} or std {g_a0.std().item()}, {g_a0}"
                )
            if (g_a1 == 0).all() or g_a1.std() == 0:
                self.logger.debug(
                    f"❌ g_a1 has mean {g_a1.mean().item()} or std {g_a1.std().item()}, {g_a1}"
                )

            if (
                torch.isnan(g_p).any()
                or torch.isnan(g_a0).any()
                or torch.isnan(g_a1).any()
            ):
                self.logger.error(
                    f"❌ NaNs detected in g AFTER gumbel-softmax, tau: {tau}, hard: {False}, log: {self.log}"
                )

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

        if (
            torch.isnan(vhat_p).any()
            or torch.isnan(vhat_a0).any()
            or torch.isnan(vhat_a1).any()
        ):
            self.logger.error("❌ NaNs detected in vhat")
            raise ValueError("NaNs detected in vhat")

        if (vhat_p == 0).all() or vhat_p.std() == 0:
            self.logger.debug(
                f"❌ vhat_p has mean {vhat_p.mean().item()} or std {vhat_p.std().item()}, {vhat_p[0, 0, :5]}"
            )
        if (vhat_a0 == 0).all() or vhat_a0.std() == 0:
            self.logger.debug(
                f"❌ vhat_a0 has mean {vhat_a0.mean().item()} or std {vhat_a0.std().item()}, {vhat_a0[0, 0, :5]}"
            )
        if (vhat_a1 == 0).all() or vhat_a1.std() == 0:
            self.logger.debug(
                f"❌ vhat_a1 has mean {vhat_a1.mean().item()} or std {vhat_a1.std().item()}, {vhat_a1[0, 0, :5]}"
            )

        return {
            "p": {"vhat": vhat_p, "d": d_p, "g": g_p, "F": self.F_matrices["p"]},
            "a0": {"vhat": vhat_a0, "d": d_a0, "g": g_a0, "F": self.F_matrices["a0"]},
            "a1": {"vhat": vhat_a1, "d": d_a1, "g": g_a1, "F": self.F_matrices["a1"]},
        }

    def process_through_shared(self, v_z, v_sentence):
        x = torch.cat((v_z, v_sentence), dim=-1)

        for i in range(self.num_layers):
            x = self.encoder_shared[i](x)
            x = self.activation_func(x)
            if self.use_batch_norm:
                if torch.isnan(x).any():
                    self.logger.error("❌ NaNs detected in batch normalization input")
                    raise ValueError("NaNs detected in batch normalization input")
                x = self.batch_norms_shared[i](x)
            x = self.dropout(x)

        if torch.isnan(x).any():
            self.logger.error("❌ NaNs detected after processing through shared layers")
            raise ValueError("NaNs detected after processing through shared layers")

        return x
