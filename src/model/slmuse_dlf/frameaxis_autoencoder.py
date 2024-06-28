import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, softmax

from utils.logging_manager import LoggerManager

from torch.cuda.amp import autocast


class FrameAxisAutoencoder(nn.Module):
    def __init__(
        self,
        embedding_dim,  # embedding dimension (e.g. RoBERTa 768)
        frameaxis_dim,  # frameaxis dimension
        hidden_dim,  # hidden dimension
        num_classes,  # number of classes to predict
        num_layers=2,  # number of layers in the encoder
        dropout_prob=0.3,  # dropout probability
        activation="relu",  # activation function (relu, gelu, leaky_relu, elu)
        use_batch_norm=True,  # whether to use batch normalization
        matmul_input="g",  # g or d (g = gumbel-softmax, d = softmax)
        concat_frameaxis=True,  # whether to concatenate frameaxis with sentence
        log=False,  # whether to use log gumbel softmax
        _debug=False,
    ):
        super(FrameAxisAutoencoder, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.matmul_input = matmul_input
        self.concat_frameaxis = concat_frameaxis
        self.log = log

        # Initialize activation function
        self.activation_func = self._get_activation(activation)

        # Determine input dimension based on whether to concatenate frameaxis with sentence
        input_dim = embedding_dim + frameaxis_dim if concat_frameaxis else embedding_dim

        # Initialize the layers for the encoder
        self.encoder = nn.ModuleList()
        self.batch_norms_encoder = nn.ModuleList()

        for i in range(num_layers):
            self.encoder.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms_encoder.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim

        # Unique feed-forward layers for each view
        self.feed_forward_2 = nn.Linear(hidden_dim, num_classes)

        self.F = nn.Parameter(torch.Tensor(num_classes, frameaxis_dim))
        nn.init.xavier_uniform_(self.F.data, gain=nn.init.calculate_gain("relu"))

        # Additional layers and parameters
        self.dropout = nn.Dropout(dropout_prob)

        self._debug = _debug

        # Debugging
        self.logger.debug(f"✅ FrameAxisAutoencoder successfully initialized")

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
        v_frameaxis,
        v_sentence,
        tau,
        mixed_precision="fp16",  # mixed precision as a parameter
    ):
        precision_dtype = (
            torch.float16
            if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16" else None
        )

        with autocast(
            enabled=mixed_precision in ["fp16", "bf16"], dtype=precision_dtype
        ):
            h = self.process_through_first(v_frameaxis, v_sentence)

            if torch.isnan(h).any():
                self.logger.error("❌ NaNs detected in h")
                raise ValueError("NaNs detected in h")

            if (h == 0).all() or (h.std() == 0):
                self.logger.debug(
                    f"❌ h has mean {h.mean().item()} or std {h.std().item()}"
                )

            logits = self.feed_forward_2(h)

            if (logits == 0).all() or (logits.std() == 0):
                self.logger.debug(
                    f"❌ logits has mean {logits.mean().item()} or std {logits.std().item()}"
                )

            if torch.isnan(logits).any():
                self.logger.error("❌ NaNs detected in logits")
                raise ValueError("NaNs detected in logits")

            d = torch.softmax(logits, dim=1)

            if (d == 0).all() or (d.std() == 0):
                self.logger.debug(
                    f"❌ d has mean {d.mean().item()} or std {d.std().item()}"
                )

            g = self.custom_gumbel_softmax(d, tau=tau, hard=False, log=self.log)

            if (g == 0).all() or (g.std() == 0):
                self.logger.debug(
                    f"❌ g has mean {g.mean().item()} or std {g.std().item()}"
                )

            if self.matmul_input == "d":
                vhat = torch.matmul(d, self.F)
            elif self.matmul_input == "g":
                vhat = torch.matmul(g, self.F)
            else:
                raise ValueError("matmul_input must be 'd' or 'g'.")

        if (vhat == 0).all() or (vhat.std() == 0):
            self.logger.debug(
                f"❌ vhat has mean {vhat.mean().item()} or std {vhat.std().item()}"
            )

        if torch.isnan(vhat).any():
            self.logger.error("❌ NaNs detected in vhat")
            raise ValueError("NaNs detected in vhat")

        return {"vhat": vhat, "d": d, "g": g, "F": self.F}

    def process_through_first(self, v_z, v_sentence):
        # Concatenating v_z with the sentence embedding if concat_frameaxis is True
        if self.concat_frameaxis:
            x = torch.cat((v_z, v_sentence), dim=-1)
        else:
            x = v_sentence

        # Passing through the encoder layers
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.activation_func(x)
            if self.use_batch_norm:
                x = self.batch_norms_encoder[i](x)
            x = self.dropout(x)

        return x
