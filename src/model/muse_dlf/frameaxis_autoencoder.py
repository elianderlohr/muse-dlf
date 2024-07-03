import torch
import torch.nn as nn
from model.muse_dlf.helper import custom_gumbel_sigmoid
from utils.logging_manager import LoggerManager

from torch.cuda.amp import autocast


class MUSEFrameAxisAutoencoder(nn.Module):
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
        log=False,  # whether to use log gumbel softmax
        _debug=False,
    ):
        super(MUSEFrameAxisAutoencoder, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.matmul_input = matmul_input
        self.log = log

        # Initialize activation function
        self.activation_func = self._get_activation(activation)

        # Determine input dimension based on whether to concatenate frameaxis with sentence
        input_dim = embedding_dim + frameaxis_dim

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

    def forward(
        self,
        v_frameaxis,
        mask,
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
            h = self.process_through_first(v_frameaxis, v_sentence, mask)

            if torch.isnan(h).any():
                self.logger.error("❌ NaNs detected in h")
                raise ValueError("NaNs detected in h")

            logits = self.feed_forward_2(h) * mask.unsqueeze(-1).float()

            if (logits == 0).all() or (logits.std() == 0):
                self.logger.debug(
                    f"❌ logits has mean {logits.mean().item()} or std {logits.std().item()}"
                )

            if torch.isnan(logits).any():
                self.logger.error("❌ NaNs detected in logits")
                raise ValueError("NaNs detected in logits")

            d = torch.sigmoid(logits) * mask.unsqueeze(-1).float()

            g = (
                custom_gumbel_sigmoid(d, tau=tau, hard=False, log=self.log)
                * mask.unsqueeze(-1).float()
            )

            if self.matmul_input == "d":
                vhat = torch.matmul(d, self.F) * mask.unsqueeze(-1).float()
            elif self.matmul_input == "g":
                vhat = torch.matmul(g, self.F) * mask.unsqueeze(-1).float()
            else:
                raise ValueError("matmul_input must be 'd' or 'g'.")

        if torch.isnan(vhat).any():
            self.logger.error("❌ NaNs detected in vhat")
            raise ValueError("NaNs detected in vhat")

        return {"vhat": vhat, "d": d, "g": g, "F": self.F}

    def process_through_first(self, v_z, v_sentence, mask):
        x = torch.cat((v_z, v_sentence), dim=-1)

        # Passing through the encoder layers
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.activation_func(x)
            if self.use_batch_norm:
                x = self.batch_norms_encoder[i](x)
            x = self.dropout(x)

        x = x * mask.unsqueeze(-1).float()

        return x
