import torch
import torch.nn as nn
from model.muse_dlf.helper import custom_gumbel_sigmoid

from utils.logging_manager import LoggerManager


class MuSEFrameAxisAutoencoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        frameaxis_dim,
        hidden_dim,
        num_classes,
        num_layers=2,
        dropout_prob=0.3,
        activation="relu",
        use_batch_norm=True,
        matmul_input="g",
        log=False,
        _debug=False,
    ):
        super(MuSEFrameAxisAutoencoder, self).__init__()

        self.logger = LoggerManager.get_logger(__name__)

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.matmul_input = matmul_input
        self.log = log

        self.activation_func = self._get_activation(activation)

        input_dim = embedding_dim + frameaxis_dim

        # Shared layer components
        self.dropout1 = nn.Dropout(dropout_prob)
        self.feed_forward_shared = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = (
            nn.BatchNorm1d(hidden_dim)
            if use_batch_norm
            else nn.Identity()  # Revert back to BatchNorm1d if it doesn't work
        )
        self.activation = self.activation_func
        self.dropout2 = nn.Dropout(dropout_prob)

        # Combined individual and output layers
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    (
                        nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
                    ),  # Revert back to BatchNorm1d if it doesn't work
                    self.activation_func,
                    nn.Dropout(dropout_prob),
                ]
            )
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.feed_forward_combined = nn.Sequential(*layers)

        self.F = nn.Parameter(torch.Tensor(num_classes, frameaxis_dim))
        nn.init.xavier_uniform_(self.F.data, gain=nn.init.calculate_gain("relu"))

        self._debug = _debug

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

    def process_through_first(self, v_z, v_sentence):
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
        v_frameaxis,
        mask,
        v_sentence,
        tau,
    ):
        h = self.process_through_first(v_frameaxis, v_sentence)

        if torch.isnan(h).any():
            self.logger.error("❌ NaNs detected in h")
            raise ValueError("NaNs detected in h")

        # Pass through combined individual and output layers
        logits = self.feed_forward_combined(h) * mask.unsqueeze(-1).float()

        epsilon = 1e-10
        logits = logits + (1 - mask.unsqueeze(-1).float()) * epsilon

        if (logits == 0).all() or (logits.std() == 0):
            self.logger.debug(
                f"❌ logits has mean {logits.mean().item()} or std {logits.std().item()}"
            )

        if torch.isnan(logits).any():
            self.logger.error("❌ NaNs detected in logits")
            raise ValueError("NaNs detected in logits")

        d = torch.sigmoid(logits) * mask.unsqueeze(-1).float()

        del logits

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

        del h
        torch.cuda.empty_cache()

        return {"vhat": vhat, "d": d, "g": g, "F": self.F}
