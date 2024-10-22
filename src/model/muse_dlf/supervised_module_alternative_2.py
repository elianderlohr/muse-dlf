import torch
import torch.nn as nn
from utils.logging_manager import LoggerManager


class MuSESupervisedAlternative2(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes,
        frameaxis_dim,
        num_sentences,
        hidden_dim,
        dropout_prob=0.6,
        concat_frameaxis=False,
        _debug=False,
    ):
        super(MuSESupervisedAlternative2, self).__init__()

        self.logger = LoggerManager.get_logger(__name__)
        self.embedding_dim = embedding_dim
        self.frameaxis_dim = frameaxis_dim
        self.num_classes = num_classes
        self.concat_frameaxis = concat_frameaxis
        self._debug = _debug

        D_h = embedding_dim + (frameaxis_dim if concat_frameaxis else 0)

        self.flatten = nn.Flatten(start_dim=1)

        # Reduced number of layers and changed activation to ReLU
        self.feed_forward_sentence = nn.Sequential(
            nn.Linear(D_h * num_sentences, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm to LayerNorm
            nn.ReLU(),  # Changed from GELU to ReLU
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.logger.debug("✅ MuSESupervisedAlternative2 successfully initialized")

    def forward(self, d_p, d_a0, d_a1, d_fx, vs, frameaxis_data):
        batch_size, num_sentences, num_args, embedding_dim = d_p.shape

        d_p_flatten = d_p.view(batch_size, num_sentences * num_args, embedding_dim)
        d_a0_flatten = d_a0.view(batch_size, num_sentences * num_args, embedding_dim)
        d_a1_flatten = d_a1.view(batch_size, num_sentences * num_args, embedding_dim)

        # Create masks for non-padded elements
        mask_p = (d_p_flatten.abs().sum(dim=-1) != 0).float()
        mask_a0 = (d_a0_flatten.abs().sum(dim=-1) != 0).float()
        mask_a1 = (d_a1_flatten.abs().sum(dim=-1) != 0).float()
        mask_fx = (d_fx.abs().sum(dim=-1) != 0).float()

        # Calculate the mean ignoring padded elements
        d_p_mean = (d_p_flatten * mask_p.unsqueeze(-1)).sum(dim=1) / torch.clamp(
            mask_p.sum(dim=1, keepdim=True), min=1
        )
        d_a0_mean = (d_a0_flatten * mask_a0.unsqueeze(-1)).sum(dim=1) / torch.clamp(
            mask_a0.sum(dim=1, keepdim=True), min=1
        )
        d_a1_mean = (d_a1_flatten * mask_a1.unsqueeze(-1)).sum(dim=1) / torch.clamp(
            mask_a1.sum(dim=1, keepdim=True), min=1
        )
        d_fx_mean = (d_fx * mask_fx.unsqueeze(-1)).sum(dim=1) / torch.clamp(
            mask_fx.sum(dim=1, keepdim=True), min=1
        )

        # Combine and normalize the final descriptor
        y_hat_span = (d_p_mean + d_a0_mean + d_a1_mean + d_fx_mean) / 4

        if self.concat_frameaxis:
            vs = torch.cat([vs, frameaxis_data], dim=-1)

        flattened = self.flatten(vs)
        y_hat_sent = self.feed_forward_sentence(flattened)

        combined = y_hat_span + y_hat_sent

        other = {
            "predicate": d_p_mean,
            "arg0": d_a0_mean,
            "arg1": d_a1_mean,
            "frameaxis": d_fx_mean,
        }

        if self._debug:
            self.logger.debug(
                f"Forward pass debug info: batch_size={batch_size}, num_sentences={num_sentences}, embedding_dim={embedding_dim}"
            )
            self.logger.debug(
                f"Shapes - d_p_mean: {d_p_mean.shape}, d_a0_mean: {d_a0_mean.shape}, d_a1_mean: {d_a1_mean.shape}, d_fx_mean: {d_fx_mean.shape}"
            )
            self.logger.debug(
                f"Shapes - y_hat_span: {y_hat_span.shape}, flattened: {flattened.shape}, y_hat_sent: {y_hat_sent.shape}, combined: {combined.shape}"
            )

        return y_hat_span, y_hat_sent, combined, other
