import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from utils.logging_manager import LoggerManager


class SLMUSESupervisedAlternative7(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        num_classes=15,
        frameaxis_dim=0,
        num_sentences=24,
        hidden_dim=2048,
        dropout_prob=0.1,
        concat_frameaxis=False,
        num_heads=8,
        num_layers=2,
        _debug=False,
    ):
        super(SLMUSESupervisedAlternative7, self).__init__()

        self.logger = LoggerManager.get_logger(__name__)

        self.embedding_dim = embedding_dim
        self.frameaxis_dim = frameaxis_dim
        self.num_classes = num_classes
        self.num_sentences = num_sentences

        D_h = embedding_dim + (frameaxis_dim if concat_frameaxis else 0)

        self.num_heads = num_heads
        self.adjusted_D_h = (D_h // self.num_heads) * self.num_heads
        if self.adjusted_D_h != D_h:
            self.logger.warning(
                f"Adjusted D_h from {D_h} to {self.adjusted_D_h} to ensure divisibility by {self.num_heads} heads"
            )

        self.dim_adjuster = (
            nn.Linear(D_h, self.adjusted_D_h)
            if D_h != self.adjusted_D_h
            else nn.Identity()
        )

        self.sentence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.adjusted_D_h,
                nhead=self.num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout_prob,
                activation="gelu",
            ),
            num_layers=num_layers,
        )

        self.dim_reduction = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.adjusted_D_h * num_sentences, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 4, num_classes),  # No GELU here
        )

        self.concat_frameaxis = concat_frameaxis
        self._debug = _debug

        self.logger.debug(f"✅ SLMUSESupervisedAlternative7 successfully initialized")

    def forward(
        self,
        d_p,
        d_a0,
        d_a1,
        d_fx,
        vs,
        frameaxis_data,
    ):

        batch_size, num_sentences, num_args, embedding_dim = d_p.shape

        d_p_flatten = d_p.view(batch_size, num_sentences * num_args, embedding_dim)
        d_a0_flatten = d_a0.view(batch_size, num_sentences * num_args, embedding_dim)
        d_a1_flatten = d_a1.view(batch_size, num_sentences * num_args, embedding_dim)

        mask_p = (d_p_flatten.abs().sum(dim=-1) != 0).float()
        mask_a0 = (d_a0_flatten.abs().sum(dim=-1) != 0).float()
        mask_a1 = (d_a1_flatten.abs().sum(dim=-1) != 0).float()
        mask_fx = (d_fx.abs().sum(dim=-1) != 0).float()

        d_p_mean = self.weighted_mean(d_p_flatten, mask_p)
        d_a0_mean = self.weighted_mean(d_a0_flatten, mask_a0)
        d_a1_mean = self.weighted_mean(d_a1_flatten, mask_a1)
        d_fx_mean = self.weighted_mean(d_fx, mask_fx)

        y_hat_u = (d_p_mean + d_a0_mean + d_a1_mean + d_fx_mean) / 4

        if self.concat_frameaxis:
            vs = torch.cat([vs, frameaxis_data], dim=-1)

        vs_adjusted = self.dim_adjuster(vs)

        vs_encoded = self.sentence_encoder(vs_adjusted.transpose(0, 1)).transpose(0, 1)

        y_hat_s = self.dim_reduction(vs_encoded)

        combined = y_hat_u + y_hat_s

        other = {
            "predicate": d_p_mean,
            "arg0": d_a0_mean,
            "arg1": d_a1_mean,
            "frameaxis": d_fx_mean,
        }

        return y_hat_u, y_hat_s, combined, other

    def weighted_mean(self, tensor, mask):
        return (tensor * mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(
            mask.sum(dim=1, keepdim=True), min=1
        )
