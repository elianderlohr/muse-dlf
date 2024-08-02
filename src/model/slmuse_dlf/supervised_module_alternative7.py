import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging_manager import LoggerManager


class SLMuSESupervisedAlternative7(nn.Module):
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
        super(SLMuSESupervisedAlternative7, self).__init__()

        self.logger = LoggerManager.get_logger(__name__)

        self.embedding_dim = embedding_dim
        self.frameaxis_dim = frameaxis_dim
        self.num_classes = num_classes
        self.num_sentences = num_sentences
        self.concat_frameaxis = concat_frameaxis
        self._debug = _debug

        # Automatically adjust D_h to be divisible by num_heads
        D_h = embedding_dim + (frameaxis_dim if concat_frameaxis else 0)
        D_h = self.adjust_dim(D_h, num_heads)

        # Adjust hidden_dim to be divisible by num_heads
        hidden_dim = self.adjust_dim(hidden_dim, num_heads)

        self.logger.debug(f"Adjusted D_h: {D_h}, Adjusted hidden_dim: {hidden_dim}")

        # ArticleClassifier components
        self.sentence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=D_h,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout_prob,
            ),
            num_layers=num_layers,
        )

        self.layer_norm1 = nn.LayerNorm(D_h)
        self.attention = nn.MultiheadAttention(
            embed_dim=D_h, num_heads=num_heads, dropout=dropout_prob
        )
        self.layer_norm2 = nn.LayerNorm(D_h)

        self.fc1 = nn.Linear(D_h, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Linear layer to adjust input dimension if necessary
        self.input_adjust = (
            nn.Linear(embedding_dim + (frameaxis_dim if concat_frameaxis else 0), D_h)
            if D_h != embedding_dim + (frameaxis_dim if concat_frameaxis else 0)
            else nn.Identity()
        )

        self.logger.debug(f"✅ SLMuSESupervisedAlternative7 successfully initialized")

    @staticmethod
    def adjust_dim(dim, num_heads):
        return ((dim + num_heads - 1) // num_heads) * num_heads

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

        # Adjust input dimension if necessary
        vs = self.input_adjust(vs)

        # Process vs through the ArticleClassifier components
        vs = vs.permute(1, 0, 2)  # [num_sentences, batch_size, embedding_dim]

        vs = self.sentence_encoder(vs)
        vs = self.layer_norm1(vs)

        attn_output, _ = self.attention(vs, vs, vs)
        vs = vs + attn_output
        vs = self.layer_norm2(vs)

        vs = torch.mean(vs, dim=0)  # [batch_size, embedding_dim]

        vs = F.relu(self.fc1(vs))
        vs = self.dropout1(vs)
        y_hat_s = self.fc2(vs)

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
