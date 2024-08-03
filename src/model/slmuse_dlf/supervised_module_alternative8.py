import torch
import torch.nn as nn
from utils.logging_manager import LoggerManager


class SLMuSESupervisedAlternative8(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes,
        frameaxis_dim,
        num_sentences,
        hidden_dim,
        dropout_prob=0.3,
        concat_frameaxis=False,
        _debug=False,
    ):
        super(SLMuSESupervisedAlternative8, self).__init__()

        self.logger = LoggerManager.get_logger(__name__)
        self.embedding_dim = embedding_dim
        self.frameaxis_dim = frameaxis_dim
        self.num_classes = num_classes
        self.concat_frameaxis = concat_frameaxis
        self._debug = _debug

        D_h = embedding_dim + (frameaxis_dim if concat_frameaxis else 0)

        # Define feed-forward network for sentence embeddings
        self.feed_forward_sentence = nn.Sequential(
            nn.Linear(D_h, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes),
        )

        self.logger.debug("✅ SLMuSESupervisedAlternative8 successfully initialized")

    def forward(self, d_p, d_a0, d_a1, d_fx, vs, frameaxis_data):

        batch_size, num_sentences, num_args, embedding_dim = d_p.shape

        # Create masks for non-padded elements
        mask_p = (
            d_p.abs().sum(dim=-1) != 0
        ).float()  # [batch_size, num_sentences, num_args]
        mask_a0 = (d_a0.abs().sum(dim=-1) != 0).float()
        mask_a1 = (d_a1.abs().sum(dim=-1) != 0).float()
        mask_fx = (
            d_fx.abs().sum(dim=-1) != 0
        ).float()  # [batch_size, num_sentences, frameaxis_dim]

        # Calculate sentence-level averages for each view, ignoring padded elements
        d_p_sent = (d_p * mask_p.unsqueeze(-1)).sum(dim=2) / mask_p.sum(
            dim=2, keepdim=True
        ).clamp(min=1)
        d_a0_sent = (d_a0 * mask_a0.unsqueeze(-1)).sum(dim=2) / mask_a0.sum(
            dim=2, keepdim=True
        ).clamp(min=1)
        d_a1_sent = (d_a1 * mask_a1.unsqueeze(-1)).sum(dim=2) / mask_a1.sum(
            dim=2, keepdim=True
        ).clamp(min=1)

        # Handle d_fx differently as it has a different shape
        d_fx_sent = d_fx  # d_fx is already at the sentence level

        # Create sentence-level mask (True if any argument in the sentence is non-padded)
        sent_mask = (
            (mask_p.sum(dim=2) + mask_a0.sum(dim=2) + mask_a1.sum(dim=2)) > 0
        ).float()  # [batch_size, num_sentences]

        # Sum over all valid sentences per article
        d_p_article = (d_p_sent * sent_mask.unsqueeze(-1)).sum(
            dim=1
        )  # [batch_size, embedding_dim]
        d_a0_article = (d_a0_sent * sent_mask.unsqueeze(-1)).sum(dim=1)
        d_a1_article = (d_a1_sent * sent_mask.unsqueeze(-1)).sum(dim=1)
        d_fx_article = (d_fx_sent * sent_mask.unsqueeze(-1)).sum(dim=1)

        # Sum over all four views and divide by 4
        y_hat_span = (d_p_article + d_a0_article + d_a1_article + d_fx_article) / 4

        if self.concat_frameaxis:
            vs = torch.cat([vs, frameaxis_data], dim=-1)

        # Predict y for each sentence
        vs_flattened = vs.view(-1, vs.size(-1))
        y_hat_sent_per_sentence = self.feed_forward_sentence(vs_flattened)

        # Reshape back to [batch_size, num_sentences, num_classes]
        y_hat_sent_per_sentence = y_hat_sent_per_sentence.view(
            batch_size, num_sentences, -1
        )

        # Sum logits for all valid sentences to have one logit per article
        y_hat_sent = (y_hat_sent_per_sentence * sent_mask.unsqueeze(-1)).sum(dim=1)

        combined = y_hat_span + y_hat_sent

        other = {
            "predicate": d_p_article,
            "arg0": d_a0_article,
            "arg1": d_a1_article,
            "frameaxis": d_fx_article,
        }

        # print first batch of y_hat_span and y_hat_sent
        self.logger.debug(f"y_hat_span: {y_hat_span[0]}")
        self.logger.debug(f"y_hat_sent: {y_hat_sent[0]}")

        if self._debug:
            self.logger.debug(
                f"Forward pass debug info: batch_size={batch_size}, num_sentences={num_sentences}, embedding_dim={embedding_dim}"
            )
            self.logger.debug(
                f"Shapes - d_p_article: {d_p_article.shape}, d_a0_article: {d_a0_article.shape}, d_a1_article: {d_a1_article.shape}, d_fx_article: {d_fx_article.shape}"
            )
            self.logger.debug(
                f"Shapes - y_hat_span: {y_hat_span.shape}, y_hat_sent: {y_hat_sent.shape}, combined: {combined.shape}"
            )

        return y_hat_span, y_hat_sent, combined, other
