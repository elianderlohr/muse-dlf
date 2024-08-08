import torch
import torch.nn as nn


class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, alpha=None, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(WeightedAsymmetricLoss, self).__init__()
        self.alpha = alpha
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # Apply sigmoid to logits
        logits_sigmoid = torch.sigmoid(logits)

        # Apply clipping to logits sigmoid output
        if self.clip > 0:
            logits_clipped = torch.clamp(
                logits_sigmoid, min=self.clip, max=1.0 - self.clip
            )
        else:
            logits_clipped = logits_sigmoid

        # Calculate positive and negative losses
        loss_pos = targets * torch.log(logits_clipped + self.eps)
        loss_neg = (1 - targets) * torch.log(1 - logits_clipped + self.eps)

        # Apply the asymmetric focusing
        loss_pos = (1 - logits_clipped) ** self.gamma_pos * loss_pos
        loss_neg = logits_clipped**self.gamma_neg * loss_neg

        # Combine positive and negative losses
        loss = loss_pos + loss_neg

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            alpha_weight = targets * self.alpha + (1 - targets)
            loss *= alpha_weight

        return -loss.mean()
