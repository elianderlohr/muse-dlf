import torch
from torch import Tensor
from torch import nn
from typing import Optional, Dict


class MultiLabelFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        # Initialize the BCEWithLogitsLoss without weight
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Compute binary cross-entropy loss
        bce_loss = self.bce_with_logits(x, y)

        # Compute probabilities
        pt = torch.sigmoid(x)
        pt = torch.where(y == 1, pt, 1 - pt)

        # Compute the focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = torch.where(y == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_weight * focal_term * bce_loss
        else:
            focal_loss = focal_term * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def prepare_alpha(
    class_freq_dict: Dict[str, float],
    min_freq: float = 0.01,
    scaling_factor: float = 10.0,
    device: str = "cpu",
) -> Tensor:
    # Adjust frequencies and normalize
    adjusted_freqs = {k: max(v, min_freq) for k, v in class_freq_dict.items()}
    total = sum(adjusted_freqs.values())
    adjusted_freqs = {k: v / total for k, v in adjusted_freqs.items()}

    # Calculate inverse frequencies
    inverse_freqs = {k: 1 / v for k, v in adjusted_freqs.items()}

    # Scale the inverse frequencies
    max_inverse = max(inverse_freqs.values())
    alpha_dict = {
        k: (v / max_inverse) * scaling_factor for k, v in inverse_freqs.items()
    }

    # Convert to tensor
    alpha = torch.tensor(list(alpha_dict.values()), device=device)

    return alpha


def multi_label_focal_loss(
    class_freq_dict: Dict[str, float],
    min_freq: float = 0.01,
    scaling_factor: float = 10.0,
    gamma: float = 2.0,
    reduction: str = "mean",
    device: str = "cpu",
) -> MultiLabelFocalLoss:
    alpha = prepare_alpha(class_freq_dict, min_freq, scaling_factor, device)
    return MultiLabelFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
