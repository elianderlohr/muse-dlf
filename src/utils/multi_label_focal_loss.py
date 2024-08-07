import torch
from torch import Tensor
from torch import nn
from typing import Optional, Dict
from torch.nn import functional as F


class MultiLabelFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        epsilon: float = 1e-6,
        scale: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon
        self.scale = scale

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(x)

        # Clip probabilities to prevent log(0)
        p = torch.clamp(p, self.epsilon, 1 - self.epsilon)

        # Calculate binary cross entropy loss
        BCE_loss = F.binary_cross_entropy(p, y, reduction="none")

        # Calculate modulating factor
        pt = torch.where(y == 1, p, 1 - p)
        modulating_factor = (1.0 - pt) ** self.gamma

        # Apply alpha factor
        alpha_factor = self.alpha * y + (1 - self.alpha) * (1 - y)

        # Calculate focal loss
        focal_loss = alpha_factor * modulating_factor * BCE_loss

        # Scale the loss
        focal_loss = self.scale * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def prepare_alpha(
    class_freq_dict: Dict[str, float], min_freq: float = 0.01, device: str = "cpu"
) -> Tensor:
    # Adjust frequencies and normalize
    adjusted_freqs = {k: max(v, min_freq) for k, v in class_freq_dict.items()}
    total = sum(adjusted_freqs.values())
    adjusted_freqs = {k: v / total for k, v in adjusted_freqs.items()}

    # Calculate alpha as the inverse of adjusted frequencies
    alpha_dict = {k: 1 - v for k, v in adjusted_freqs.items()}

    # Convert to tensor
    alpha = torch.tensor(list(alpha_dict.values()), device=device)

    return alpha


def multi_label_focal_loss(
    class_freq_dict: Dict[str, float],
    min_freq: float = 0.01,
    gamma: float = 2.0,
    reduction: str = "mean",
    scale: float = 1.0,
    device: str = "cpu",
) -> MultiLabelFocalLoss:
    alpha = prepare_alpha(class_freq_dict, min_freq, device)
    return MultiLabelFocalLoss(
        alpha=alpha, gamma=gamma, reduction=reduction, scale=scale
    )
