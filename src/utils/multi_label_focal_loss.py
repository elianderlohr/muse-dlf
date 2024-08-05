from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    """Multi-Label Focal Loss for multi-label classification tasks.

    This implementation is adapted from the original Focal Loss to work with
    multi-label scenarios where each sample can belong to multiple classes.

    Shape:
        - x: (batch_size, C) where C is the number of classes.
        - y: (batch_size, C) where C is the number of classes.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Predicted probabilities of shape (batch_size, num_classes)
            y (Tensor): Target labels of shape (batch_size, num_classes)
        """
        eps = 1e-12

        # Ensure input is a probability distribution
        x = torch.sigmoid(x)

        # Clip probabilities to prevent log(0)
        x = torch.clamp(x, eps, 1 - eps)

        # Calculate focal loss for both positive and negative cases
        loss = -(
            y * torch.pow(1 - x, self.gamma) * torch.log(x)
            + (1 - y) * torch.pow(x, self.gamma) * torch.log(1 - x)
        )

        # Apply alpha weighting if provided
        if self.alpha is not None:
            loss = loss * self.alpha

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def multi_label_focal_loss(
    alpha: Optional[Sequence] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
    device="cpu",
    dtype=torch.float32,
) -> MultiLabelFocalLoss:
    """Factory function for MultiLabelFocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): Focusing parameter. Defaults to 2.0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A MultiLabelFocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    return MultiLabelFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
