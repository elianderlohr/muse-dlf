import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional


class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label version of Focal Loss.

    Focal Loss, as described in https://arxiv.org/abs/1708.02002, adapted for multi-label tasks.
    It is useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain multi-hot encoded target labels.

    Shape:
        - x: (batch_size, C) where C is the number of classes.
        - y: (batch_size, C), same shape as the input.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            reduction (str, optional): 'mean', 'sum' or 'none'. Defaults to 'mean'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        # Initialize the BCEWithLogitsLoss
        self.bce_with_logits = nn.BCEWithLogitsLoss(weight=alpha, reduction="none")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Compute binary cross-entropy loss
        bce_loss = self.bce_with_logits(x, y)

        # Compute the focal term
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt) ** self.gamma

        # Compute the final focal loss
        loss = focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def multi_label_focal_loss(
    alpha: Optional[Tensor] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
    device="cpu",
    dtype=torch.float32,
) -> MultiLabelFocalLoss:
    """
    Factory function for MultiLabelFocalLoss.

    Args:
        alpha (Tensor, optional): Weights for each class. Defaults to None.
        gamma (float, optional): Focusing parameter. Defaults to 2.0.
        reduction (str, optional): 'mean', 'sum' or 'none'. Defaults to 'mean'.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to. Defaults to torch.float32.

    Returns:
        A MultiLabelFocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    return MultiLabelFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)


# Example usage:
# num_classes = 14  # Number of classes in your problem
# alpha = torch.tensor([1.0] * num_classes)  # Equal weight for all classes
# criterion = multi_label_focal_loss(alpha=alpha, gamma=2.0, reduction='mean', device='cuda')

# In your training loop:
# outputs = model(inputs)  # shape: (batch_size, num_classes)
# targets = target_labels  # shape: (batch_size, num_classes), multi-hot encoded
# loss = criterion(outputs, targets)
