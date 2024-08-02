import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    """Multi-Label Focal Loss for multi-label classification tasks.

    This loss function is an adaptation of Focal Loss for multi-label scenarios.
    It is useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain multi-hot encoded labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-6,
    ):
        """Constructor.

        Args:
            alpha (float, optional): A balancing factor.
                Defaults to 1.0.
            gamma (float, optional): Focusing parameter for modulating factor (1-p).
                Defaults to 2.0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            eps (float, optional): A small value to avoid numerical instability.
                Defaults to 1e-6.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.permute(0, *range(2, y.ndim), 1).reshape(-1, c)

        # Compute sigmoid of predictions
        p = torch.sigmoid(x)

        # Clip the prediction value to prevent any division by 0 error
        p = torch.clamp(p, self.eps, 1 - self.eps)

        # Calculate the focal loss
        loss = -self.alpha * (
            y * ((1 - p) ** self.gamma) * torch.log(p)
            + (1 - y) * (p**self.gamma) * torch.log(1 - p)
        )

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "reduction", "eps"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"


def multi_label_focal_loss(
    alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean", eps: float = 1e-6
) -> MultiLabelFocalLoss:
    """Factory function for MultiLabelFocalLoss.

    Args:
        alpha (float, optional): A balancing factor.
            Defaults to 1.0.
        gamma (float, optional): Focusing parameter for modulating factor (1-p).
            Defaults to 2.0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        eps (float, optional): A small value to avoid numerical instability.
            Defaults to 1e-6.

    Returns:
        A MultiLabelFocalLoss object
    """
    return MultiLabelFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction, eps=eps)
