import torch
import torch.nn as nn

"""
This implementation is based on the following paper:

@misc{benbaruch2020asymmetric, 
    title={Asymmetric Loss For Multi-Label Classification}, 
    author={Emanuel Ben-Baruch and Tal Ridnik and Nadav Zamir and Asaf Noy and Itamar Friedman and Matan Protter and Lihi Zelnik-Manor}, 
    year={2020}, 
    eprint={2009.14119},
    archivePrefix={arXiv}, 
    primaryClass={cs.CV} }

and the following code implementation from the authors of the paper: https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py

The `AsymmetricLossOptimized` was further advanced by incoporating a class specific alpha value. The `WeightedAsymmetricLoss` is the final implementation that is used in the project.
"""


class WeightedAsymmetricLoss(nn.Module):
    def __init__(
        self,
        alpha=None,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        super(WeightedAsymmetricLoss, self).__init__()
        self.alpha = alpha

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = (
            self.asymmetric_w
        ) = self.loss = None

    def forward(self, x, y):
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            self.loss = self.alpha * self.loss

        return -self.loss.mean()
