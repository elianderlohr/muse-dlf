import torch
import torch.nn as nn
import torch.nn.functional as F
from model.oldmuse.combined_autoencoder import CombinedAutoencoder
from model.oldmuse.frameaxis_autoencoder import FrameAxisAutoencoder

from model.oldmuse.loss_module import LossModule

# Assuming you have already defined CombinedAutoencoder and its methods as provided earlier.


class MUSEFrameAxisUnsupervised(nn.Module):
    def __init__(
        self,
        D_w,
        D_h,
        K,
        frameaxis_dim,
        lambda_orthogonality,
        M,
        t,
        dropout_prob=0.3,
    ):
        super(MUSEFrameAxisUnsupervised, self).__init__()

        self.loss_fn = LossModule(lambda_orthogonality, M, t)

        self.frameaxis_autoencoder = FrameAxisAutoencoder(
            D_w, D_h, frameaxis_dim, K, dropout_prob=dropout_prob
        )

    def forward(
        self,
        v_sentence,
        v_fx,
        fx_negatives,
        tau,
    ):
        outputs_fx = self.frameaxis_autoencoder(v_fx, v_sentence, tau)

        outputs_fx["v"] = v_fx

        loss = self.loss_fn(
            outputs_fx,
            fx_negatives,
        )

        results = {
            "loss": loss,
            "fx": outputs_fx,
        }

        return results
