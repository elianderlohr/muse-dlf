import torch
import torch.nn as nn
import torch.nn.functional as F
from model.oldmuse.combined_autoencoder import CombinedAutoencoder
from model.oldmuse.frameaxis_autoencoder import FrameAxisAutoencoder

from model.oldmuse.loss_module import LossModule

# Assuming you have already defined CombinedAutoencoder and its methods as provided earlier.


class MuSEUnsupervised(nn.Module):
    def __init__(
        self,
        D_w,
        D_h,
        K,
        lambda_orthogonality,
        M,
        t,
        dropout_prob=0.3,
    ):
        super(MuSEUnsupervised, self).__init__()

        self.loss_fn = LossModule(lambda_orthogonality, M, t)

        # Using the CombinedAutoencoder instead of individual Autoencoders
        self.combined_autoencoder = CombinedAutoencoder(
            D_w, D_h, K, dropout_prob=dropout_prob
        )

    def forward(
        self,
        v_p,
        v_a0,
        v_a1,
        v_sentence,
        p_negatives,
        a0_negatives,
        a1_negatives,
        tau,
    ):
        outputs = self.combined_autoencoder(v_p, v_a0, v_a1, v_sentence, tau)

        outputs_p = outputs["p"]
        outputs_p["v"] = v_p

        outputs_a0 = outputs["a0"]
        outputs_a0["v"] = v_a0

        outputs_a1 = outputs["a1"]
        outputs_a1["v"] = v_a1

        loss_p = self.loss_fn(
            outputs_p,
            p_negatives,
        )

        loss_a0 = self.loss_fn(
            outputs_a0,
            a0_negatives,
        )

        loss_a1 = self.loss_fn(
            outputs_a1,
            a1_negatives,
        )

        loss = loss_p + loss_a0 + loss_a1

        results = {
            "loss": loss,
            "p": outputs["p"],
            "a0": outputs["a0"],
            "a1": outputs["a1"],
        }

        return results
