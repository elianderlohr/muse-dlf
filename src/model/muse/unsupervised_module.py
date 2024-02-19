import torch
import torch.nn as nn
import torch.nn.functional as F
from model.muse.combined_autoencoder import CombinedAutoencoder
from model.muse.frameaxis_autoencoder import FrameAxisAutoencoder

from model.muse.loss_module import LossModule

# Assuming you have already defined CombinedAutoencoder and its methods as provided earlier.


class MUSEUnsupervised(nn.Module):
    def __init__(
        self,
        D_w,
        D_h,
        K,
        frameaxis_dim,
        num_frames,
        lambda_orthogonality,
        M,
        t,
        dropout_prob=0.3,
    ):
        super(MUSEUnsupervised, self).__init__()

        self.loss_fn = LossModule(lambda_orthogonality, M, t)

        # Using the CombinedAutoencoder instead of individual Autoencoders
        self.combined_autoencoder = CombinedAutoencoder(
            D_w, D_h, K, dropout_prob=dropout_prob
        )

        self.frameaxis_autoencoder = FrameAxisAutoencoder(
            D_w, D_h, frameaxis_dim, K, dropout_prob=dropout_prob
        )

    def forward(
        self,
        v_p,
        v_a0,
        v_a1,
        v_sentence,
        v_fx,
        p_negatives,
        a0_negatives,
        a1_negatives,
        fx_negatives,
        tau,
    ):
        outputs = self.combined_autoencoder(v_p, v_a0, v_a1, v_sentence, tau)

        outputs_fx = self.frameaxis_autoencoder(v_fx, v_sentence, tau)

        outputs_p = outputs["p"]
        outputs_p["v"] = v_p

        outputs_a0 = outputs["a0"]
        outputs_a0["v"] = v_a0

        outputs_a1 = outputs["a1"]
        outputs_a1["v"] = v_a1

        outputs_fx["v"] = v_fx

        loss = self.loss_fn(
            outputs_p,
            outputs_a0,
            outputs_a1,
            outputs_fx,
            p_negatives,
            a0_negatives,
            a1_negatives,
            fx_negatives,
        )

        results = {
            "loss": loss,
            "p": outputs["p"],
            "a0": outputs["a0"],
            "a1": outputs["a1"],
            "fx": outputs_fx,
        }

        return results
