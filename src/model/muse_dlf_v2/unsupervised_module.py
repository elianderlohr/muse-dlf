import torch
import torch.nn as nn
import torch.nn.functional as F
from model.muse_dlf_v2.combined_autoencoder import CombinedAutoencoder
from model.muse_dlf_v2.loss_module import LossModule


class MUSEUnsupervised(nn.Module):
    def __init__(
        self,
        embedding_dim,  # embedding dimension (e.g. RoBERTa 768)
        hidden_dim,  # hidden dimension
        num_classes,  # number of classes to predict
        # LossModule Parameters
        lambda_orthogonality,  # lambda for orthogonality loss
        M,  # M for orthogonality loss
        t,  # t for orthogonality loss
        # FrameAxisAutoencoder Parameters
        num_layers=2,  # number of layers in the encoder
        dropout_prob=0.3,  # dropout probability
        activation="relu",  # activation function (relu, gelu, leaky_relu, elu)
        use_batch_norm=True,  # whether to use batch normalization
        matmul_input="g",  # g or d (g = gumbel-softmax, d = softmax)
        gumbel_softmax_hard=False,  # whether to use hard gumbel softmax
        gumbel_softmax_log=False,  # whether to use log gumbel softmax
    ):
        super(MUSEUnsupervised, self).__init__()

        self.combined_autoencoder = CombinedAutoencoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            activation=activation,
            use_batch_norm=use_batch_norm,
            matmul_input=matmul_input,
            hard=gumbel_softmax_hard,
            log=gumbel_softmax_log,
        )

        self.loss_fn = LossModule(lambda_orthogonality, M, t)

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
