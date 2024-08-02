import torch
import torch.nn as nn
import torch.nn.functional as F
from model.muse_dlf.combined_autoencoder import MuSECombinedAutoencoder
from model.muse_dlf.loss_module import MuSELossModule

from utils.logging_manager import LoggerManager


class MuSEUnsupervised(nn.Module):
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
        gumbel_softmax_log=False,  # whether to use log gumbel softmax
        _debug=False,
    ):
        super(MuSEUnsupervised, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.combined_autoencoder = MuSECombinedAutoencoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            activation=activation,
            use_batch_norm=use_batch_norm,
            matmul_input=matmul_input,
            log=gumbel_softmax_log,
            _debug=_debug,
        )

        self.loss_fn = MuSELossModule(lambda_orthogonality, M, t, _debug=_debug)

        self._debug = _debug

        # Debugging:
        self.logger.debug(f"✅ MuSEUnsupervised successfully initialized")

    def forward(
        self,
        v_p,
        v_a0,
        v_a1,
        mask_p,
        mask_a0,
        mask_a1,
        v_sentence,
        p_negatives,
        a0_negatives,
        a1_negatives,
        tau,
    ):
        # outputs = {
        # "p": {"vhat": vhat_p, "d": d_p, "g": g_p, "F": self.F_matrices["p"]},
        # "a0": {"vhat": vhat_a0, "d": d_a0, "g": g_a0, "F": self.F_matrices["a0"]},
        # "a1": {"vhat": vhat_a1, "d": d_a1, "g": g_a1, "F": self.F_matrices["a1"]},
        # }
        outputs = self.combined_autoencoder(
            v_p,
            v_a0,
            v_a1,
            mask_p,
            mask_a0,
            mask_a1,
            v_sentence,
            tau,
        )

        outputs_p = outputs["p"]
        outputs_p["v"] = v_p

        outputs_a0 = outputs["a0"]
        outputs_a0["v"] = v_a0

        outputs_a1 = outputs["a1"]
        outputs_a1["v"] = v_a1

        loss_p = self.loss_fn(
            outputs_p,
            p_negatives,
            mask_p,
        )

        loss_a0 = self.loss_fn(
            outputs_a0,
            a0_negatives,
            mask_a0,
        )

        loss_a1 = self.loss_fn(
            outputs_a1,
            a1_negatives,
            mask_a1,
        )

        results = {
            "loss_p": loss_p,
            "loss_a0": loss_a0,
            "loss_a1": loss_a1,
            "p": outputs["p"],
            "a0": outputs["a0"],
            "a1": outputs["a1"],
        }

        return results
