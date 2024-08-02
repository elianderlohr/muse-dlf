import torch
import torch.nn as nn
import torch.nn.functional as F
from model.muse_dlf.frameaxis_autoencoder import MuSEFrameAxisAutoencoder
from model.muse_dlf.loss_module import MuSELossModule

from utils.logging_manager import LoggerManager


class MuSEFrameAxisUnsupervised(nn.Module):
    def __init__(
        self,
        embedding_dim,  # embedding dimension (e.g. RoBERTa 768)
        frameaxis_dim,  # frameaxis dimension
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
        super(MuSEFrameAxisUnsupervised, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.loss_fn = MuSELossModule(lambda_orthogonality, M, t, _debug=_debug)

        self.frameaxis_autoencoder = MuSEFrameAxisAutoencoder(
            embedding_dim=embedding_dim,
            frameaxis_dim=frameaxis_dim,
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

        self._debug = _debug

        # Debugging:
        self.logger.debug(f"✅ MuSEFrameAxisUnsupervised successfully initialized")

    def forward(
        self,
        v_fx,
        mask,
        v_sentence,
        fx_negatives,
        tau,
    ):
        outputs_fx = self.frameaxis_autoencoder(
            v_fx,
            mask,
            v_sentence,
            tau,
        )

        outputs_fx["v"] = v_fx

        loss = self.loss_fn(
            outputs_fx,
            fx_negatives,
            mask,
        )

        results = {
            "loss": loss,
            "fx": outputs_fx,
        }

        return results
