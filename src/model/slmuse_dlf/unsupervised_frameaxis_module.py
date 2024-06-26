import torch
import torch.nn as nn
import torch.nn.functional as F
from model.slmuse_dlf.frameaxis_autoencoder import FrameAxisAutoencoder
from model.slmuse_dlf.loss_module import LossModule

from utils.logging_manager import LoggerManager

from torch.cuda.amp import autocast


class MUSEFrameAxisUnsupervised(nn.Module):
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
        concat_frameaxis=True,  # whether to concatenate frameaxis with sentence
        gumbel_softmax_log=False,  # whether to use log gumbel softmax
        _debug=False,
    ):
        super(MUSEFrameAxisUnsupervised, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.loss_fn = LossModule(lambda_orthogonality, M, t, _debug=_debug)

        self.frameaxis_autoencoder = FrameAxisAutoencoder(
            embedding_dim=embedding_dim,
            frameaxis_dim=frameaxis_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            activation=activation,
            use_batch_norm=use_batch_norm,
            matmul_input=matmul_input,
            concat_frameaxis=concat_frameaxis,
            log=gumbel_softmax_log,
            _debug=_debug,
        )

        self._debug = _debug

        # Debugging:
        self.logger.debug(f"✅ MUSEFrameAxisUnsupervised successfully initialized")

    def forward(
        self,
        v_sentence,
        v_fx,
        fx_negatives,
        tau,
        mixed_precision="fp16",  # mixed precision as a parameter
    ):
        precision_dtype = (
            torch.float16
            if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16" else None
        )

        with autocast(
            enabled=mixed_precision in ["fp16", "bf16"], dtype=precision_dtype
        ):
            outputs_fx = self.frameaxis_autoencoder(
                v_fx, v_sentence, tau, mixed_precision
            )

            outputs_fx["v"] = v_fx

            loss = self.loss_fn(
                outputs_fx,
                fx_negatives,
                mixed_precision=mixed_precision,
            )

            results = {
                "loss": loss,
                "fx": outputs_fx,
            }

        return results
