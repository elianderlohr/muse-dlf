import torch
import torch.nn as nn
from utils.logging_manager import LoggerManager
from torch.cuda.amp import autocast


class MUSESupervisedAlternative5(nn.Module):
    def __init__(
        self,
        embedding_dim,  # Embedding dimension (e.g. RoBERTa 768)
        num_classes,  # Number of classes to predict
        frameaxis_dim,  # Frameaxis dimension
        hidden_dim,  # Hidden size of the feed-forward network
        dropout_prob=0.1,  # Dropout probability
        concat_frameaxis=False,  # Whether to concatenate frameaxis with sentence
        activation_functions=(
            "gelu",
            "relu",
        ),  # Tuple of activation functions: first and last
        _debug=False,
    ):
        super(MUSESupervisedAlternative5, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.embedding_dim = embedding_dim
        self.frameaxis_dim = frameaxis_dim
        self.num_classes = num_classes

        D_h = embedding_dim + (frameaxis_dim if concat_frameaxis else 0)

        # Layer 1
        self.dropout_1 = nn.Dropout(dropout_prob)
        self.Wr = nn.Linear(D_h, hidden_dim)
        self.activation_first = self.get_activation(activation_functions[0])
        self.batch_norm_1 = nn.BatchNorm1d(D_h)  # Correct the number of features

        # Layer 2
        self.dropout_2 = nn.Dropout(dropout_prob)
        self.Wt = nn.Linear(hidden_dim, hidden_dim)
        self.activation_last = self.get_activation(activation_functions[1])
        self.batch_norm_2 = nn.BatchNorm1d(hidden_dim)

        # Layer 3
        self.dropout_3 = nn.Dropout(dropout_prob)
        self.Wo = nn.Linear(hidden_dim, num_classes)

        self.concat_frameaxis = concat_frameaxis

        self._debug = _debug

        # Debugging:
        self.logger.debug(f"✅ MUSESupervised successfully initialized")

    def get_activation(self, activation_function):
        if activation_function == "relu":
            return nn.ReLU()
        elif activation_function == "gelu":
            return nn.GELU()
        elif activation_function == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_function == "elu":
            return nn.ELU()
        else:
            raise ValueError(
                f"Unsupported activation function. Use 'relu', 'gelu', 'leaky_relu' or 'elu'. Found: {activation_function}."
            )

    def forward(
        self,
        d_p,
        d_a0,
        d_a1,
        d_fx,
        vs,
        frameaxis_data,
        mixed_precision="fp16",  # mixed precision as a parameter
    ):
        precision_dtype = (
            torch.float16
            if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
        )

        with autocast(
            enabled=mixed_precision in ["fp16", "bf16", "fp32"], dtype=precision_dtype
        ):
            batch_size, num_sentences, num_args, embedding_dim = d_p.shape

            d_p_flatten = d_p.view(batch_size, num_sentences * num_args, embedding_dim)
            d_a0_flatten = d_a0.view(
                batch_size, num_sentences * num_args, embedding_dim
            )
            d_a1_flatten = d_a1.view(
                batch_size, num_sentences * num_args, embedding_dim
            )

            # Create masks for non-padded elements
            mask_p = (d_p_flatten.abs().sum(dim=-1) != 0).float()
            mask_a0 = (d_a0_flatten.abs().sum(dim=-1) != 0).float()
            mask_a1 = (d_a1_flatten.abs().sum(dim=-1) != 0).float()
            mask_fx = (d_fx.abs().sum(dim=-1) != 0).float()

            # Calculate the mean ignoring padded elements
            d_p_mean = (d_p_flatten * mask_p.unsqueeze(-1)).sum(dim=1) / torch.clamp(
                mask_p.sum(dim=1, keepdim=True), min=1
            )
            d_a0_mean = (d_a0_flatten * mask_a0.unsqueeze(-1)).sum(dim=1) / torch.clamp(
                mask_a0.sum(dim=1, keepdim=True), min=1
            )
            d_a1_mean = (d_a1_flatten * mask_a1.unsqueeze(-1)).sum(dim=1) / torch.clamp(
                mask_a1.sum(dim=1, keepdim=True), min=1
            )
            d_fx_mean = (d_fx * mask_fx.unsqueeze(-1)).sum(dim=1) / torch.clamp(
                mask_fx.sum(dim=1, keepdim=True), min=1
            )

            # Combine and normalize the final descriptor
            y_hat_u = (d_p_mean + d_a0_mean + d_a1_mean + d_fx_mean) / 4

            if self.concat_frameaxis:
                vs = torch.cat([vs, frameaxis_data], dim=-1)

            # Layer 1
            vs = self.dropout_1(vs)
            vs = self.Wr(vs)
            vs = self.activation_first(vs)

            # Apply batch normalization along the embedding dimension
            vs = vs.view(-1, vs.size(-1))  # Flatten batch and sentence dimensions
            vs = self.batch_norm_1(vs)
            vs = vs.view(
                batch_size, num_sentences, -1
            )  # Restore the original dimensions

            # Average predictions across sentences but ignore padded elements (all zeros)
            mask_vs = (vs.abs().sum(dim=-1) != 0).float()
            vs = (vs * mask_vs.unsqueeze(-1)).sum(dim=1) / torch.clamp(
                mask_vs.sum(dim=1, keepdim=True), min=1
            )

            # Layer 2
            vs = self.dropout_2(vs)
            vs = self.Wt(vs)
            vs = self.activation_last(vs)

            # Apply second batch normalization
            vs = self.batch_norm_2(vs)

            # Layer 3
            vs = self.dropout_3(vs)
            vs = self.Wo(vs)

            y_hat_s = vs

            other = {
                "predicate": d_p_mean,
                "arg0": d_a0_mean,
                "arg1": d_a1_mean,
                "frameaxis": d_fx_mean,
            }

        return y_hat_u, y_hat_s, y_hat_u + y_hat_s, other
