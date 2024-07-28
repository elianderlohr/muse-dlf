import torch
import torch.nn as nn
from utils.logging_manager import LoggerManager
from torch.cuda.amp import autocast


class SLMUSESupervisedAlternative6(nn.Module):
    def __init__(
        self,
        embedding_dim,  # Embedding dimension (e.g. RoBERTa 768)
        num_classes,  # Number of classes to predict
        frameaxis_dim,  # Frameaxis dimension
        num_sentences,  # Number of sentences
        hidden_dim,  # Hidden size of the feed-forward network
        dropout_prob=0.1,  # Dropout probability
        concat_frameaxis=False,  # Whether to concatenate frameaxis with sentence
        activation_function="relu",  # Activation function
        _debug=False,
    ):
        super(SLMUSESupervisedAlternative6, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.embedding_dim = embedding_dim
        self.frameaxis_dim = frameaxis_dim
        self.num_classes = num_classes

        D_h = embedding_dim + (frameaxis_dim if concat_frameaxis else 0)

        self.flatten = nn.Flatten(start_dim=1)
        self.concat_frameaxis = concat_frameaxis
        self._debug = _debug

        # Define feed-forward network for sentence embeddings with gradual dimensionality reduction
        self.feed_forward_sentence = nn.Sequential(
            nn.Linear(D_h * num_sentences, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(4096, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes),
        )

        self.initialize_parameters()

        # Debugging:
        self.logger.debug(f"✅ MUSESupervised successfully initialized")

    def initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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

            flattened = self.flatten(vs)

            y_hat_s = self.feed_forward_sentence(flattened)

            combined = y_hat_u + y_hat_s

            other = {
                "predicate": d_p_mean,
                "arg0": d_a0_mean,
                "arg1": d_a1_mean,
                "frameaxis": d_fx_mean,
            }

        # Debugging:
        if self._debug:
            self.logger.debug(
                f"Forward pass debug info: batch_size={batch_size}, num_sentences={num_sentences}, embedding_dim={embedding_dim}"
            )
            self.logger.debug(
                f"Shapes - d_p_mean: {d_p_mean.shape}, d_a0_mean: {d_a0_mean.shape}, d_a1_mean: {d_a1_mean.shape}, d_fx_mean: {d_fx_mean.shape}"
            )
            self.logger.debug(
                f"Shapes - y_hat_u: {y_hat_u.shape}, flattened: {flattened.shape}, y_hat_s: {y_hat_s.shape}, combined: {combined.shape}"
            )

        return y_hat_u, y_hat_s, combined, other
