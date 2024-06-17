import torch
import torch.nn as nn

from utils.logging_manager import LoggerManager


class MUSESupervised(nn.Module):
    def __init__(
        self,
        embedding_dim,  # Embedding dimension (e.g. RoBERTa 768)
        num_classes,  # Number of classes to predict
        frameaxis_dim,  # Frameaxis dimension
        num_sentences,  # Number of sentences
        dropout_prob=0.3,  # Dropout probability
        concat_frameaxis=True,  # Whether to concatenate frameaxis with sentence
        num_layers=3,  # Number of layers in feed-forward network
        activation_function="relu",  # Activation function: "relu", "gelu", "leaky_relu", "elu"
        _debug=False,
    ):
        super(MUSESupervised, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.embedding_dim = embedding_dim
        self.frameaxis_dim = frameaxis_dim

        D_h = embedding_dim + (frameaxis_dim if concat_frameaxis else 0)

        # Define the activation function
        if activation_function == "relu":
            self.activation = nn.ReLU()
        elif activation_function == "gelu":
            self.activation = nn.GELU()
        elif activation_function == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_function == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(
                f"Unsupported activation function. Use 'relu', 'gelu', 'leaky_relu' or 'elu'. Found: {activation_function}."
            )

        # Feed-forward networks for sentence embeddings
        layers = []
        input_dim = D_h * num_sentences
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(input_dim, embedding_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(embedding_dim, num_classes))

        self.feed_forward_sentence = nn.Sequential(*layers)

        # Flatten
        self.flatten = nn.Flatten(start_dim=1)

        self.concat_frameaxis = concat_frameaxis

        self._debug = _debug

        # Debugging:
        self.logger.debug(f"✅ MUSESupervised successfully initialized")

    def forward(
        self,
        d_p,
        d_a0,
        d_a1,
        d_fx,
        vs,
        frameaxis_data,
    ):
        batch_size, num_sentences, num_args, embedding_dim = d_p.shape

        d_p_flatten = d_p.view(batch_size, num_sentences * num_args, embedding_dim)
        d_a0_flatten = d_a0.view(batch_size, num_sentences * num_args, embedding_dim)
        d_a1_flatten = d_a1.view(batch_size, num_sentences * num_args, embedding_dim)

        d_p_mean = d_p_flatten.mean(dim=1)
        d_a0_mean = d_a0_flatten.mean(dim=1)
        d_a1_mean = d_a1_flatten.mean(dim=1)

        d_fx_mean = d_fx.mean(dim=1)

        # Debugging:
        self.logger.debug(f"d_p_mean: {d_p_mean.shape}")
        self.logger.debug(f"d_a0_mean: {d_a0_mean.shape}")
        self.logger.debug(f"d_a1_mean: {d_a1_mean.shape}")
        self.logger.debug(f"d_fx_mean: {d_fx_mean.shape}")

        # Combine and normalize the final descriptor
        y_hat_u = (d_p_mean + d_a0_mean + d_a1_mean + d_fx_mean) / 4

        if self.concat_frameaxis:
            vs = torch.cat([vs, frameaxis_data], dim=-1)

        # reshape vs from [batch_size, num_sentences, embedding_dim] to [batch_size * num_sentences, embedding_dim]
        ws_flattened = self.flatten(vs)

        y_hat_s = self.feed_forward_sentence(ws_flattened)

        # Sum the two predictions
        combined = y_hat_u + y_hat_s

        other = {
            "predicate": d_p_mean,
            "arg0": d_a0_mean,
            "arg1": d_a1_mean,
            "frameaxis": d_fx_mean,
        }

        return y_hat_u, y_hat_s, combined, other