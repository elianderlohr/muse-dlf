import torch
import torch.nn as nn

from model.slmuse_dlf.srl_embeddings import SRLEmbeddings
from model.slmuse_dlf.supervised_module import MUSESupervised
from model.slmuse_dlf.unsupervised_module import MUSEUnsupervised
from model.slmuse_dlf.unsupervised_frameaxis_module import MUSEFrameAxisUnsupervised
from utils.logging_manager import LoggerManager


class MUSEDLF(nn.Module):
    def __init__(
        self,
        embedding_dim,
        frameaxis_dim,
        hidden_dim,
        num_classes,
        num_sentences,
        # General Parameters
        dropout_prob=0.3,  # dropout probability
        # SRLEmbeddings Parameters
        bert_model_name="roberta-base",  # Name of the pre-trained model to use from huggingface.co/models
        bert_model_name_or_path="roberta-base",  # Path to the pre-trained model or model identifier from huggingface.co/models
        srl_embeddings_pooling="mean",  # mean or cls
        # LossModule Parameters
        lambda_orthogonality=1e-3,  # lambda for orthogonality loss
        M=8,  # M for orthogonality loss
        t=8,  # t for orthogonality loss
        # MUSEUnsupervised Parameters
        muse_unsupervised_num_layers=2,  # Number of layers in the encoder
        muse_unsupervised_activation="relu",  # Activation function: "relu", "gelu", "leaky_relu", "elu"
        muse_unsupervised_use_batch_norm=True,  # Whether to use batch normalization
        muse_unsupervised_matmul_input="g",  # g or d (g = gumbel-softmax, d = softmax)
        muse_unsupervised_gumbel_softmax_hard=False,  # Whether to use hard gumbel softmax
        muse_unsupervised_gumbel_softmax_log=False,  # Whether to use log gumbel softmax
        # MUSEFrameAxisUnsupervised Parameters
        muse_frameaxis_unsupervised_num_layers=2,  # Number of layers in the encoder
        muse_frameaxis_unsupervised_activation="relu",  # Activation function: "relu", "gelu", "leaky_relu", "elu"
        muse_frameaxis_unsupervised_use_batch_norm=True,  # Whether to use batch normalization
        muse_frameaxis_unsupervised_matmul_input="g",  # g or d (g = gumbel-softmax, d = softmax)
        muse_frameaxis_unsupervised_concat_frameaxis=True,  # Whether to concatenate frameaxis with sentence
        muse_frameaxis_unsupervised_gumbel_softmax_hard=False,  # Whether to use hard gumbel softmax
        muse_frameaxis_unsupervised_gumbel_softmax_log=False,  # Whether to use log gumbel softmax
        # SupervisedModule Parameters
        supervised_concat_frameaxis=True,  # Whether to concatenate frameaxis with sentence
        supervised_num_layers=2,  # Number of layers in the encoder
        supervised_activation="relu",  # Activation function: "relu", "gelu", "leaky_relu", "elu"
        # Debugging
        _debug=False,
    ):
        super(MUSEDLF, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        # Aggregation layer replaced with SRL_Embeddings
        self.aggregation = SRLEmbeddings(
            model_name_or_path=bert_model_name_or_path,
            model_type=bert_model_name,
            pooling=srl_embeddings_pooling,
            _debug=_debug,
        )

        # Unsupervised training module
        self.unsupervised = MUSEUnsupervised(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            lambda_orthogonality=lambda_orthogonality,
            M=M,
            t=t,
            num_layers=muse_unsupervised_num_layers,
            dropout_prob=dropout_prob,
            activation=muse_unsupervised_activation,
            use_batch_norm=muse_unsupervised_use_batch_norm,
            matmul_input=muse_unsupervised_matmul_input,
            gumbel_softmax_hard=muse_unsupervised_gumbel_softmax_hard,
            gumbel_softmax_log=muse_unsupervised_gumbel_softmax_log,
            _debug=_debug,
        )

        self.unsupervised_fx = MUSEFrameAxisUnsupervised(
            embedding_dim=embedding_dim,
            frameaxis_dim=frameaxis_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            lambda_orthogonality=lambda_orthogonality,
            M=M,
            t=t,
            num_layers=muse_frameaxis_unsupervised_num_layers,
            dropout_prob=dropout_prob,
            activation=muse_frameaxis_unsupervised_activation,
            use_batch_norm=muse_frameaxis_unsupervised_use_batch_norm,
            matmul_input=muse_frameaxis_unsupervised_matmul_input,
            concat_frameaxis=muse_frameaxis_unsupervised_concat_frameaxis,
            gumbel_softmax_hard=muse_frameaxis_unsupervised_gumbel_softmax_hard,
            gumbel_softmax_log=muse_frameaxis_unsupervised_gumbel_softmax_log,
            _debug=_debug,
        )

        # Supervised training module
        self.supervised = MUSESupervised(
            embedding_dim,
            num_classes=num_classes,
            frameaxis_dim=frameaxis_dim,
            num_sentences=num_sentences,
            dropout_prob=dropout_prob,
            concat_frameaxis=supervised_concat_frameaxis,
            num_layers=supervised_num_layers,
            activation_function=supervised_activation,
            _debug=_debug,
        )

        self._debug = _debug

        # Debugging:
        self.logger.debug(f"✅ MUSEDLF successfully initialized")

    def negative_sampling(self, embeddings, num_negatives=-1):
        if num_negatives == -1:
            num_negatives = embeddings.size(0)

        batch_size, num_sentences, num_args, embedding_dim = embeddings.size()
        all_negatives = []

        for i in range(batch_size):
            for j in range(num_sentences):
                # Flatten the arguments dimension to sample across all arguments in the sentence
                flattened_embeddings = embeddings[i, j].view(-1, embedding_dim)

                # Get indices of non-padded embeddings (assuming padding is represented by all-zero vectors)
                non_padded_indices = torch.where(
                    torch.any(flattened_embeddings != 0, dim=1)
                )[0]

                # Randomly sample negative indices from non-padded embeddings
                if len(non_padded_indices) > 0:
                    negative_indices = non_padded_indices[
                        torch.randint(0, len(non_padded_indices), (num_negatives,))
                    ]
                else:
                    # If no non-padded embeddings, use zeros
                    negative_indices = torch.zeros(num_negatives, dtype=torch.long)

                negative_samples = flattened_embeddings[negative_indices, :]
                all_negatives.append(negative_samples)

        # Concatenate all negative samples into a single tensor
        all_negatives = torch.cat(all_negatives, dim=0)

        # If more samples than required, randomly select 'num_negatives' samples
        if all_negatives.size(0) > num_negatives:
            indices = torch.randperm(all_negatives.size(0))[:num_negatives]
            all_negatives = all_negatives[indices]

        return all_negatives

    def negative_fx_sampling(self, fxs, num_negatives=8):
        batch_size, num_sentences, frameaxis_dim = fxs.size()
        all_negatives = []

        for i in range(batch_size):
            for j in range(num_sentences):
                # Flatten the arguments dimension to sample across all arguments in the sentence
                flattened_fxs = fxs[i, j].view(-1, frameaxis_dim)

                # Get indices of non-padded embeddings (assuming padding is represented by all-zero vectors)
                non_padded_indices = torch.where(torch.any(flattened_fxs != 0, dim=1))[
                    0
                ]

                # Randomly sample negative indices from non-padded embeddings
                if len(non_padded_indices) > 0:
                    negative_indices = non_padded_indices[
                        torch.randint(0, len(non_padded_indices), (num_negatives,))
                    ]
                else:
                    # If no non-padded embeddings, use zeros
                    negative_indices = torch.zeros(num_negatives, dtype=torch.long)

                negative_samples = flattened_fxs[negative_indices, :]
                all_negatives.append(negative_samples)

        # Concatenate all negative samples into a single tensor
        all_negatives = torch.cat(all_negatives, dim=0)

        # If more samples than required, randomly select 'num_negatives' samples
        if all_negatives.size(0) > num_negatives:
            indices = torch.randperm(all_negatives.size(0))[:num_negatives]
            all_negatives = all_negatives[indices]

        return all_negatives

    def forward(
        self,
        sentence_ids,
        sentence_attention_masks,
        predicate_ids,
        arg0_ids,
        arg1_ids,
        frameaxis_data,
        tau,
    ):
        # Convert input IDs to embeddings
        sentence_embeddings, predicate_embeddings, arg0_embeddings, arg1_embeddings = (
            self.aggregation(
                sentence_ids,
                sentence_attention_masks,
                predicate_ids,
                arg0_ids,
                arg1_ids,
            )
        )

        # Handle multiple spans by averaging predictions
        unsupervised_losses = torch.zeros(
            (sentence_embeddings.size(0),), device=sentence_embeddings.device
        )

        # Creating storage for aggregated d tensors
        d_p_list, d_a0_list, d_a1_list, d_fx_list = [], [], [], []

        negatives_p = self.negative_sampling(predicate_embeddings)
        negatives_a0 = self.negative_sampling(arg0_embeddings)
        negatives_a1 = self.negative_sampling(arg1_embeddings)

        negatives_fx = self.negative_fx_sampling(frameaxis_data)

        # Process each sentence
        for sentence_idx in range(sentence_embeddings.size(1)):
            s_sentence_span = sentence_embeddings[:, sentence_idx, :]
            v_fx = frameaxis_data[:, sentence_idx, :]

            d_p_sentence_list = []
            d_a0_sentence_list = []
            d_a1_sentence_list = []

            # Process each span
            for span_idx in range(predicate_embeddings.size(2)):
                v_p_span = predicate_embeddings[:, sentence_idx, span_idx, :]
                v_a0_span = arg0_embeddings[:, sentence_idx, span_idx, :]
                v_a1_span = arg1_embeddings[:, sentence_idx, span_idx, :]

                # Feed the embeddings to the unsupervised module
                unsupervised_results = self.unsupervised(
                    v_p_span,
                    v_a0_span,
                    v_a1_span,
                    s_sentence_span,
                    negatives_p,
                    negatives_a0,
                    negatives_a1,
                    tau,
                )
                unsupervised_losses += unsupervised_results["loss"]

                if torch.isnan(unsupervised_results["loss"]).any():
                    self.logger.debug("loss is nan")

                # check if [p][d] has nan values
                if torch.isnan(unsupervised_results["p"]["d"]).any():
                    self.logger.debug("p d has nan")
                    self.logger.debug(unsupervised_results["p"]["d"])
                if torch.isnan(unsupervised_results["a0"]["d"]).any():
                    self.logger.debug("a0 d has nan")
                    self.logger.debug(unsupervised_results["a0"]["d"])
                if torch.isnan(unsupervised_results["a1"]["d"]).any():
                    self.logger.debug("a1 d has nan")
                    self.logger.debug(unsupervised_results["a1"]["d"])

                # Use the vhat (reconstructed embeddings) for supervised predictions
                d_p_sentence_list.append(unsupervised_results["p"]["d"])
                d_a0_sentence_list.append(unsupervised_results["a0"]["d"])
                d_a1_sentence_list.append(unsupervised_results["a1"]["d"])

            # Aggregating across all spans
            d_p_sentence = torch.stack(d_p_sentence_list, dim=1)
            d_a0_sentence = torch.stack(d_a0_sentence_list, dim=1)
            d_a1_sentence = torch.stack(d_a1_sentence_list, dim=1)

            d_p_list.append(d_p_sentence)
            d_a0_list.append(d_a0_sentence)
            d_a1_list.append(d_a1_sentence)

            # As per sentence only one frameaxis data set is present calculate only once
            unsupervised_fx_results = self.unsupervised_fx(
                s_sentence_span,
                v_fx,
                negatives_fx,
                tau,
            )

            d_fx_list.append(unsupervised_fx_results["fx"]["d"])

            # add the loss to the unsupervised losses
            unsupervised_losses += unsupervised_fx_results["loss"]

        # Aggregating across all spans
        d_p_aggregated = torch.stack(d_p_list, dim=1)
        d_a0_aggregated = torch.stack(d_a0_list, dim=1)
        d_a1_aggregated = torch.stack(d_a1_list, dim=1)
        d_fx_aggregated = torch.stack(d_fx_list, dim=1)

        # check if d_p_aggregated has nan values
        if torch.isnan(d_p_aggregated).any():
            self.logger.debug("d_p_aggregated has nan")
            self.logger.debug(d_p_aggregated)
        if torch.isnan(d_a0_aggregated).any():
            self.logger.debug("d_a0_aggregated has nan")
            self.logger.debug(d_a0_aggregated)
        if torch.isnan(d_a1_aggregated).any():
            self.logger.debug("d_a1_aggregated has nan")
            self.logger.debug(d_a1_aggregated)
        if torch.isnan(d_fx_aggregated).any():
            self.logger.debug("d_fx_aggregated has nan")
            self.logger.debug(d_fx_aggregated)

        # Supervised predictions
        span_pred, sentence_pred, combined_pred, other = self.supervised(
            d_p_aggregated,
            d_a0_aggregated,
            d_a1_aggregated,
            d_fx_aggregated,
            sentence_embeddings,
            frameaxis_data,
        )

        # Identify valid (non-nan) losses
        valid_losses = ~torch.isnan(unsupervised_losses)

        # Take average by summing the valid losses and dividing by num sentences so that padded sentences are also taken in equation
        unsupervised_loss = unsupervised_losses[valid_losses].sum() / (
            sentence_embeddings.shape[0]
            * sentence_embeddings.shape[1]
            * sentence_embeddings.shape[2]
        )

        return unsupervised_loss, span_pred, sentence_pred, combined_pred, other
