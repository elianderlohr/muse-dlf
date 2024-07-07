import torch
import torch.nn as nn

from model.muse_dlf.embeddings import MUSEEmbeddings
from model.muse_dlf.supervised_module import MUSESupervised
from model.muse_dlf.unsupervised_module import MUSEUnsupervised
from model.muse_dlf.unsupervised_frameaxis_module import MUSEFrameAxisUnsupervised
from utils.logging_manager import LoggerManager

from torch.cuda.amp import autocast


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
        M=8,  # M total margin budget for triplet loss
        t=8,  # t number of negatives for triplet loss (select t descriptors in Fz with smallest weights in gz)
        # MUSEUnsupervised Parameters
        muse_unsupervised_num_layers=2,  # Number of layers in the encoder
        muse_unsupervised_activation="relu",  # Activation function: "relu", "gelu", "leaky_relu", "elu"
        muse_unsupervised_use_batch_norm=True,  # Whether to use batch normalization
        muse_unsupervised_matmul_input="g",  # g or d (g = gumbel-softmax, d = softmax)
        muse_unsupervised_gumbel_softmax_log=False,  # Whether to use log gumbel softmax
        # MUSEFrameAxisUnsupervised Parameters
        muse_frameaxis_unsupervised_num_layers=2,  # Number of layers in the encoder
        muse_frameaxis_unsupervised_activation="relu",  # Activation function: "relu", "gelu", "leaky_relu", "elu"
        muse_frameaxis_unsupervised_use_batch_norm=True,  # Whether to use batch normalization
        muse_frameaxis_unsupervised_matmul_input="g",  # g or d (g = gumbel-softmax, d = softmax)
        muse_frameaxis_unsupervised_gumbel_softmax_log=False,  # Whether to use log gumbel softmax
        # MUSEUnsupervised & MUSEFrameAxisUnsupervised Parameters
        num_negatives=-1,  # Number of negative samples to use for triplet loss
        # SupervisedModule Parameters
        supervised_concat_frameaxis=True,  # Whether to concatenate frameaxis with sentence
        supervised_num_layers=2,  # Number of layers in the encoder
        supervised_activation="relu",  # Activation function: "relu", "gelu", "leaky_relu", "elu"
        # Debugging
        _debug=False,
        _detect_anomaly=False,
    ):
        super(MUSEDLF, self).__init__()

        self.model_type = "muse-dlf"

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        if _debug:
            self.logger.warning("🚨 Debug mode is enabled")
            if _detect_anomaly:
                torch.autograd.set_detect_anomaly(_detect_anomaly)
                self.logger.warning(
                    f"🚨 torch.autograd.set_detect_anomaly({_detect_anomaly}) activated"
                )

        # Aggregation layer replaced with SRL_Embeddings
        self.aggregation = MUSEEmbeddings(
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

        self.num_negatives = num_negatives
        self.num_classes = num_classes

        self._debug = _debug

        # Store configuration
        self._config = {
            "embedding_dim": embedding_dim,
            "frameaxis_dim": frameaxis_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "num_sentences": num_sentences,
            "dropout_prob": dropout_prob,
            "bert_model_name": bert_model_name,
            "bert_model_name_or_path": bert_model_name_or_path,
            "srl_embeddings_pooling": srl_embeddings_pooling,
            "lambda_orthogonality": lambda_orthogonality,
            "M": M,
            "t": t,
            "muse_unsupervised_num_layers": muse_unsupervised_num_layers,
            "muse_unsupervised_activation": muse_unsupervised_activation,
            "muse_unsupervised_use_batch_norm": muse_unsupervised_use_batch_norm,
            "muse_unsupervised_matmul_input": muse_unsupervised_matmul_input,
            "muse_unsupervised_gumbel_softmax_log": muse_unsupervised_gumbel_softmax_log,
            "muse_frameaxis_unsupervised_num_layers": muse_frameaxis_unsupervised_num_layers,
            "muse_frameaxis_unsupervised_activation": muse_frameaxis_unsupervised_activation,
            "muse_frameaxis_unsupervised_use_batch_norm": muse_frameaxis_unsupervised_use_batch_norm,
            "muse_frameaxis_unsupervised_matmul_input": muse_frameaxis_unsupervised_matmul_input,
            "muse_frameaxis_unsupervised_gumbel_softmax_log": muse_frameaxis_unsupervised_gumbel_softmax_log,
            "num_negatives": num_negatives,
            "supervised_concat_frameaxis": supervised_concat_frameaxis,
            "supervised_num_layers": supervised_num_layers,
            "supervised_activation": supervised_activation,
            "_debug": _debug,
            "_detect_anomaly": _detect_anomaly,
        }

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
                    num_samples = min(num_negatives, len(non_padded_indices))
                    negative_indices = non_padded_indices[
                        torch.randperm(len(non_padded_indices))[:num_samples]
                    ]
                    negative_samples = flattened_embeddings[negative_indices, :]
                    all_negatives.append(negative_samples)

        if len(all_negatives) == 0:
            return torch.zeros((num_negatives, embedding_dim), device=embeddings.device)

        # Concatenate all negative samples into a single tensor
        all_negatives = torch.cat(all_negatives, dim=0)

        # Ensure we have the right number of negatives
        if all_negatives.size(0) > num_negatives:
            indices = torch.randperm(all_negatives.size(0))[:num_negatives]
            all_negatives = all_negatives[indices]

        return all_negatives

    def negative_fx_sampling(self, fxs, num_negatives=-1):
        if num_negatives == -1:
            num_negatives = fxs.size(0)

        batch_size, num_sentences, frameaxis_dim = fxs.size()
        all_negatives = []

        for i in range(batch_size):
            for j in range(num_sentences):
                # Flatten the frameaxis dimension to sample across all elements in the sentence
                flattened_fxs = fxs[i, j].view(-1, frameaxis_dim)

                # Get indices of non-padded embeddings (assuming padding is represented by all-zero vectors)
                non_padded_indices = torch.where(torch.any(flattened_fxs != 0, dim=1))[
                    0
                ]

                # Randomly sample negative indices from non-padded embeddings
                if len(non_padded_indices) > 0:
                    num_samples = min(num_negatives, len(non_padded_indices))
                    negative_indices = non_padded_indices[
                        torch.randperm(len(non_padded_indices))[:num_samples]
                    ]
                    negative_samples = flattened_fxs[negative_indices, :]
                    all_negatives.append(negative_samples)

        if len(all_negatives) == 0:
            return torch.zeros((num_negatives, frameaxis_dim), device=fxs.device)

        # Concatenate all negative samples into a single tensor
        all_negatives = torch.cat(all_negatives, dim=0)

        # Ensure we have the right number of negatives
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
        mixed_precision="fp16",  # mixed precision as a parameter
    ):
        self.logger.debug(
            f"🚦 Mixed precision enabled: {mixed_precision in ['fp16', 'bf16', 'fp32']}"
        )

        precision_dtype = (
            torch.float16
            if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
        )

        with autocast(
            enabled=mixed_precision in ["fp16", "bf16", "fp32"], dtype=precision_dtype
        ):
            # Convert input IDs to embeddings
            (
                sentence_embeddings,
                predicate_embeddings,
                arg0_embeddings,
                arg1_embeddings,
            ) = self.aggregation(
                sentence_ids,
                sentence_attention_masks,
                predicate_ids,
                arg0_ids,
                arg1_ids,
                mixed_precision=mixed_precision,
            )

            # Delete input tensors after conversion to embeddings
            del (
                sentence_ids,
                sentence_attention_masks,
                predicate_ids,
                arg0_ids,
                arg1_ids,
            )
            torch.cuda.empty_cache()

            # Creating storage for aggregated d tensors
            d_p_list, d_a0_list, d_a1_list, d_fx_list = [], [], [], []

            negatives_p = self.negative_sampling(
                predicate_embeddings, num_negatives=self.num_negatives
            )
            negatives_a0 = self.negative_sampling(
                arg0_embeddings, num_negatives=self.num_negatives
            )
            negatives_a1 = self.negative_sampling(
                arg1_embeddings, num_negatives=self.num_negatives
            )

            negatives_fx = self.negative_fx_sampling(
                frameaxis_data, num_negatives=self.num_negatives
            )

            # Initialize unsupervised_losses tensor and count tensor
            unsupervised_losses = torch.zeros(
                (sentence_embeddings.size(0),), device=sentence_embeddings.device
            )
            valid_counts = torch.zeros(
                (sentence_embeddings.size(0),), device=sentence_embeddings.device
            )

            # Process each sentence
            for sentence_idx in range(sentence_embeddings.size(1)):

                self.logger.debug(f"###################################")
                self.logger.debug(f"Processing sentence: {sentence_idx}")

                s_sentence_span = sentence_embeddings[:, sentence_idx, :]
                v_fx = frameaxis_data[:, sentence_idx, :]

                d_p_sentence_list = []
                d_a0_sentence_list = []
                d_a1_sentence_list = []

                sentence_loss = torch.zeros(
                    (sentence_embeddings.size(0),), device=sentence_embeddings.device
                )

                # run if sentence embedding is not all zeros
                if not torch.all(s_sentence_span == 0):

                    # Process each span
                    for span_idx in range(predicate_embeddings.size(2)):
                        v_p_span = predicate_embeddings[:, sentence_idx, span_idx, :]
                        v_a0_span = arg0_embeddings[:, sentence_idx, span_idx, :]
                        v_a1_span = arg1_embeddings[:, sentence_idx, span_idx, :]

                        # Mask to ignore padded sentences for each span individually
                        mask_p = (v_p_span.abs().sum(dim=-1) != 0).float().bool()
                        mask_a0 = (v_a0_span.abs().sum(dim=-1) != 0).float().bool()
                        mask_a1 = (v_a1_span.abs().sum(dim=-1) != 0).float().bool()

                        if not mask_p.any():
                            self.logger.debug(
                                f"Idx: [{sentence_idx}, {span_idx}] Found {(mask_p.size(0) - mask_p.sum()).item()} of {mask_p.size(0)} zeros in mask_p"
                            )
                        if not mask_a0.any():
                            self.logger.debug(
                                f"Idx: [{sentence_idx}, {span_idx}] Found {(mask_a0.size(0) - mask_a0.sum()).item()} of {mask_a0.size(0)} zeros in mask_a0"
                            )
                        if not mask_a1.any():
                            self.logger.debug(
                                f"Idx: [{sentence_idx}, {span_idx}] Found {(mask_a1.size(0) - mask_a1.sum()).item()} of {mask_a1.size(0)} zeros in mask_a1"
                            )

                        # Skip the unsupervised module call if all masks are zero
                        if mask_p.any() and mask_a0.any() and mask_a1.any():
                            # Feed the embeddings to the unsupervised module
                            unsupervised_results = self.unsupervised(
                                v_p_span,
                                v_a0_span,
                                v_a1_span,
                                mask_p.float(),
                                mask_a0.float(),
                                mask_a1.float(),
                                s_sentence_span,
                                negatives_p,
                                negatives_a0,
                                negatives_a1,
                                tau,
                                mixed_precision=mixed_precision,
                            )

                            sentence_loss += (
                                unsupervised_results["loss"]
                                * (mask_p & mask_a0 & mask_a1).float()
                            )
                            valid_counts += (mask_p & mask_a0 & mask_a1).float()

                            # Use the vhat (reconstructed embeddings) for supervised predictions
                            d_p_sentence_list.append(unsupervised_results["p"]["d"])
                            d_a0_sentence_list.append(unsupervised_results["a0"]["d"])
                            d_a1_sentence_list.append(unsupervised_results["a1"]["d"])

                            # Delete unsupervised_results to free memory
                            del unsupervised_results
                            torch.cuda.empty_cache()
                        else:
                            d_p_sentence_list.append(
                                torch.zeros(
                                    (
                                        predicate_embeddings.size(0),
                                        self.num_classes,
                                    ),
                                    device=predicate_embeddings.device,
                                )
                            )
                            d_a0_sentence_list.append(
                                torch.zeros(
                                    (
                                        predicate_embeddings.size(0),
                                        self.num_classes,
                                    ),
                                    device=predicate_embeddings.device,
                                )
                            )
                            d_a1_sentence_list.append(
                                torch.zeros(
                                    (
                                        predicate_embeddings.size(0),
                                        self.num_classes,
                                    ),
                                    device=predicate_embeddings.device,
                                )
                            )

                        # Delete span-related tensors after use
                        del v_p_span, v_a0_span, v_a1_span, mask_p, mask_a0, mask_a1
                        torch.cuda.empty_cache()

                    mask_fx = (v_fx.abs().sum(dim=-1) != 0).float()

                    # As per sentence only one frameaxis data set is present calculate only once
                    unsupervised_fx_results = self.unsupervised_fx(
                        v_fx,
                        mask_fx,
                        s_sentence_span,
                        negatives_fx,
                        tau,
                        mixed_precision=mixed_precision,
                    )

                    d_fx_list.append(unsupervised_fx_results["fx"]["d"])

                    # Add the loss to the unsupervised losses
                    sentence_loss += unsupervised_fx_results["loss"] * (mask_fx).float()
                    valid_counts += mask_fx

                    # Delete unsupervised_fx_results to free memory
                    del unsupervised_fx_results, mask_fx
                    torch.cuda.empty_cache()

                    # Apply mask to sentence loss
                    unsupervised_losses += sentence_loss
                else:
                    self.logger.debug(
                        f"Idx: [{sentence_idx}] Found all zeros in sentence embeddings"
                    )

                    for span_idx in range(predicate_embeddings.size(2)):

                        d_p_sentence_list.append(
                            torch.zeros(
                                (
                                    predicate_embeddings.size(0),
                                    self.num_classes,
                                ),
                                device=predicate_embeddings.device,
                            )
                        )
                        d_a0_sentence_list.append(
                            torch.zeros(
                                (
                                    predicate_embeddings.size(0),
                                    self.num_classes,
                                ),
                                device=predicate_embeddings.device,
                            )
                        )
                        d_a1_sentence_list.append(
                            torch.zeros(
                                (
                                    predicate_embeddings.size(0),
                                    self.num_classes,
                                ),
                                device=predicate_embeddings.device,
                            )
                        )

                    d_fx_list.append(
                        torch.zeros(
                            (
                                predicate_embeddings.size(0),
                                self.num_classes,
                            ),
                            device=predicate_embeddings.device,
                        )
                    )

                del s_sentence_span, v_fx

                # Aggregating across all spans
                if len(d_p_sentence_list) > 0:
                    max_dim = max(d.shape[-1] for d in d_p_sentence_list)
                    d_p_sentence_list = [
                        torch.nn.functional.pad(d, (0, max_dim - d.shape[-1]))
                        for d in d_p_sentence_list
                    ]
                if len(d_a0_sentence_list) > 0:
                    max_dim = max(d.shape[-1] for d in d_a0_sentence_list)
                    d_a0_sentence_list = [
                        torch.nn.functional.pad(d, (0, max_dim - d.shape[-1]))
                        for d in d_a0_sentence_list
                    ]
                if len(d_a1_sentence_list) > 0:
                    max_dim = max(d.shape[-1] for d in d_a1_sentence_list)
                    d_a1_sentence_list = [
                        torch.nn.functional.pad(d, (0, max_dim - d.shape[-1]))
                        for d in d_a1_sentence_list
                    ]

                d_p_sentence = torch.stack(d_p_sentence_list, dim=1)
                d_a0_sentence = torch.stack(d_a0_sentence_list, dim=1)
                d_a1_sentence = torch.stack(d_a1_sentence_list, dim=1)

                self.logger.debug(f"Shape of d_p_sentence: {d_p_sentence.shape}")
                self.logger.debug(f"Shape of d_a0_sentence: {d_a0_sentence.shape}")
                self.logger.debug(f"Shape of d_a1_sentence: {d_a1_sentence.shape}")

                d_p_list.append(d_p_sentence)
                d_a0_list.append(d_a0_sentence)
                d_a1_list.append(d_a1_sentence)

                # Delete sentence-related tensors after use
                del d_p_sentence_list, d_a0_sentence_list, d_a1_sentence_list
                torch.cuda.empty_cache()

            # Aggregating across all sentences
            if len(d_p_list) > 0:
                max_dim = max(d.shape[-1] for d in d_p_list)
                d_p_list = [
                    torch.nn.functional.pad(d, (0, max_dim - d.shape[-1]))
                    for d in d_p_list
                ]
            if len(d_a0_list) > 0:
                max_dim = max(d.shape[-1] for d in d_a0_list)
                d_a0_list = [
                    torch.nn.functional.pad(d, (0, max_dim - d.shape[-1]))
                    for d in d_a0_list
                ]
            if len(d_a1_list) > 0:
                max_dim = max(d.shape[-1] for d in d_a1_list)
                d_a1_list = [
                    torch.nn.functional.pad(d, (0, max_dim - d.shape[-1]))
                    for d in d_a1_list
                ]
            if len(d_fx_list) > 0:
                max_dim = max(d.shape[-1] for d in d_fx_list)
                d_fx_list = [
                    torch.nn.functional.pad(d, (0, max_dim - d.shape[-1]))
                    for d in d_fx_list
                ]

            d_p_aggregated = (
                torch.stack(d_p_list, dim=1)
                if d_p_list
                else torch.tensor([], device=sentence_embeddings.device)
            )
            d_a0_aggregated = (
                torch.stack(d_a0_list, dim=1)
                if d_a0_list
                else torch.tensor([], device=sentence_embeddings.device)
            )
            d_a1_aggregated = (
                torch.stack(d_a1_list, dim=1)
                if d_a1_list
                else torch.tensor([], device=sentence_embeddings.device)
            )
            d_fx_aggregated = (
                torch.stack(d_fx_list, dim=1)
                if d_fx_list
                else torch.tensor([], device=sentence_embeddings.device)
            )

            self.logger.debug(f"Shape of d_p_aggregated: {d_p_aggregated.shape}")
            self.logger.debug(f"Shape of d_a0_aggregated: {d_a0_aggregated.shape}")
            self.logger.debug(f"Shape of d_a1_aggregated: {d_a1_aggregated.shape}")
            self.logger.debug(f"Shape of d_fx_aggregated: {d_fx_aggregated.shape}")

            # Supervised predictions
            span_pred, sentence_pred, combined_pred, other = self.supervised(
                d_p_aggregated,
                d_a0_aggregated,
                d_a1_aggregated,
                d_fx_aggregated,
                sentence_embeddings,
                frameaxis_data,
                mixed_precision=mixed_precision,
            )

            # Normalize the unsupervised losses by valid counts for each batch
            unsupervised_loss = (unsupervised_losses / valid_counts).mean()

            # Delete aggregated tensors after use
            del d_p_aggregated, d_a0_aggregated, d_a1_aggregated, d_fx_aggregated
            torch.cuda.empty_cache()

            return unsupervised_loss, span_pred, sentence_pred, combined_pred, other
