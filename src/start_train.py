import argparse
from datetime import timedelta
import json
import math
import os
import random

import numpy as np

from model.slmuse_dlf.muse import SLMuSEDLF
from model.muse_dlf.muse import MuSEDLF
from preprocessing.pre_processor import PreProcessor
import torch
from transformers import BertTokenizer, RobertaTokenizerFast
from torch.optim import Adam, AdamW
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate import DistributedDataParallelKwargs
import warnings
import wandb

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from accelerate.utils import set_seed

from training.trainer import Trainer
from utils.logging_manager import LoggerManager

from utils.focal_loss import FocalLoss
from utils.multi_label_focal_loss import MultiLabelFocalLoss

# Suppress specific warnings from numpy
warnings.filterwarnings(
    "ignore", message="Mean of empty slice.", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

wandb.require("core")


def load_model(
    model_type,
    embedding_dim,
    frameaxis_dim,
    hidden_dim,
    num_classes,
    num_sentences,
    dropout_prob,
    bert_model_name,
    bert_model_name_or_path,
    sentence_pooling,
    hidden_state,
    lambda_orthogonality,
    M,
    t,
    muse_unsupervised_num_layers,
    muse_unsupervised_activation,
    muse_unsupervised_use_batch_norm,
    muse_unsupervised_matmul_input,
    muse_unsupervised_gumbel_softmax_log,
    muse_frameaxis_unsupervised_num_layers,
    muse_frameaxis_unsupervised_activation,
    muse_frameaxis_unsupervised_use_batch_norm,
    muse_frameaxis_unsupervised_matmul_input,
    muse_frameaxis_unsupervised_gumbel_softmax_log,
    num_negatives,
    supervised_concat_frameaxis,
    supervised_num_layers,
    supervised_activation,
    alternative_supervised,
    path_pretrained_model="",
    device="cuda",
    logger=LoggerManager.get_logger(__name__),
    _debug=False,
    _detect_anomaly=False,
):
    try:
        logger.info("Try to load model of type: %s", model_type)

        if model_type == "slmuse-dlf":
            logger.info("Loading SLMuSE-DLF model")
            model = SLMuSEDLF(
                embedding_dim=embedding_dim,
                frameaxis_dim=frameaxis_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_sentences=num_sentences,
                dropout_prob=dropout_prob,
                bert_model_name=bert_model_name,
                bert_model_name_or_path=bert_model_name_or_path,
                sentence_pooling=sentence_pooling,
                hidden_state=hidden_state,
                lambda_orthogonality=lambda_orthogonality,
                M=M,
                t=t,
                muse_unsupervised_num_layers=muse_unsupervised_num_layers,
                muse_unsupervised_activation=muse_unsupervised_activation,
                muse_unsupervised_use_batch_norm=muse_unsupervised_use_batch_norm,
                muse_unsupervised_matmul_input=muse_unsupervised_matmul_input,
                muse_unsupervised_gumbel_softmax_log=muse_unsupervised_gumbel_softmax_log,
                muse_frameaxis_unsupervised_num_layers=muse_frameaxis_unsupervised_num_layers,
                muse_frameaxis_unsupervised_activation=muse_frameaxis_unsupervised_activation,
                muse_frameaxis_unsupervised_use_batch_norm=muse_frameaxis_unsupervised_use_batch_norm,
                muse_frameaxis_unsupervised_matmul_input=muse_frameaxis_unsupervised_matmul_input,
                muse_frameaxis_unsupervised_gumbel_softmax_log=muse_frameaxis_unsupervised_gumbel_softmax_log,
                num_negatives=num_negatives,
                supervised_concat_frameaxis=supervised_concat_frameaxis,
                supervised_num_layers=supervised_num_layers,
                supervised_activation=supervised_activation,
                alternative_supervised=alternative_supervised,
                _debug=_debug,
                _detect_anomaly=_detect_anomaly,
            )
        else:
            logger.info("Loading MuSE-DLF model")
            try:
                model = MuSEDLF(
                    embedding_dim=embedding_dim,
                    frameaxis_dim=frameaxis_dim,
                    hidden_dim=hidden_dim,
                    num_classes=num_classes,
                    num_sentences=num_sentences,
                    dropout_prob=dropout_prob,
                    bert_model_name=bert_model_name,
                    bert_model_name_or_path=bert_model_name_or_path,
                    sentence_pooling=sentence_pooling,
                    hidden_state=hidden_state,
                    lambda_orthogonality=lambda_orthogonality,
                    M=M,
                    t=t,
                    muse_unsupervised_num_layers=muse_unsupervised_num_layers,
                    muse_unsupervised_activation=muse_unsupervised_activation,
                    muse_unsupervised_use_batch_norm=muse_unsupervised_use_batch_norm,
                    muse_unsupervised_matmul_input=muse_unsupervised_matmul_input,
                    muse_unsupervised_gumbel_softmax_log=muse_unsupervised_gumbel_softmax_log,
                    muse_frameaxis_unsupervised_num_layers=muse_frameaxis_unsupervised_num_layers,
                    muse_frameaxis_unsupervised_activation=muse_frameaxis_unsupervised_activation,
                    muse_frameaxis_unsupervised_use_batch_norm=muse_frameaxis_unsupervised_use_batch_norm,
                    muse_frameaxis_unsupervised_matmul_input=muse_frameaxis_unsupervised_matmul_input,
                    muse_frameaxis_unsupervised_gumbel_softmax_log=muse_frameaxis_unsupervised_gumbel_softmax_log,
                    num_negatives=num_negatives,
                    supervised_concat_frameaxis=supervised_concat_frameaxis,
                    supervised_num_layers=supervised_num_layers,
                    supervised_activation=supervised_activation,
                    alternative_supervised=alternative_supervised,
                    _debug=_debug,
                    _detect_anomaly=_detect_anomaly,
                )
            except Exception as e:
                logger.error(f"Error loading MuSE-DLF model: {e}")
                raise

        logger.info("Model loaded successfully, move to device: %s", device)
        model = model.to(device)

        if path_pretrained_model:
            logger.info("Loading model from pretrained path: %s", path_pretrained_model)
            model.load_state_dict(
                torch.load(path_pretrained_model, map_location=device)
            )

        logger.info("Return model")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_logging(debug):
    try:
        if debug:
            LoggerManager.use_accelerate(accelerate_used=True, log_level="DEBUG")
        else:
            LoggerManager.use_accelerate(accelerate_used=True, log_level="INFO")
        logger = LoggerManager.get_logger(__name__)
        return logger
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise


def initialize_wandb(
    wandb_api_key,
    project_name,
    tags,
    config,
    mixed_precision,
    accumulation_steps,
    run_name,
):
    try:
        wandb.login(key=wandb_api_key, timeout=120)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=accumulation_steps,
            kwargs_handlers=[kwargs],  # ddp_kwargs],
        )
        accelerator.init_trackers(
            project_name,
            config,
            init_kwargs={
                "wandb": {
                    "tags": tags.split(","),
                    "name": run_name,
                    "job_type": "train",
                }
            },
        )
        return accelerator
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        raise


def set_custom_seed(seed):
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        set_seed(seed)
    except Exception as e:
        print(f"Error setting seed: {e}")
        raise


def main():
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    parser = argparse.ArgumentParser(description="Train MuSE model")

    # Required arguments
    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "--path_data", type=str, required=True, help="Path to the data file"
    )
    required_args.add_argument(
        "--wandb_api_key", type=str, required=True, help="Wandb API key"
    )
    required_args.add_argument(
        "--project_name", type=str, required=True, help="Wandb project name"
    )
    required_args.add_argument(
        "--model_type",
        type=str,
        default="muse-dlf",
        help="Type of model to train (muse-dlf, slmuse-dlf)",
        choices=["muse-dlf", "slmuse-dlf"],
    )
    # run_name
    required_args.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run",
    )

    parser.add_argument(
        "--tags",
        type=str,
        default="MuSE, Frame Classification, FrameAxis",
        help="Tags to describe the model",
    )

    # Model Configuration
    model_config = parser.add_argument_group("Model Configuration")
    model_config.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="Dimension of the word embeddings",
    )
    model_config.add_argument(
        "--hidden_dim",
        type=int,
        default=768,
        help="Dimension of the hidden layer in the model",
    )
    model_config.add_argument(
        "--num_classes", type=int, default=15, help="Number of classes"
    )
    model_config.add_argument(
        "--frameaxis_dim", type=int, default=10, help="Dimension of the frame axis"
    )
    model_config.add_argument(
        "--lambda_orthogonality",
        type=float,
        default=1e-3,
        help="Orthogonality regularization parameter",
    )
    model_config.add_argument(
        "--dropout_prob", type=float, default=0.3, help="Dropout probability"
    )
    model_config.add_argument(
        "--M", type=int, default=8, help="Large M used in loss function"
    )
    model_config.add_argument(
        "--t",
        type=int,
        default=8,
        help="Number of negative samples used for loss function",
    )
    model_config.add_argument(
        "--muse_unsupervised_num_layers",
        type=int,
        default=2,
        help="Number of layers in the MuSE unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_unsupervised_activation",
        type=str,
        default="relu",
        help="Activation function in the MuSE unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_unsupervised_use_batch_norm",
        type=str2bool,
        default=True,
        help="Use batch normalization in the MuSE unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_unsupervised_matmul_input",
        type=str,
        default="g",
        help="Input type for matmul in the MuSE unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_unsupervised_gumbel_softmax_log",
        type=str2bool,
        default=False,
        help="Use log gumbel softmax in the MuSE unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_frameaxis_unsupervised_num_layers",
        type=int,
        default=2,
        help="Number of layers in the MuSE frameaxis unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_frameaxis_unsupervised_activation",
        type=str,
        default="relu",
        help="Activation function in the MuSE frameaxis unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_frameaxis_unsupervised_use_batch_norm",
        type=str2bool,
        default=True,
        help="Use batch normalization in the MuSE frameaxis unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_frameaxis_unsupervised_matmul_input",
        type=str,
        default="g",
        help="Input type for matmul in the MuSE frameaxis unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_frameaxis_unsupervised_gumbel_softmax_log",
        type=str2bool,
        default=False,
        help="Use log gumbel softmax in the MuSE frameaxis unsupervised encoder",
    )
    model_config.add_argument(
        "--num_negatives",
        type=int,
        default=5,
        help="Number of negative samples used for loss function",
    )
    model_config.add_argument(
        "--supervised_concat_frameaxis",
        type=str2bool,
        default=True,
        help="Concatenate frameaxis with sentence in the supervised module",
    )
    model_config.add_argument(
        "--supervised_num_layers",
        type=int,
        default=2,
        help="Number of layers in the supervised module",
    )
    model_config.add_argument(
        "--supervised_activation",
        type=str,
        default="relu",
        help="Activation function in the supervised module",
    )
    model_config.add_argument(
        "--sentence_pooling",
        type=str,
        default="mean",
        help="Pooling method for SRL embeddings",
    )
    model_config.add_argument(
        "--hidden_state",
        type=str,
        default="last",
        help="Hidden state to use for sentence embeddings",
    )
    # alternative_supervised
    model_config.add_argument(
        "--alternative_supervised",
        type=str,
        default="default",
        help="Use alternative supervised module",
    )
    model_config.add_argument(
        "--mixed_precision", type=str, default="fp16", help="Mixed precision training"
    )

    # Training Parameters
    training_params = parser.add_argument_group("Training Parameters")
    training_params.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha parameter for the loss function"
    )
    training_params.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate for the optimizer"
    )
    training_params.add_argument(
        "--adam_weight_decay",
        type=float,
        default=5e-7,
        help="Adam weight decay parameter",
    )
    training_params.add_argument(
        "--adamw_weight_decay",
        type=float,
        default=5e-7,
        help="AdamW weight decay parameter",
    )
    training_params.add_argument(
        "--ams_grad_options",
        type=str2bool,
        default=True,
        help="Use amsgrad option for the optimizer",
    )
    training_params.add_argument(
        "--optimizer", type=str, default="adamw", help="Optimizer to use for training"
    )
    training_params.add_argument(
        "--batch_size", type=int, default=24, help="Batch size"
    )
    training_params.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs"
    )
    # planned_epochs = 10
    training_params.add_argument(
        "--planned_epochs", type=int, default=10, help="Number of planned epochs"
    )
    training_params.add_argument(
        "--test_size", type=float, default=0.1, help="Size of the test set"
    )
    training_params.add_argument(
        "--tau_min", type=float, default=0.5, help="Minimum temperature parameter"
    )
    training_params.add_argument(
        "--tau_decay",
        type=float,
        default=5e-4,
        help="Decay parameter for the temperature",
    )
    training_params.add_argument(
        "--accumulation_steps", type=int, default=1, help="Gradient accumulation steps"
    )
    # clip_value
    training_params.add_argument(
        "--clip_value", type=float, default=1, help="Clip value for the gradient"
    )
    # focal_loss_alpha
    training_params.add_argument(
        "--focal_loss_alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for the focal loss",
    )
    # focal_loss_gamma
    training_params.add_argument(
        "--focal_loss_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for the focal loss",
    )

    # test_every_n_batches
    training_params.add_argument(
        "--test_every_n_batches",
        type=int,
        default=50,
        help="Save the model every n steps",
    )
    # save_threshold
    training_params.add_argument(
        "--save_threshold",
        type=float,
        default=0.5,
        help="Save the model if the accuracy is greater than the threshold",
    )
    # save_metric
    training_params.add_argument(
        "--save_metric",
        type=str,
        default="accuracy",
        help="Save the model based on the metric",
    )
    # early_stopping_patience
    training_params.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )

    # save_model true/false
    training_params.add_argument(
        "--save_model",
        type=str2bool,
        default=True,
        help="Save the model",
    )

    # Data Processing
    data_processing = parser.add_argument_group("Data Processing")
    data_processing.add_argument(
        "--num_sentences", type=int, default=32, help="Number of sentences in the input"
    )
    data_processing.add_argument(
        "--max_sentence_length",
        type=int,
        default=32,
        help="Maximum length of a sentence",
    )
    data_processing.add_argument(
        "--max_args_per_sentence",
        type=int,
        default=10,
        help="Maximum number of arguments per sentence",
    )
    data_processing.add_argument(
        "--max_arg_length", type=int, default=16, help="Maximum length of an argument"
    )

    # Input/Output Paths
    io_paths = parser.add_argument_group("Input/Output Paths")
    io_paths.add_argument(
        "--name_tokenizer",
        type=str,
        default="bert-base-uncased",
        help="Name or path of the tokenizer model",
        required=True,
    )
    io_paths.add_argument(
        "--path_name_pretrained_muse_model",
        type=str,
        default="",
        help="Path or name of the pretrained muse model",
    )
    io_paths.add_argument(
        "--path_name_bert_model",
        type=str,
        default="bert-base-uncased",
        help="Name or path of the bert model",
        required=True,
    )
    io_paths.add_argument(
        "--path_srls", type=str, default="", help="Path to the SRLs file", required=True
    )
    io_paths.add_argument(
        "--path_frameaxis",
        type=str,
        default="",
        help="Path to the FrameAxis file",
        required=True,
    )
    io_paths.add_argument(
        "--path_antonym_pairs",
        type=str,
        default="",
        help="Path to the antonym pairs file",
        required=True,
    )
    io_paths.add_argument(
        "--dim_names",
        type=str,
        default="positive,negative",
        help="Dimension names for the FrameAxis",
    )
    io_paths.add_argument(
        "--save_base_path",
        type=str,
        default="",
        help="Base path to save the model",
        required=True,
    )
    # class_column_names
    io_paths.add_argument(
        "--class_column_names",
        type=str,
        default="",
        help="Class column names",
        required=True,
    )

    # Advanced Settings
    advanced_settings = parser.add_argument_group("Advanced Settings")
    advanced_settings.add_argument(
        "--force_recalculate_srls",
        type=str2bool,
        default=False,
        help="Force recalculate SRLs",
    )
    advanced_settings.add_argument(
        "--force_recalculate_frameaxis",
        type=str2bool,
        default=False,
        help="Force recalculate FrameAxis",
    )
    advanced_settings.add_argument(
        "--sample_size", type=int, default=-1, help="Sample size"
    )
    advanced_settings.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--debug", type=str2bool, default=False, help="Debug mode")
    parser.add_argument(
        "--detect_anomaly",
        type=str2bool,
        default=False,
        help="Detect anomaly in the model",
    )

    args = parser.parse_args()

    try:
        # Setup logging
        logger = setup_logging(args.debug)

        logger.info(
            """#####################################################
#                                                   #
#              Welcome to MuSE-DLF TRAIN!           #
#                                                   #
# MuSE-DLF: Multi-View-Semantic Enhanced Dictionary #
#          Learning for Frame Classification        #
#                                                   #
#####################################################"""
        )

        logger.info(f"Running {args.model_type} model...")

        logger.info("Running the model with the following arguments: %s", args)

        config = {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "num_classes": args.num_classes,
            "lambda_orthogonality": args.lambda_orthogonality,
            "dropout_prob": args.dropout_prob,
            "M": args.M,
            "t": args.t,
            "num_sentences": args.num_sentences,
            "frameaxis_dim": args.frameaxis_dim,
            "max_sentence_length": args.max_sentence_length,
            "max_args_per_sentence": args.max_args_per_sentence,
            "max_arg_length": args.max_arg_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "test_size": args.test_size,
            "tau_min": args.tau_min,
            "tau_decay": args.tau_decay,
            "muse_unsupervised_num_layers": args.muse_unsupervised_num_layers,
            "muse_unsupervised_activation": args.muse_unsupervised_activation,
            "muse_unsupervised_use_batch_norm": args.muse_unsupervised_use_batch_norm,
            "muse_unsupervised_matmul_input": args.muse_unsupervised_matmul_input,
            "muse_unsupervised_gumbel_softmax_log": args.muse_unsupervised_gumbel_softmax_log,
            "muse_frameaxis_unsupervised_num_layers": args.muse_frameaxis_unsupervised_num_layers,
            "muse_frameaxis_unsupervised_activation": args.muse_frameaxis_unsupervised_activation,
            "muse_frameaxis_unsupervised_use_batch_norm": args.muse_frameaxis_unsupervised_use_batch_norm,
            "muse_frameaxis_unsupervised_matmul_input": args.muse_frameaxis_unsupervised_matmul_input,
            "muse_frameaxis_unsupervised_gumbel_softmax_log": args.muse_frameaxis_unsupervised_gumbel_softmax_log,
            "supervised_concat_frameaxis": args.supervised_concat_frameaxis,
            "supervised_num_layers": args.supervised_num_layers,
            "supervised_activation": args.supervised_activation,
            "alternative_supervised": args.alternative_supervised,
            "sentence_pooling": args.sentence_pooling,
            "hidden_state": args.hidden_state,
            "lr": args.lr,
            "adam_weight_decay": args.adam_weight_decay,
            "adamw_weight_decay": args.adamw_weight_decay,
            "optimizer": args.optimizer,
            "alpha": args.alpha,
            "debug": args.debug,
            "mixed_precision": args.mixed_precision,
            "num_negatives": args.num_negatives,
            "accumulation_steps": args.accumulation_steps,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "detect_anomaly": args.detect_anomaly,
            "bert_model_name": args.name_tokenizer,
            "path_name_bert_model": args.path_name_bert_model,
            "path_name_pretrained_muse_model": args.path_name_pretrained_muse_model,
        }

        # generate run name
        run_name = args.run_name

        # Initialize wandb and accelerator
        accelerator = initialize_wandb(
            args.wandb_api_key,
            args.project_name,
            args.tags,
            config,
            args.mixed_precision,
            args.accumulation_steps,
            run_name,
        )

        if args.seed:
            set_custom_seed(args.seed)
            logger.info("Random seed set to: %d", args.seed)

        # build save path
        save_path = f"{args.save_base_path}/{run_name}"

        # only muse-dlf and slmuse-dlf models are supported
        if args.model_type not in ["muse-dlf", "slmuse-dlf"]:
            raise ValueError("Only muse-dlf and slmuse-dlf models are supported")

        model = load_model(
            model_type=args.model_type,
            embedding_dim=args.embedding_dim,
            frameaxis_dim=args.frameaxis_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            num_sentences=args.num_sentences,
            dropout_prob=args.dropout_prob,
            bert_model_name=args.name_tokenizer,
            bert_model_name_or_path=args.path_name_bert_model,
            sentence_pooling=args.sentence_pooling,
            hidden_state=args.hidden_state,
            lambda_orthogonality=args.lambda_orthogonality,
            M=args.M,
            t=args.t,
            muse_unsupervised_num_layers=args.muse_unsupervised_num_layers,
            muse_unsupervised_activation=args.muse_unsupervised_activation,
            muse_unsupervised_use_batch_norm=args.muse_unsupervised_use_batch_norm,
            muse_unsupervised_matmul_input=args.muse_unsupervised_matmul_input,
            muse_unsupervised_gumbel_softmax_log=args.muse_unsupervised_gumbel_softmax_log,
            muse_frameaxis_unsupervised_num_layers=args.muse_frameaxis_unsupervised_num_layers,
            muse_frameaxis_unsupervised_activation=args.muse_frameaxis_unsupervised_activation,
            muse_frameaxis_unsupervised_use_batch_norm=args.muse_frameaxis_unsupervised_use_batch_norm,
            muse_frameaxis_unsupervised_matmul_input=args.muse_frameaxis_unsupervised_matmul_input,
            muse_frameaxis_unsupervised_gumbel_softmax_log=args.muse_frameaxis_unsupervised_gumbel_softmax_log,
            num_negatives=args.num_negatives,
            supervised_concat_frameaxis=args.supervised_concat_frameaxis,
            supervised_num_layers=args.supervised_num_layers,
            supervised_activation=args.supervised_activation,
            alternative_supervised=args.alternative_supervised,
            path_pretrained_model=args.path_name_pretrained_muse_model,
            device="cuda",
            logger=logger,
            _debug=args.debug,
            _detect_anomaly=args.detect_anomaly,
        )

        if args.name_tokenizer == "roberta-base":
            tokenizer = RobertaTokenizerFast.from_pretrained(args.name_tokenizer)
        elif args.name_tokenizer == "bert-base-uncased":
            tokenizer = BertTokenizer.from_pretrained(args.name_tokenizer)

        logger.info("Tokenizer loaded successfully")

        # Preprocess the dim_names
        dim_names = args.dim_names.split(",")

        # Split class_column_names into a list
        class_column_names = args.class_column_names.split(";")

        # Preprocess the input
        preprocessor = PreProcessor(
            tokenizer,
            batch_size=args.batch_size,
            max_sentences_per_article=args.num_sentences,
            max_sentence_length=args.max_sentence_length,
            max_args_per_sentence=args.max_args_per_sentence,
            max_arg_length=args.max_arg_length,
            test_size=args.test_size,
            frameaxis_dim=args.frameaxis_dim,
            bert_model_name=args.name_tokenizer,
            name_tokenizer=args.name_tokenizer,
            path_name_bert_model=args.path_name_bert_model,
            path_antonym_pairs=args.path_antonym_pairs,
            dim_names=dim_names,
            class_column_names=class_column_names,
            num_workers=1,
        )

        logger.info("Preprocessor loaded successfully")

        # Set stratification
        if args.model_type == "slmuse-dlf":
            statification = "single"
        elif args.model_type == "muse-dlf":
            statification = "multi"
        else:
            statification = None

        # Load the data
        _, _, train_dataloader, test_dataloader = preprocessor.get_dataloader(
            args.path_data,
            "json",
            dataframe_path={
                "srl": args.path_srls,
                "frameaxis": args.path_frameaxis,
            },
            force_recalculate={
                "srl": args.force_recalculate_srls,
                "frameaxis": args.force_recalculate_frameaxis,
            },
            sample_size=args.sample_size,
            statification=statification,
        )

        logger.info("Data loaded successfully")

        # prepare components for accelerate
        model, train_dataloader, test_dataloader = accelerator.prepare(
            model, train_dataloader, test_dataloader
        )

        # Loss function and optimizer
        if args.model_type == "slmuse-dlf":
            min_freq = 0.05  # Use a minimum frequency of 0.05 = 5%
            class_freq_dict = {
                "Capacity and Resources": 0.035401,
                "Crime and Punishment": 0.135367,
                "Cultural Identity": 0.093729,
                "Economic": 0.069791,
                "External Regulation and Reputation": 0.022252,
                "Fairness and Equality": 0.026129,
                "Health and Safety": 0.040290,
                "Legality, Constitutionality, Jurisdiction": 0.161328,
                "Morality": 0.012812,
                "Other": 0.001517,
                "Policy Prescription and Evaluation": 0.079737,
                "Political": 0.163351,
                "Public Sentiment": 0.040964,
                "Quality of Life": 0.069117,
                "Security and Defense": 0.048213,
            }

            adjusted_freqs = {k: max(v, min_freq) for k, v in class_freq_dict.items()}
            total = sum(adjusted_freqs.values())
            adjusted_freqs = {k: v / total for k, v in adjusted_freqs.items()}

            inverse_freqs = {k: 1 / v for k, v in adjusted_freqs.items()}
            total_inverse = sum(inverse_freqs.values())
            alpha_dict = {
                k: (v / total_inverse) * 100 for k, v in inverse_freqs.items()
            }

            alpha = torch.tensor(list(alpha_dict.values())).to(accelerator.device)

            class_freqs = list(class_freq_dict.values())

            # Normalize alpha values so they sum to 1
            alpha_inverse = torch.tensor(
                [torch.sqrt(torch.tensor(1.0 / freq)) for freq in class_freqs]
            ).to(accelerator.device)

            loss_function = FocalLoss(
                alpha=alpha,
                gamma=args.focal_loss_gamma,
                reduction="mean",
            )

            # loss_function = nn.CrossEntropyLoss()

            logger.info("Loss function set to FocalLoss")
        else:
            min_freq = 0.05
            class_freq_dict = {
                "Capacity_and_resources": 0.0180,
                "Crime_and_punishment": 0.1406,
                "Cultural_identity": 0.0192,
                "Economic": 0.0173,
                "External_regulation_and_reputation": 0.0749,
                "Fairness_and_equality": 0.0712,
                "Health_and_safety": 0.0378,
                "Legality_Constitutionality_and_jurisprudence": 0.1257,
                "Morality": 0.1257,
                "Policy_prescription_and_evaluation": 0.0402,
                "Political": 0.1455,
                "Public_opinion": 0.0142,
                "Quality_of_life": 0.0539,
                "Security_and_defense": 0.1158,
            }

            adjusted_freqs = {k: max(v, min_freq) for k, v in class_freq_dict.items()}
            total = sum(adjusted_freqs.values())
            adjusted_freqs = {k: v / total for k, v in adjusted_freqs.items()}

            inverse_freqs = {k: 1 / v for k, v in adjusted_freqs.items()}
            total_inverse = sum(inverse_freqs.values())
            alpha_dict = {
                k: (v / total_inverse) * 100 for k, v in inverse_freqs.items()
            }

            alpha = torch.tensor(list(alpha_dict.values())).to(accelerator.device)

            loss_function = MultiLabelFocalLoss(
                alpha=alpha,
                gamma=args.focal_loss_gamma,
                reduction="sum",
            )
            logger.info("Loss function set to Focal Loss")

        lr = args.lr

        # scale learning rate by the number of GPUs
        lr = lr * accelerator.num_processes

        optimizer_type = args.optimizer
        if optimizer_type == "adam":
            weight_decay = args.adam_weight_decay
            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "adamw":
            weight_decay = args.adamw_weight_decay
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=weight_decay,
                amsgrad=args.ams_grad_options,
            )

        # Prepare the scheduler
        steps_per_epoch = math.ceil(
            len(train_dataloader.dataset)
            / (args.batch_size * accelerator.num_processes)
        )
        total_steps = steps_per_epoch * args.epochs

        T_0 = max(total_steps // 3, steps_per_epoch)

        T_mult = 2

        eta_min = lr / 100

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )

        logger.info(f"Num GPUS: {accelerator.num_processes}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"T_0: {T_0}")
        logger.info(f"T_mult: {T_mult}")
        logger.info(f"eta_min: {eta_min}")

        logger.info("Loss function and optimizer loaded successfully")

        # prepare optimizer and scheduler
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        logger.info("Log model using WANDB", main_process_only=True)
        logger.info("WANDB project name: %s", args.project_name, main_process_only=True)
        logger.info("WANDB run name: %s", run_name, main_process_only=True)
        logger.info("WANDB tags: %s", args.tags, main_process_only=True)

        # make dir
        os.makedirs(save_path, exist_ok=True)

        # save the "config" to save_path as "config.json" file
        with open(f"{save_path}/config.json", "w") as json_file:
            json.dump(config, json_file, indent=4)

        # Calculate the number of batches per epoch
        batches_per_epoch = math.ceil(
            len(train_dataloader.dataset)
            / (args.batch_size * accelerator.num_processes)
        )

        # Set test_every_n_batches to ~1/4 of an epoch
        test_every_n_batches = max(1, batches_per_epoch // 4)

        logger.info(f"Batches per epoch: {batches_per_epoch}")
        logger.info(f"Testing every {test_every_n_batches} batches")

        # Train the model
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
            scheduler=scheduler,
            model_type=args.model_type,
            training_management="accelerate",
            tau_min=args.tau_min,
            tau_decay=args.tau_decay,
            save_path=save_path,
            run_name=run_name,
            accelerator_instance=accelerator,
            mixed_precision=args.mixed_precision,
            accumulation_steps=args.accumulation_steps,
            test_every_n_batches=test_every_n_batches,
            save_threshold=args.save_threshold,
            save_metric=args.save_metric,
            model_config=config,
            clip_value=args.clip_value,
            save_model=args.save_model,
            early_stopping_patience=args.early_stopping_patience,
        )

        trainer = accelerator.prepare(trainer)

        trainer.run_training(epochs=args.epochs, alpha=args.alpha)

        logger.info("Training completed successfully")

        accelerator.end_training()

    except Exception as e:
        if "logger" in locals():
            logger.error(f"Error during training: {e}")
        else:
            print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
