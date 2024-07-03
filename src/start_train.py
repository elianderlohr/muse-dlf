import argparse
import random

import numpy as np

from model.slmuse_dlf.muse import SLMUSEDLF
from model.muse_dlf.muse import MUSEDLF
from preprocessing.pre_processor import PreProcessor
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import Adam, AdamW
from accelerate import Accelerator
import warnings
import wandb

from training.trainer import Trainer
from utils.logging_manager import LoggerManager

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
    srl_embeddings_pooling,
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
    path_pretrained_model="",
    device="cuda",
    logger=LoggerManager.get_logger(__name__),
    _debug=False,
    _detect_anomaly=False,
):
    logger.info("Loading model of type: %s", model_type)

    if model_type == "slmuse-dlf":
        model = SLMUSEDLF(
            embedding_dim=embedding_dim,
            frameaxis_dim=frameaxis_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_sentences=num_sentences,
            dropout_prob=dropout_prob,
            bert_model_name=bert_model_name,
            bert_model_name_or_path=bert_model_name_or_path,
            srl_embeddings_pooling=srl_embeddings_pooling,
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
            _debug=_debug,
            _detect_anomaly=_detect_anomaly,
        )
    else:
        model = MUSEDLF(
            embedding_dim=embedding_dim,
            frameaxis_dim=frameaxis_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_sentences=num_sentences,
            dropout_prob=dropout_prob,
            bert_model_name=bert_model_name,
            bert_model_name_or_path=bert_model_name_or_path,
            srl_embeddings_pooling=srl_embeddings_pooling,
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
            _debug=_debug,
            _detect_anomaly=_detect_anomaly,
        )

    model = model.to(device)

    if path_pretrained_model:
        logger.info("Loading model from pretrained path: %s", path_pretrained_model)
        model.load_state_dict(torch.load(path_pretrained_model, map_location=device))

    logger.info("Model loaded successfully")
    return model


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
    if debug:
        LoggerManager.use_accelerate(accelerate_used=True, log_level="DEBUG")
    else:
        LoggerManager.use_accelerate(accelerate_used=True, log_level="INFO")
    logger = LoggerManager.get_logger(__name__)
    return logger


def initialize_wandb(wandb_api_key, project_name, tags, config, mixed_precision):
    wandb.login(key=wandb_api_key)
    accelerator = Accelerator(log_with="wandb", mixed_precision=mixed_precision)
    accelerator.init_trackers(
        project_name,
        config,
        init_kwargs={"wandb": {"tags": tags.split(",")}},
    )
    return accelerator


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train MUSE model")

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

    parser.add_argument(
        "--tags",
        type=str,
        default="MUSE, Frame Classification, FrameAxis",
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
        help="Number of layers in the MUSE unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_unsupervised_activation",
        type=str,
        default="relu",
        help="Activation function in the MUSE unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_unsupervised_use_batch_norm",
        type=str2bool,
        default=True,
        help="Use batch normalization in the MUSE unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_unsupervised_matmul_input",
        type=str,
        default="g",
        help="Input type for matmul in the MUSE unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_unsupervised_gumbel_softmax_log",
        type=str2bool,
        default=False,
        help="Use log gumbel softmax in the MUSE unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_frameaxis_unsupervised_num_layers",
        type=int,
        default=2,
        help="Number of layers in the MUSE frameaxis unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_frameaxis_unsupervised_activation",
        type=str,
        default="relu",
        help="Activation function in the MUSE frameaxis unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_frameaxis_unsupervised_use_batch_norm",
        type=str2bool,
        default=True,
        help="Use batch normalization in the MUSE frameaxis unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_frameaxis_unsupervised_matmul_input",
        type=str,
        default="g",
        help="Input type for matmul in the MUSE frameaxis unsupervised encoder",
    )
    model_config.add_argument(
        "--muse_frameaxis_unsupervised_gumbel_softmax_log",
        type=str2bool,
        default=False,
        help="Use log gumbel softmax in the MUSE frameaxis unsupervised encoder",
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
        "--srl_embeddings_pooling",
        type=str,
        default="mean",
        help="Pooling method for SRL embeddings",
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
        "--optimizer", type=str, default="adamw", help="Optimizer to use for training"
    )
    training_params.add_argument(
        "--batch_size", type=int, default=24, help="Batch size"
    )
    training_params.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs"
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
        "--accumulation_steps", type=int, default=2, help="Gradient accumulation steps"
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
        "--save_path",
        type=str,
        default="",
        help="Path to save the model",
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

    # Setup logging
    logger = setup_logging(args.debug)

    logger.info(
        """#####################################################
#                                                   #
#              Welcome to MUSE-DLF TRAIN!           #
#                                                   #
# MUSE-DLF: Multi-View-Semantic Enhanced Dictionary #
#          Learning for Frame Classification        #
#                                                   #
#####################################################"""
    )

    logger.info(f"Running {args.model_type} model...")

    logger.info("Running the model with the following arguments: %s", args)

    if args.seed:
        set_seed(args.seed)
        logger.info("Random seed set to: %d", args.seed)

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
        "srl_embeddings_pooling": args.srl_embeddings_pooling,
        "lr": args.lr,
        "adam_weight_decay": args.adam_weight_decay,
        "adamw_weight_decay": args.adamw_weight_decay,
        "optimizer": args.optimizer,
        "alpha": args.alpha,
        "debug": args.debug,
        "mixed_precision": args.mixed_precision,
        "num_negatives": args.num_negatives,
        "accumulation_steps": args.accumulation_steps,
    }

    # Initialize wandb and accelerator
    accelerator = initialize_wandb(
        args.wandb_api_key, args.project_name, args.tags, config, args.mixed_precision
    )

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
        srl_embeddings_pooling=args.srl_embeddings_pooling,
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
    )

    logger.info("Preprocessor loaded successfully")

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
    )

    # prepare components for accelerate
    model, train_dataloader, test_dataloader = accelerator.prepare(
        model, train_dataloader, test_dataloader
    )

    # Loss function and optimizer
    if args.model_type == "slmuse-dlf":
        loss_function = nn.CrossEntropyLoss()
        logger.info("Loss function set to CrossEntropyLoss")
    else:
        loss_function = nn.BCEWithLogitsLoss()
        logger.info("Loss function set to BCEWithLogitsLoss")

    lr = args.lr

    optimizer_type = args.optimizer
    if optimizer_type == "adam":
        weight_decay = args.adam_weight_decay
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        weight_decay = args.adamw_weight_decay
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=len(train_dataloader) * args.epochs,
    )

    logger.info("Loss function and optimizer loaded successfully")

    # prepare optimizer and scheduler
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    logger.info("Log model using WANDB", main_process_only=True)
    logger.info("WANDB project name: %s", args.project_name, main_process_only=True)
    logger.info("WANDB tags: %s", args.tags, main_process_only=True)

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
        save_path=args.save_path,
        accelerator_instance=accelerator,
        mixed_precision=args.mixed_precision,
        accumulation_steps=args.accumulation_steps,
    )

    trainer = accelerator.prepare(trainer)

    trainer.run_training(epochs=args.epochs, alpha=args.alpha)

    accelerator.end_training()


if __name__ == "__main__":
    main()
