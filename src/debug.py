# main script

import argparse
from multiprocessing import pool
import random
import numpy as np
from model.slmuse_dlf.muse import SRLEmbeddings
from preprocessing.pre_processor import PreProcessor
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    BertTokenizer,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import Adam, AdamW
from accelerate import Accelerator
import warnings
import wandb
from training.debug_trainer import DEBUGTrainer
from training.trainer import Trainer
from utils.logging_manager import LoggerManager

# Suppress specific warnings from numpy
warnings.filterwarnings(
    "ignore", message="Mean of empty slice.", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

wandb.require("core")


def load_model(
    bert_model_name,
    bert_model_name_or_path,
    srl_embeddings_pooling,
    device="cuda",
    logger=LoggerManager.get_logger(__name__),
    _debug=False,
):
    model = SRLEmbeddings(
        model_name_or_path=bert_model_name_or_path,
        model_type=bert_model_name,
        pooling=srl_embeddings_pooling,
        _debug=_debug,
    )
    model = model.to(device)
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


def main():
    parser = argparse.ArgumentParser(description="Train MUSE model")

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

    parser.add_argument(
        "--tags",
        type=str,
        default="MUSE, Frame Classification, FrameAxis",
        help="Tags to describe the model",
    )

    model_config = parser.add_argument_group("Model Configuration")

    model_config.add_argument(
        "--srl_embeddings_pooling",
        type=str,
        default="mean",
        help="Pooling method for SRL embeddings",
    )

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

    args = parser.parse_args()

    wandb.login(key=args.wandb_api_key)
    accelerator = Accelerator(log_with="wandb")
    logger = LoggerManager.get_logger(__name__)
    logger.info("Starting the MUSE-DLF training...")

    if args.seed:
        seed = args.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = {
        "num_sentences": args.num_sentences,
        "frameaxis_dim": args.frameaxis_dim,
        "max_sentence_length": args.max_sentence_length,
        "max_args_per_sentence": args.max_args_per_sentence,
        "max_arg_length": args.max_arg_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "srl_embeddings_pooling": args.srl_embeddings_pooling,
        "lr": args.lr,
        "adam_weight_decay": args.adam_weight_decay,
        "adamw_weight_decay": args.adamw_weight_decay,
        "optimizer": args.optimizer,
        "alpha": args.alpha,
        "debug": args.debug,
    }

    model = load_model(
        bert_model_name=args.name_tokenizer,
        bert_model_name_or_path=args.path_name_bert_model,
        srl_embeddings_pooling=args.srl_embeddings_pooling,
        device="cuda",
        logger=logger,
        _debug=args.debug,
    )

    logger.info("Model loaded successfully")

    if args.name_tokenizer == "roberta-base":
        tokenizer = RobertaTokenizerFast.from_pretrained(args.name_tokenizer)
    elif args.name_tokenizer == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(args.name_tokenizer)

    logger.info("Tokenizer loaded successfully")

    dim_names = args.dim_names.split(",")

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

    model, train_dataloader, test_dataloader = accelerator.prepare(
        model, train_dataloader, test_dataloader
    )

    loss_function = nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = Adam(
            model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.adamw_weight_decay
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=len(train_dataloader) * args.epochs,
    )

    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    project_name = args.project_name

    accelerator.init_trackers(
        project_name,
        config,
        init_kwargs={"wandb": {"tags": args.tags.split(",")}},
    )

    trainer = DEBUGTrainer(
        model,
        train_dataloader,
        test_dataloader,
        loss_function,
        optimizer,
        scheduler,
        accelerator,
        logger,
    )

    trainer.run_training(epochs=args.epochs, alpha=args.alpha)

    accelerator.end_training()


if __name__ == "__main__":
    main()
