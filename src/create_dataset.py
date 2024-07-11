import argparse
import random

import numpy as np

from preprocessing.pre_processor import PreProcessor
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    RobertaTokenizerFast,
)
import warnings
import wandb

from pathlib import Path

from utils.logging_manager import LoggerManager
import pickle

# Suppress specific warnings from numpy
warnings.filterwarnings(
    "ignore", message="Mean of empty slice.", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

wandb.require("core")


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
    # project
    required_args.add_argument(
        "----project_name",
        type=str,
        default="mfc-roberta-finetune",
        help="Project name",
    )
    required_args.add_argument(
        "--wandb_api_key", type=str, required=True, help="Wandb API key"
    )
    required_args.add_argument(
        "--frameaxis_dim", type=int, default=10, help="Dimension of the frame axis"
    )
    required_args.add_argument(
        "--test_size", type=float, default=0.1, help="Size of the test set"
    )
    required_args.add_argument(
        "--num_sentences", type=int, default=32, help="Number of sentences in the input"
    )
    required_args.add_argument(
        "--max_sentence_length",
        type=int,
        default=32,
        help="Maximum length of a sentence",
    )
    required_args.add_argument(
        "--max_args_per_sentence",
        type=int,
        default=10,
        help="Maximum number of arguments per sentence",
    )
    required_args.add_argument(
        "--max_arg_length", type=int, default=16, help="Maximum length of an argument"
    )
    required_args.add_argument(
        "--name_tokenizer",
        type=str,
        default="bert-base-uncased",
        help="Name or path of the tokenizer model",
        required=True,
    )
    required_args.add_argument(
        "--path_name_bert_model",
        type=str,
        default="bert-base-uncased",
        help="Name or path of the bert model",
        required=True,
    )
    required_args.add_argument(
        "--path_srls", type=str, default="", help="Path to the SRLs file", required=True
    )
    required_args.add_argument(
        "--path_frameaxis",
        type=str,
        default="",
        help="Path to the FrameAxis file",
        required=True,
    )
    required_args.add_argument(
        "--path_antonym_pairs",
        type=str,
        default="",
        help="Path to the antonym pairs file",
        required=True,
    )
    required_args.add_argument(
        "--dim_names",
        type=str,
        default="positive,negative",
        help="Dimension names for the FrameAxis",
    )
    # class_column_names
    required_args.add_argument(
        "--class_column_names",
        type=str,
        default="",
        help="Class column names",
        required=True,
    )
    required_args.add_argument(
        "--force_recalculate_srls",
        type=str2bool,
        default=False,
        help="Force recalculate SRLs",
    )
    required_args.add_argument(
        "--force_recalculate_frameaxis",
        type=str2bool,
        default=False,
        help="Force recalculate FrameAxis",
    )
    required_args.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.debug)

    logger.info("Create dataset started")

    if args.seed:
        set_seed(args.seed)
        logger.info("Random seed set to: %d", args.seed)

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
    )

    logger.info("Preprocessor loaded successfully")

    # Load the data
    train_dataset, test_dataset = preprocessor.get_dataset(
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
    )

    # Serialize datasets
    train_artifact_filepath = Path("./train_dataset_artifact.pkl")
    test_artifact_filepath = Path("./test_dataset_artifact.pkl")

    with train_artifact_filepath.open("wb") as f:
        pickle.dump(train_dataset, f)

    with test_artifact_filepath.open("wb") as f:
        pickle.dump(test_dataset, f)

    logger.info("Data loaded successfully")

    # save the datasets to W&B
    # Initialize W&B run
    run = wandb.init(project=args.project_name)

    # Log the train dataset artifact
    train_artifact = wandb.Artifact("train_dataset_artifact", type="dataset")
    train_artifact.add_file(train_artifact_filepath)
    run.log_artifact(train_artifact)

    # Log the test dataset artifact
    test_artifact = wandb.Artifact("test_dataset_artifact", type="dataset")
    test_artifact.add_file(test_artifact_filepath)
    run.log_artifact(test_artifact)

    # Link the artifacts
    run.link_artifact(
        train_artifact,
        target_path=f"elianderlohr-org/wandb-registry-dataset/{args.project_name}",
    )
    run.link_artifact(
        test_artifact,
        target_path=f"elianderlohr-org/wandb-registry-dataset/{args.project_name}",
    )

    # Finish the run
    run.finish()


if __name__ == "__main__":
    main()
