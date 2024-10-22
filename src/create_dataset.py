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

import os

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
    parser = argparse.ArgumentParser(description="Create Dataset")

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # Required arguments
    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "--path_data", type=str, required=True, help="Path to the data file"
    )
    # project
    required_args.add_argument(
        "--project_name",
        type=str,
        default="mfc-roberta-finetune",
        help="Project name",
    )
    required_args.add_argument(
        "--wandb_api_key", type=str, required=True, help="Wandb API key"
    )
    # path_frameaxis_microframe
    required_args.add_argument(
        "--path_frameaxis_microframe",
        type=str,
        default="",
        help="Path to the FrameAxis microframe file",
        required=True,
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
    # artifact name
    required_args.add_argument(
        "--artifact_name",
        type=str,
        default="mfc-roberta-finetune_dataset",
        help="Artifact name",
    )
    # stratification multi, single or none
    required_args.add_argument(
        "--stratification",
        type=str,
        default="multi",
        help="stratification the dataset",
    )

    # train_mode
    required_args.add_argument(
        "--train_mode",
        type=str2bool,
        default=True,
        help="Train mode",
    )
    # split_train_test
    required_args.add_argument(
        "--split_train_test",
        type=str2bool,
        default=True,
        help="Split train test",
    )

    # registry_name
    required_args.add_argument(
        "--registry_name",
        type=str,
        default="wandb-registry-dataset",
        help="Registry name",
    )

    required_args.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup logging
    LoggerManager.use_accelerate(accelerate_used=False, log_level="INFO")
    logger = LoggerManager.get_logger(__name__)

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

    if args.train_mode:

        if args.split_train_test:
            logger.info("Prepare data for training with train test split")
            # Load the data
            train_dataset, test_dataset, _, _ = preprocessor.get_datasets(
                args.path_data,
                "json",
                dataframe_path={
                    "srl": args.path_srls,
                    "frameaxis": args.path_frameaxis,
                    "frameaxis_microframe": args.path_frameaxis_microframe,
                },
                force_recalculate={
                    "srl": args.force_recalculate_srls,
                    "frameaxis": args.force_recalculate_frameaxis,
                },
                train_mode=args.train_mode,
                random_state=args.seed if args.seed else None,
                stratification=args.stratification,
                device=0,
            )

            # Serialize datasets
            train_artifact_filepath = Path("./train_dataset_artifact.pkl")
            test_artifact_filepath = Path("./test_dataset_artifact.pkl")

            with train_artifact_filepath.open("wb") as f:
                pickle.dump(train_dataset, f)

            with test_artifact_filepath.open("wb") as f:
                pickle.dump(test_dataset, f)

            logger.info("Data loaded successfully")

            # Initialize W&B run
            run = wandb.init(
                project=args.project_name,
                settings=wandb.Settings(_service_wait=300),
                job_type="create-dataset",
            )

            # Log the train dataset artifact
            artifact = wandb.Artifact(args.artifact_name, type="dataset")
            artifact.add_file(train_artifact_filepath)
            artifact.add_file(test_artifact_filepath)
            run.log_artifact(artifact)

            # Link the artifacts
            run.link_artifact(
                artifact,
                target_path=f"elianderlohr-org/wandb-registry-dataset/{args.registry_name}",
            )
        else:
            logger.info("Prepare data for training without train test split")
            # Load the data
            dataset = preprocessor.get_dataset(
                args.path_data,
                "json",
                dataframe_path={
                    "srl": args.path_srls,
                    "frameaxis": args.path_frameaxis,
                    "frameaxis_microframe": args.path_frameaxis_microframe,
                },
                force_recalculate={
                    "srl": args.force_recalculate_srls,
                    "frameaxis": args.force_recalculate_frameaxis,
                },
                train_mode=True,
                device=0,
            )

            # Serialize datasets
            artifact_filepath = Path("./dataset_artifact.pkl")

            logger.info("File path: %s", artifact_filepath)

            with artifact_filepath.open("wb") as f:
                pickle.dump(dataset, f)

            logger.info("Data loaded successfully")

            # Initialize W&B run
            run = wandb.init(
                project=args.project_name,
                settings=wandb.Settings(_service_wait=300),
                job_type="create-dataset",
            )

            # Log the dataset artifact
            artifact = wandb.Artifact(args.artifact_name, type="dataset")
            artifact.add_file(artifact_filepath)
            run.log_artifact(artifact)

            # Link the artifacts
            run.link_artifact(
                artifact,
                target_path=f"elianderlohr-org/wandb-registry-dataset/{args.registry_name}",
            )

            logger.info(
                f"Link the artifacts to elianderlohr-org/wandb-registry-dataset/{args.registry_name}"
            )
    else:

        logger.info("Prepare data for inference")

        dataset = preprocessor.get_datasets(
            args.path_data,
            "json",
            dataframe_path={
                "srl": args.path_srls,
                "frameaxis": args.path_frameaxis,
                "frameaxis_microframe": args.path_frameaxis_microframe,
            },
            force_recalculate={
                "srl": args.force_recalculate_srls,
                "frameaxis": args.force_recalculate_frameaxis,
            },
            train_mode=args.train_mode,
            device=0,
        )

        # Serialize datasets
        artifact_filepath = Path("./dataset_artifact.pkl")

        with artifact_filepath.open("wb") as f:
            pickle.dump(dataset, f)

        logger.info("Data loaded successfully")

        # Initialize W&B run
        run = wandb.init(
            project=args.project_name,
            settings=wandb.Settings(_service_wait=300),
            job_type="create-dataset",
        )

        # Log the dataset artifact
        artifact = wandb.Artifact(args.artifact_name, type="dataset")
        artifact.add_file(artifact_filepath)
        run.log_artifact(artifact)

        # Link the artifacts
        run.link_artifact(
            artifact,
            target_path=f"elianderlohr-org/wandb-registry-dataset/{args.registry_name}",
        )

        logger.info(
            f"Link the artifacts to elianderlohr-org/wandb-registry-dataset/{args.registry_name}"
        )

    # Finish the run
    run.finish()


if __name__ == "__main__":
    main()
