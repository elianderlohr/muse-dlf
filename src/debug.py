# main script

import argparse
import random
import numpy as np
from model.slmuse_dlf.muse import SRLEmbeddings
from preprocessing.pre_processor import PreProcessor
import torch
from transformers import (
    BertTokenizer,
    RobertaTokenizerFast,
)
import warnings
from utils.logging_manager import LoggerManager

# Suppress specific warnings from numpy
warnings.filterwarnings(
    "ignore", message="Mean of empty slice.", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")


def load_model(
    bert_model_name,
    bert_model_name_or_path,
    srl_embeddings_pooling,
    mixed_precision="fp16",
    device="cuda",
    logger=LoggerManager.get_logger(__name__),
    _debug=False,
):
    model = SRLEmbeddings(
        model_name_or_path=bert_model_name_or_path,
        model_type=bert_model_name,
        pooling=srl_embeddings_pooling,
        mixed_precision=mixed_precision,
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


def check_for_nans(tensor, tensor_name, logger):
    if torch.isnan(tensor).any():
        logger.error(f"NaN values detected in {tensor_name}")
        return True
    return False


def process_dataloader(dataloader, model, device, logger):
    nan_found = False
    for batch in dataloader:
        sentence_ids = batch["sentence_ids"].to(device)
        sentence_attention_masks = batch["sentence_attention_masks"].to(device)
        predicate_ids = batch["predicate_ids"].to(device)
        arg0_ids = batch["arg0_ids"].to(device)
        arg1_ids = batch["arg1_ids"].to(device)

        with torch.no_grad():
            (
                sentence_embeddings_avg,
                predicate_embeddings,
                arg0_embeddings,
                arg1_embeddings,
            ) = model(
                sentence_ids,
                sentence_attention_masks,
                predicate_ids,
                arg0_ids,
                arg1_ids,
            )

        if (
            check_for_nans(sentence_embeddings_avg, "sentence_embeddings_avg", logger)
            or check_for_nans(predicate_embeddings, "predicate_embeddings", logger)
            or check_for_nans(arg0_embeddings, "arg0_embeddings", logger)
            or check_for_nans(arg1_embeddings, "arg1_embeddings", logger)
        ):
            nan_found = True

    return nan_found


def main():
    parser = argparse.ArgumentParser(description="Debug MUSE model")

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "--path_data", type=str, required=True, help="Path to the data file"
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

    # frameaxis_dim
    model_config.add_argument(
        "--frameaxis_dim",
        type=int,
        default=2,
        help="Dimension of the FrameAxis",
    )

    # mixed_precision
    model_config.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        help="Mixed precision for the model",
    )

    training_params = parser.add_argument_group("Training Parameters")
    training_params.add_argument(
        "--batch_size", type=int, default=24, help="Batch size"
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
        type.str,
        default="positive,negative",
        help="Dimension names for the FrameAxis",
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
        "--sample_size", type.int, default=-1, help="Sample size"
    )
    advanced_settings.add_argument("--seed", type.int, default=42, help="Random seed")

    parser.add_argument("--debug", type=str2bool, default=False, help="Debug mode")

    args = parser.parse_args()

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
        "srl_embeddings_pooling": args.srl_embeddings_pooling,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        bert_model_name=args.name_tokenizer,
        bert_model_name_or_path=args.path_name_bert_model,
        srl_embeddings_pooling=args.srl_embeddings_pooling,
        mixed_precision=args.mixed_precision,
        device=device,
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
        test_size=0.1,
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

    logger.info("Checking train dataloader for NaN values...")
    train_nan_found = process_dataloader(train_dataloader, model, device, logger)
    if not train_nan_found:
        logger.info("No NaN values found in train dataloader.")

    logger.info("Checking test dataloader for NaN values...")
    test_nan_found = process_dataloader(test_dataloader, model, device, logger)
    if not test_nan_found:
        logger.info("No NaN values found in test dataloader.")


if __name__ == "__main__":
    main()
