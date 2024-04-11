import argparse
from model.muse.muse import MUSE
from preprocessing.pre_processor import PreProcessor
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import StepLR

import wandb
from training.trainer import Trainer
from transformers import BertTokenizer, RobertaTokenizerFast
from accelerate import Accelerator
import warnings

from utils.logging_manager import LoggerManager

# Suppress specific warnings from numpy
warnings.filterwarnings(
    "ignore", message="Mean of empty slice.", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")


project_name = "muse-dlf"


def load_model(
    embedding_dim,
    D_h,
    lambda_orthogonality,
    M,
    t,
    num_sentences,
    K,
    num_frames,
    frameaxis_dim,
    dropout_prob,
    bert_model_name="bert-base-uncased",
    path_name_bert_model="bert-base-uncased",
    path_pretrained_model="",
    device="cuda",
    logger=LoggerManager.get_logger(__name__),
):
    # Model instantiation
    model = MUSE(
        embedding_dim,
        D_h,
        lambda_orthogonality,
        M,
        t,
        num_sentences,
        K,
        num_frames,
        frameaxis_dim=frameaxis_dim,
        dropout_prob=dropout_prob,
        bert_model_name=bert_model_name,
        bert_model_name_or_path=path_name_bert_model,
    )

    model = model.to(device)

    if path_pretrained_model:
        logger.info(
            "Loading model from path_pretrained_model: %s", path_pretrained_model
        )
        assert path_pretrained_model != ""
        model.load_state_dict(torch.load(path_pretrained_model, map_location=device))

    return model


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
        "--D_h",
        type=int,
        default=768,
        help="Dimension of the hidden layer in the model",
    )
    model_config.add_argument(
        "--lambda_orthogonality",
        type=float,
        default=1e-3,  # 10−3
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
        "--K",
        type=int,
        default=15,
        help="Number of latent classes used in the auto encoder",
    )

    # Training Parameters
    training_params = parser.add_argument_group("Training Parameters")
    training_params.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha parameter for the loss function"
    )
    # learning rate
    training_params.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate for the optimizer"
    )

    # Data Processing
    data_processing = parser.add_argument_group("Data Processing")

    data_processing.add_argument(
        "--num_sentences", type=int, default=32, help="Number of sentences in the input"
    )

    data_processing.add_argument(
        "--num_frames", type=int, default=15, help="Number of frames in the input"
    )
    data_processing.add_argument(
        "--frameaxis_dim", type=int, default=15, help="Dimension of the frame axis"
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

    # Training Parameters
    training_params = parser.add_argument_group("Training Parameters")
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

    # Advanced Settings
    advanced_settings = parser.add_argument_group("Advanced Settings")
    advanced_settings.add_argument(
        "--force_recalculate_srls",
        type=bool,
        default=False,
        help="Force recalculate SRLs",
    )
    advanced_settings.add_argument(
        "--force_recalculate_frameaxis",
        type=bool,
        default=False,
        help="Force recalculate FrameAxis",
    )

    # sample size
    advanced_settings.add_argument(
        "--sample_size", type=int, default=-1, help="Sample size"
    )

    args = parser.parse_args()

    # start with seting up wandb and accelerator

    # login to wandb
    wandb.login(key=args.wandb_api_key)

    # initialize accelerator
    accelerator = Accelerator(
        log_with="wandb",
    )

    LoggerManager.use_accelerate(accelerate_used=True, log_level="INFO")

    logger = LoggerManager.get_logger(__name__)

    logger.info(
        """#####################################################
#                                                   #
#              Welcome to MUSE!                     #
#                                                   #
# MUSE-DLF: Multi-View-Semantic Enhanced Dictionary #
#          Learning for Frame Classification        #
#                                                   #
#####################################################"""
    )

    # running the model with the given arguments
    logger.info("Running the model with the following arguments: %s", args)

    # create config dictionary
    config = {
        "embedding_dim": args.embedding_dim,
        "D_h": args.D_h,
        "lambda_orthogonality": args.lambda_orthogonality,
        "dropout_prob": args.dropout_prob,
        "M": args.M,
        "t": args.t,
        "K": args.K,
        "num_sentences": args.num_sentences,
        "num_frames": args.num_frames,
        "frameaxis_dim": args.frameaxis_dim,
        "max_sentence_length": args.max_sentence_length,
        "max_args_per_sentence": args.max_args_per_sentence,
        "max_arg_length": args.max_arg_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "test_size": args.test_size,
        "tau_min": args.tau_min,
        "tau_decay": args.tau_decay,
    }

    model = load_model(
        embedding_dim=args.embedding_dim,
        D_h=args.D_h,
        lambda_orthogonality=args.lambda_orthogonality,
        M=args.M,
        t=args.t,
        num_sentences=args.num_sentences,
        K=args.K,
        num_frames=args.num_frames,
        frameaxis_dim=args.frameaxis_dim,
        dropout_prob=args.dropout_prob,
        bert_model_name=args.name_tokenizer,
        path_name_bert_model=args.path_name_bert_model,
        path_pretrained_model=args.path_name_pretrained_muse_model,
        device="cuda",
        logger=logger,
    )

    if args.name_tokenizer == "roberta-base":
        tokenizer = RobertaTokenizerFast.from_pretrained(args.name_tokenizer)
    elif args.name_tokenizer == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(args.name_tokenizer)

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
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-7, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=len(train_dataloader) * args.epochs,
    )

    # prepare optimizer and scheduler
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    accelerator.init_trackers(
        project_name,
        config,
        init_kwargs={"wandb": {"tags": args.tags.split(",")}},
    )

    logger.info("Log model using WANDB")
    logger.info("WANDB project name: %s", project_name)
    logger.info("WANDB tags: %s", args.tags)

    # Train the model
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler,
        training_management="accelerate",
        tau_min=args.tau_min,
        tau_decay=args.tau_decay,
        save_path=args.save_path,
        accelerator_instance=accelerator,
    )

    trainer = accelerator.prepare(trainer)

    trainer.run_training(epochs=args.epochs, alpha=args.alpha)

    accelerator.end_training()


if __name__ == "__main__":
    # execute only if run as a script
    main()
