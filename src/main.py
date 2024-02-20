import argparse
from model.muse.muse import MUSE
from preprocessing.pre_processor import PreProcessor
import torch
import torch.nn as nn
import torch.optim as optim

from training.trainer import Trainer
from transformers import BertTokenizer, RobertaTokenizerFast


# welcome console message
def welcome_message():
    print(
        """#####################################################
#                                                   #
#              Welcome to MUSE!                     #
#                                                   #
# MUSE-DLF: Multi-View-Semantic Enhanced Dictionary #
#          Learning for Frame Classification        #
#                                                   #
#####################################################"""
    )


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
    path_name_bert_model="bert-base-uncased",
    path_pretrained_model="",
    device="cuda",
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
        bert_model_name=path_name_bert_model,
    )

    model = model.to(device)

    if path_pretrained_model:
        print("Loading model from path_pretrained_model:", path_pretrained_model)
        assert path_pretrained_model != ""
        model.load_state_dict(torch.load(path_pretrained_model, map_location=device))

    return model


def main():

    welcome_message()

    parser = argparse.ArgumentParser(description="Train MUSE model")

    # Required arguments
    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "--path_data", type=str, required=True, help="Path to the data file"
    )
    required_args.add_argument(
        "--wandb_api_key", type=str, required=True, help="Wandb API key"
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
        default=0.1,
        help="Orthogonality regularization parameter",
    )
    model_config.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout probability"
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

    # Data Processing
    data_processing = parser.add_argument_group("Data Processing")

    data_processing.add_argument(
        "--num_sentences", type=int, default=24, help="Number of sentences in the input"
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
    )
    io_paths.add_argument(
        "--path_srls", type=str, default="", help="Path to the SRLs file"
    )
    io_paths.add_argument(
        "--path_frameaxis", type=str, default="", help="Path to the FrameAxis file"
    )
    io_paths.add_argument(
        "--path_antonym_pairs",
        type=str,
        default="",
        help="Path to the antonym pairs file",
    )
    io_paths.add_argument(
        "--save_path", type=str, default="", help="Path to save the model"
    )

    # Training Parameters
    training_params = parser.add_argument_group("Training Parameters")
    training_params.add_argument(
        "--batch_size", type=int, default=24, help="Batch size"
    )
    training_params.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs"
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

    args = parser.parse_args()

    # running the model with the given arguments
    print("Running the model with the following arguments:")
    print(args)

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
        path_name_bert_model=args.path_name_bert_model,
        path_pretrained_model=args.path_name_pretrained_muse_model,
        device="cuda",
    )

    if args.name_tokenizer == "roberta-base":
        tokenizer = RobertaTokenizerFast.from_pretrained(args.name_tokenizer)
    elif args.name_tokenizer == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(args.name_tokenizer)

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
        name_tokenizer=args.name_tokenizer,
        path_name_bert_model=args.path_name_bert_model,
        path_antonym_pairs=args.path_antonym_pairs,
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
    )

    # Loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Train the model
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        wandb_project_name="muse",
        wandb_api_key=args.wandb_api_key,
        tau_min=args.tau_min,
        tau_decay=args.tau_decay,
        save_path=args.save_path,
        config=config,
    )

    trainer.run_training(epochs=args.epochs, alpha=0.5)


if __name__ == "__main__":
    main()
