import argparse
from model.muse.muse import MUSE
from preprocessing.pre_processor import PreProcessor
import torch
import torch.nn as nn
import torch.optim as optim

from training.trainer import Trainer
from transformers import BertTokenizer


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
    path="",
    device="cuda",
):
    # print all input parameters
    print("Parameters:")
    for key, value in locals().items():
        print(f"{key}: {value}")

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
    )

    model = model.to(device)

    if path:
        print("Loading model from path:", path)
        assert path != ""
        model.load_state_dict(torch.load(path, map_location=device))

    return model


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train MUSE model")
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="Dimension of the word embeddings",
    )
    parser.add_argument(
        "--D_h",
        type=int,
        default=768,
        help="Dimension of the hidden layer in the model",
    )
    parser.add_argument(
        "--lambda_orthogonality",
        type=float,
        default=0.1,
        help="Orthogonality regularization parameter",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=5,
        help="Number of sentences in the input",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=5,
        help="Number of frames in the input",
    )
    parser.add_argument(
        "--num_sentences",
        type=int,
        default=5,
        help="Number of sentences in the input",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        help="Number of sentences in the input",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="Number of sentences in the input",
    )
    parser.add_argument(
        "--frameaxis_dim",
        type=int,
        default=5,
        help="Number of sentences in the input",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.1,
        help="Dropout probability",
    )
    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="bert-base-uncased",
        help="Name of the BERT model",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to the model file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size",
    )
    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=32,
        help="Maximum length of a sentence",
    )
    parser.add_argument(
        "--max_args_per_sentence",
        type=int,
        default=10,
        help="Maximum number of arguments per sentence",
    )
    parser.add_argument(
        "--max_arg_length",
        type=int,
        default=16,
        help="Maximum length of an argument",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Size of the test set",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs",
    )
    # tau_min = 0.5
    parser.add_argument(
        "--tau_min",
        type=float,
        default=0.5,
        help="Minimum temperature parameter",
    )
    # tau_decay = 5e-4
    parser.add_argument(
        "--tau_decay",
        type=float,
        default=5e-4,
        help="Decay parameter for the temperature",
    )

    args = parser.parse_args()

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
        bert_model_name=args.bert_model_name,
        path=args.path,
        device="cuda",
    )

    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

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
    )

    # Load the data
    _, _, train_dataloader, test_dataloader = preprocessor.get_dataloader()

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
        tau_min=args.tau_min,
        tau_decay=args.tau_decay,
    )

    trainer.run_training(epochs=args.epochs, alpha=0.5)


if __name__ == "__main__":
    main()
