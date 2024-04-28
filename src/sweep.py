from model.muse.muse import MUSE
from preprocessing.pre_processor import PreProcessor
import torch.nn as nn

import wandb
from training.trainer import Trainer
from transformers import (
    BertTokenizer,
    RobertaTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup,
)
import warnings
import os

# Suppress specific warnings from numpy
warnings.filterwarnings(
    "ignore", message="Mean of empty slice.", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")


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
    bert_model_name="bert-base-uncased",
    path_name_bert_model="bert-base-uncased",
    supervised_sentence_prediction_method="friss",  # friss or custom
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
        bert_model_name=bert_model_name,
        bert_model_name_or_path=path_name_bert_model,
        supervised_sentence_prediction_method=supervised_sentence_prediction_method,
    )

    model = model.to(device)

    return model


def main():

    wandb.login()

    wandb_instance = wandb.init(project="muse-dlf")

    path_data = os.getenv("PATH_DATA")

    D_h = wandb.config.D_h
    lambda_orthogonality = wandb.config.lambda_orthogonality
    dropout_prob = wandb.config.dropout_prob
    M = 8
    t = 8
    alpha = wandb.config.alpha
    lr = wandb.config.lr
    K = 15
    embedding_dim = 768

    supervised_sentence_prediction_method = (
        wandb.config.supervised_sentence_prediction_method
    )

    batch_size = wandb.config.batch_size
    epochs = 10
    test_size = 0.1
    tau_min = 0.5
    tau_decay = 5e-4

    # Data Processing
    num_sentences = 32  # correct
    num_frames = 15
    frameaxis_dim = 10  # correct
    max_sentence_length = 64  # correct
    max_args_per_sentence = 10  # correct
    max_arg_length = 16  # correct

    # Input/Output Paths
    name_tokenizer = os.getenv("NAME_TOKENIZER")
    path_name_bert_model = os.getenv("PATH_NAME_BERT_MODEL")
    path_srls = os.getenv("PATH_SRLS")
    path_frameaxis = os.getenv("PATH_FRAMEAXIS")
    path_antonym_pairs = os.getenv("PATH_ANTONYM_PAIRS")
    dim_names = os.getenv("DIM_NAMES")
    save_path = os.getenv("SAVE_PATH")

    # Advanced Settings
    force_recalculate_srls = False
    force_recalculate_frameaxis = False
    sample_size = -1

    model = load_model(
        embedding_dim=embedding_dim,
        D_h=D_h,
        lambda_orthogonality=lambda_orthogonality,
        M=M,
        t=t,
        num_sentences=num_sentences,
        K=K,
        num_frames=num_frames,
        frameaxis_dim=frameaxis_dim,
        dropout_prob=dropout_prob,
        bert_model_name=name_tokenizer,
        path_name_bert_model=path_name_bert_model,
        supervised_sentence_prediction_method=supervised_sentence_prediction_method,
        device="cuda",
    )

    if name_tokenizer == "roberta-base":
        tokenizer = RobertaTokenizerFast.from_pretrained(name_tokenizer)
    elif name_tokenizer == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(name_tokenizer)

    # Preprocess the dim_names
    dim_names = dim_names.split(",")

    # Preprocess the input
    preprocessor = PreProcessor(
        tokenizer,
        batch_size=batch_size,
        max_sentences_per_article=num_sentences,
        max_sentence_length=max_sentence_length,
        max_args_per_sentence=max_args_per_sentence,
        max_arg_length=max_arg_length,
        test_size=test_size,
        frameaxis_dim=frameaxis_dim,
        bert_model_name=name_tokenizer,
        name_tokenizer=name_tokenizer,
        path_name_bert_model=path_name_bert_model,
        path_antonym_pairs=path_antonym_pairs,
        dim_names=dim_names,
    )

    # Load the data
    _, _, train_dataloader, test_dataloader = preprocessor.get_dataloader(
        path_data,
        "json",
        dataframe_path={
            "srl": path_srls,
            "frameaxis": path_frameaxis,
        },
        force_recalculate={
            "srl": force_recalculate_srls,
            "frameaxis": force_recalculate_frameaxis,
        },
        sample_size=sample_size,
    )

    # Loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-7, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=len(train_dataloader) * epochs,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler,
        training_management="wandb",
        tau_min=tau_min,
        tau_decay=tau_decay,
        save_path=save_path,
        wandb_instance=wandb_instance,
    )

    early_stopping = trainer.run_training(epochs=epochs, alpha=alpha)

    if early_stopping["early_stopped"]:
        print("Early stopping triggered.")
        wandb_instance.finish(early_stopping["stopping_code"])
    else:
        wandb_instance.finish()


if __name__ == "__main__":

    main()
