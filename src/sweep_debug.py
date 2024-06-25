import random
import numpy as np
import torch
from model.slmuse_dlf.muse import MUSEDLF
from preprocessing.pre_processor import PreProcessor
import torch.nn as nn

import wandb
from utils.logging_manager import LoggerManager
from training.trainer import Trainer
from transformers import (
    BertTokenizer,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import Adam, AdamW
import warnings
import os

# Suppress specific warnings from numpy
warnings.filterwarnings(
    "ignore", message="Mean of empty slice.", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

wandb.require("core")


# welcome console message
def welcome_message():
    print(
        "#####################################################\n"
        "#                                                   #\n"
        "#              Welcome to MUSE-DLF SWEEP!                     #\n"
        "#                                                   #\n"
        "# MUSE-DLF: Multi-View-Semantic Enhanced Dictionary #\n"
        "#          Learning for Frame Classification        #\n"
        "#                                                   #\n"
        "#####################################################"
    )


def load_model(
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
    muse_unsupervised_gumbel_softmax_hard,
    muse_unsupervised_gumbel_softmax_log,
    muse_frameaxis_unsupervised_num_layers,
    muse_frameaxis_unsupervised_activation,
    muse_frameaxis_unsupervised_use_batch_norm,
    muse_frameaxis_unsupervised_matmul_input,
    muse_frameaxis_unsupervised_concat_frameaxis,
    muse_frameaxis_unsupervised_gumbel_softmax_hard,
    muse_frameaxis_unsupervised_gumbel_softmax_log,
    supervised_concat_frameaxis,
    supervised_num_layers,
    supervised_activation,
    device="cuda",
    _debug=False,
):
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
        muse_unsupervised_gumbel_softmax_hard=muse_unsupervised_gumbel_softmax_hard,
        muse_unsupervised_gumbel_softmax_log=muse_unsupervised_gumbel_softmax_log,
        muse_frameaxis_unsupervised_num_layers=muse_frameaxis_unsupervised_num_layers,
        muse_frameaxis_unsupervised_activation=muse_frameaxis_unsupervised_activation,
        muse_frameaxis_unsupervised_use_batch_norm=muse_frameaxis_unsupervised_use_batch_norm,
        muse_frameaxis_unsupervised_matmul_input=muse_frameaxis_unsupervised_matmul_input,
        muse_frameaxis_unsupervised_concat_frameaxis=muse_frameaxis_unsupervised_concat_frameaxis,
        muse_frameaxis_unsupervised_gumbel_softmax_hard=muse_frameaxis_unsupervised_gumbel_softmax_hard,
        muse_frameaxis_unsupervised_gumbel_softmax_log=muse_frameaxis_unsupervised_gumbel_softmax_log,
        supervised_concat_frameaxis=supervised_concat_frameaxis,
        supervised_num_layers=supervised_num_layers,
        supervised_activation=supervised_activation,
        _debug=_debug,
    )

    model = model.to(device)

    return model


def main():

    wandb.login()
    wandb_instance = wandb.init(project="slmuse-dlf")

    # Hardcoded parameters
    embedding_dim = 768
    num_classes = 15
    M = 8
    t = 8
    num_sentences = 32
    frameaxis_dim = 10
    max_sentence_length = 32
    max_args_per_sentence = 10
    max_arg_length = 16
    test_size = 0.1
    epochs = 1

    # Parameters from wandb.config
    hidden_dim = wandb.config.hidden_dim
    dropout_prob = wandb.config.dropout_prob
    lambda_orthogonality = wandb.config.lambda_orthogonality
    muse_unsupervised_num_layers = wandb.config.muse_unsupervised_num_layers
    muse_unsupervised_activation = wandb.config.muse_unsupervised_activation
    muse_unsupervised_use_batch_norm = wandb.config.muse_unsupervised_use_batch_norm
    muse_unsupervised_matmul_input = wandb.config.muse_unsupervised_matmul_input
    muse_unsupervised_gumbel_softmax_hard = (
        wandb.config.muse_unsupervised_gumbel_softmax_hard
    )
    muse_unsupervised_gumbel_softmax_log = (
        wandb.config.muse_unsupervised_gumbel_softmax_log
    )
    muse_frameaxis_unsupervised_num_layers = (
        wandb.config.muse_frameaxis_unsupervised_num_layers
    )
    muse_frameaxis_unsupervised_activation = (
        wandb.config.muse_frameaxis_unsupervised_activation
    )
    muse_frameaxis_unsupervised_use_batch_norm = (
        wandb.config.muse_frameaxis_unsupervised_use_batch_norm
    )
    muse_frameaxis_unsupervised_matmul_input = (
        wandb.config.muse_frameaxis_unsupervised_matmul_input
    )
    muse_frameaxis_unsupervised_concat_frameaxis = (
        wandb.config.muse_frameaxis_unsupervised_concat_frameaxis
    )
    muse_frameaxis_unsupervised_gumbel_softmax_hard = (
        wandb.config.muse_frameaxis_unsupervised_gumbel_softmax_hard
    )
    muse_frameaxis_unsupervised_gumbel_softmax_log = (
        wandb.config.muse_frameaxis_unsupervised_gumbel_softmax_log
    )
    supervised_concat_frameaxis = wandb.config.supervised_concat_frameaxis
    supervised_num_layers = wandb.config.supervised_num_layers
    supervised_activation = wandb.config.supervised_activation
    srl_embeddings_pooling = wandb.config.srl_embeddings_pooling

    alpha = wandb.config.alpha
    lr = wandb.config.lr
    batch_size = wandb.config.batch_size
    tau_min = wandb.config.tau_min
    tau_decay = wandb.config.tau_decay

    optimizer_type = wandb.config.optimizer
    if optimizer_type == "adam":
        weight_decay = wandb.config.adam_weight_decay
    elif optimizer_type == "adamw":
        weight_decay = wandb.config.adamw_weight_decay
    else:
        raise ValueError("Unsupported optimizer type")

    # Input/Output Paths
    path_data = os.getenv("PATH_DATA")
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
    sample_size = 20

    # parse debug flag from environment as bool
    debug = os.getenv("DEBUG")
    print(f"DEBUG: {debug}")
    if debug == "True":
        debug = True
        LoggerManager.use_accelerate(accelerate_used=False, log_level="DEBUG")
        print("Debugging enabled")
    else:
        debug = False
        LoggerManager.use_accelerate(accelerate_used=False, log_level="INFO")
        print("Debugging disabled")

    logger = LoggerManager.get_logger(__name__)

    logger.info("Starting sweep")

    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure reproducibility in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = load_model(
        embedding_dim=embedding_dim,
        frameaxis_dim=frameaxis_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_sentences=num_sentences,
        dropout_prob=dropout_prob,
        bert_model_name=name_tokenizer,
        bert_model_name_or_path=path_name_bert_model,
        srl_embeddings_pooling=srl_embeddings_pooling,
        lambda_orthogonality=lambda_orthogonality,
        M=M,
        t=t,
        muse_unsupervised_num_layers=muse_unsupervised_num_layers,
        muse_unsupervised_activation=muse_unsupervised_activation,
        muse_unsupervised_use_batch_norm=muse_unsupervised_use_batch_norm,
        muse_unsupervised_matmul_input=muse_unsupervised_matmul_input,
        muse_unsupervised_gumbel_softmax_hard=muse_unsupervised_gumbel_softmax_hard,
        muse_unsupervised_gumbel_softmax_log=muse_unsupervised_gumbel_softmax_log,
        muse_frameaxis_unsupervised_num_layers=muse_frameaxis_unsupervised_num_layers,
        muse_frameaxis_unsupervised_activation=muse_frameaxis_unsupervised_activation,
        muse_frameaxis_unsupervised_use_batch_norm=muse_frameaxis_unsupervised_use_batch_norm,
        muse_frameaxis_unsupervised_matmul_input=muse_frameaxis_unsupervised_matmul_input,
        muse_frameaxis_unsupervised_concat_frameaxis=muse_frameaxis_unsupervised_concat_frameaxis,
        muse_frameaxis_unsupervised_gumbel_softmax_hard=muse_frameaxis_unsupervised_gumbel_softmax_hard,
        muse_frameaxis_unsupervised_gumbel_softmax_log=muse_frameaxis_unsupervised_gumbel_softmax_log,
        supervised_concat_frameaxis=supervised_concat_frameaxis,
        supervised_num_layers=supervised_num_layers,
        supervised_activation=supervised_activation,
        device="cuda",
        _debug=debug,
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

    # print size of train and test dataloader
    logger.info(f"Train dataloader size: {len(train_dataloader)}")
    logger.info(f"Test dataloader size: {len(test_dataloader)}")

    loss_function = nn.CrossEntropyLoss()

    if optimizer_type == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1,
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
