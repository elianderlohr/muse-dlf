import random
from venv import logger
from charset_normalizer import detect
import numpy as np
import torch
from model.slmuse_dlf.muse import SLMuSEDLF
from model.muse_dlf.muse import MuSEDLF
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
        "#              Welcome to MuSE-DLF SWEEP!                     #\n"
        "#                                                   #\n"
        "# MuSE-DLF: Multi-View-Semantic Enhanced Dictionary #\n"
        "#          Learning for Frame Classification        #\n"
        "#                                                   #\n"
        "#####################################################"
    )


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
    sentence_pooling,
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
    device="cuda",
    logger=LoggerManager.get_logger(__name__),
    _debug=False,
    _detect_anomaly=False,
):
    logger.info("Loading model of type: %s", model_type)

    if model_type == "slmuse-dlf":
        model = SLMuSEDLF(
            embedding_dim=embedding_dim,
            frameaxis_dim=frameaxis_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_sentences=num_sentences,
            dropout_prob=dropout_prob,
            bert_model_name=bert_model_name,
            bert_model_name_or_path=bert_model_name_or_path,
            sentence_pooling=sentence_pooling,
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
        model = MuSEDLF(
            embedding_dim=embedding_dim,
            frameaxis_dim=frameaxis_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_sentences=num_sentences,
            dropout_prob=dropout_prob,
            bert_model_name=bert_model_name,
            bert_model_name_or_path=bert_model_name_or_path,
            sentence_pooling=sentence_pooling,
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

    logger.info("Model loaded successfully")
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
    max_sentence_length = 52
    max_args_per_sentence = 10
    max_arg_length = 10
    test_size = 0.1
    epochs = 5
    planned_epochs = 10

    num_negatives = 128

    # Gradient accumulation steps
    accumulation_steps = 4

    # Batch size
    batch_size = 8

    # real batch size = batch_size * accumulation_steps = 8 * 4 = 32 (default)

    # Parameters from wandb.config
    hidden_dim = wandb.config.hidden_dim
    dropout_prob = wandb.config.dropout_prob
    lambda_orthogonality = wandb.config.lambda_orthogonality
    muse_unsupervised_num_layers = wandb.config.muse_unsupervised_num_layers
    muse_unsupervised_activation = wandb.config.muse_unsupervised_activation
    muse_unsupervised_use_batch_norm = wandb.config.muse_unsupervised_use_batch_norm
    muse_unsupervised_matmul_input = wandb.config.muse_unsupervised_matmul_input
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
    muse_frameaxis_unsupervised_gumbel_softmax_log = (
        wandb.config.muse_frameaxis_unsupervised_gumbel_softmax_log
    )
    supervised_concat_frameaxis = wandb.config.supervised_concat_frameaxis
    supervised_num_layers = wandb.config.supervised_num_layers
    supervised_activation = wandb.config.supervised_activation
    sentence_pooling = wandb.config.sentence_pooling

    alpha = wandb.config.alpha
    lr = wandb.config.lr
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
    model_type = os.getenv("MODEL_TYPE")
    class_column_names = os.getenv("CLASS_COLUMN_NAMES")

    # Advanced Settings
    force_recalculate_srls = False
    force_recalculate_frameaxis = False
    sample_size = -1

    if os.getenv("DETECT_ANOMALY", "False") == "True":
        detect_anomaly = True
    else:
        detect_anomaly = False

    # parse debug flag from environment as bool
    debug = os.getenv("DEBUG", "False")
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
        model_type=model_type,
        embedding_dim=embedding_dim,
        frameaxis_dim=frameaxis_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_sentences=num_sentences,
        dropout_prob=dropout_prob,
        bert_model_name=name_tokenizer,
        bert_model_name_or_path=path_name_bert_model,
        sentence_pooling=sentence_pooling,
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
        device="cuda",
        logger=logger,
        _debug=debug,
        _detect_anomaly=detect_anomaly,
    )

    if name_tokenizer == "roberta-base":
        tokenizer = RobertaTokenizerFast.from_pretrained(name_tokenizer)
    elif name_tokenizer == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(name_tokenizer)

    # Preprocess the dim_names
    dim_names = dim_names.split(",")

    print(f"Class column names: {class_column_names}")

    # Preprocess the class_column_names
    class_column_names = class_column_names.split(";")

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
        class_column_names=class_column_names,
    )

    # Load the data
    _, _, train_dataloader, test_dataloader = preprocessor.get_dataloaders(
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

    if model_type == "slmuse-dlf":
        loss_function = nn.CrossEntropyLoss()
        logger.info("Loss function: CrossEntropyLoss")
    else:
        loss_function = nn.BCEWithLogitsLoss()
        logger.info("Loss function: BCEWithLogitsLoss")

    if optimizer_type == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=len(train_dataloader) * planned_epochs,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler,
        model_type=model_type,
        training_management="wandb",
        tau_min=tau_min,
        tau_decay=tau_decay,
        save_path=save_path,
        wandb_instance=wandb_instance,
        accumulation_steps=accumulation_steps,
        mixed_precision="fp16",
        early_stopping=10,
    )

    logger.info("🏋️ Starting training")

    early_stopping = trainer.run_training(epochs=epochs, alpha=alpha)

    logger.info("🏁 Training finished")

    if early_stopping["early_stopped"]:
        logger.info("✴️ Early stopping triggered.")
        wandb_instance.finish(early_stopping["stopping_code"])
    else:
        wandb_instance.finish()


if __name__ == "__main__":

    main()
