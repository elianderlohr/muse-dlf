import random
import numpy as np
import torch
from model.slmuse_dlf.muse import SLMUSEDLF
from model.muse_dlf.muse import MUSEDLF
from preprocessing.pre_processor import PreProcessor
import torch.nn as nn
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
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
import wandb
from accelerate import Accelerator

# Suppress specific warnings from numpy
warnings.filterwarnings(
    "ignore", message="Mean of empty slice.", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

wandb.require("core")


def welcome_message():
    print(
        "#####################################################\n"
        "#                                                   #\n"
        "#              Welcome to MUSE-DLF SWEEP!           #\n"
        "#                                                   #\n"
        "# MUSE-DLF: Multi-View-Semantic Enhanced Dictionary #\n"
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
    srl_embeddings_pooling,
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
        model = SLMUSEDLF(
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


def objective(trial):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    wandb.init(project="slmuse-dlf", reinit=True)
    # Define hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden_dim", 768, 2056, step=256)
    dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.4)
    lambda_orthogonality = trial.suggest_float("lambda_orthogonality", 0.0001, 0.01)
    muse_unsupervised_num_layers = trial.suggest_int(
        "muse_unsupervised_num_layers", 1, 3
    )
    muse_unsupervised_activation = trial.suggest_categorical(
        "muse_unsupervised_activation", ["relu", "elu", "gelu", "leaky_relu"]
    )
    muse_unsupervised_use_batch_norm = trial.suggest_categorical(
        "muse_unsupervised_use_batch_norm", [True, False]
    )
    muse_unsupervised_matmul_input = trial.suggest_categorical(
        "muse_unsupervised_matmul_input", [True, False]
    )
    muse_unsupervised_gumbel_softmax_log = trial.suggest_categorical(
        "muse_unsupervised_gumbel_softmax_log", [True, False]
    )
    muse_frameaxis_unsupervised_num_layers = trial.suggest_int(
        "muse_frameaxis_unsupervised_num_layers", 1, 3
    )
    muse_frameaxis_unsupervised_activation = trial.suggest_categorical(
        "muse_frameaxis_unsupervised_activation", ["relu", "elu", "gelu", "leaky_relu"]
    )
    muse_frameaxis_unsupervised_use_batch_norm = trial.suggest_categorical(
        "muse_frameaxis_unsupervised_use_batch_norm", [True, False]
    )
    muse_frameaxis_unsupervised_matmul_input = trial.suggest_categorical(
        "muse_frameaxis_unsupervised_matmul_input", [True, False]
    )
    muse_frameaxis_unsupervised_gumbel_softmax_log = trial.suggest_categorical(
        "muse_frameaxis_unsupervised_gumbel_softmax_log", [True, False]
    )
    supervised_concat_frameaxis = trial.suggest_categorical(
        "supervised_concat_frameaxis", [True, False]
    )
    supervised_num_layers = trial.suggest_int("supervised_num_layers", 1, 3)
    supervised_activation = trial.suggest_categorical(
        "supervised_activation", ["relu", "elu", "gelu", "leaky_relu"]
    )
    srl_embeddings_pooling = trial.suggest_categorical(
        "srl_embeddings_pooling", ["mean", "srl"]
    )
    lr = trial.suggest_float("lr", 1e-5, 1e-3)
    tau_decay = trial.suggest_float("tau_decay", 1e-5, 1e-3)
    optimizer_type = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3)

    tau_min = 0.5
    alpha = 0.9

    # Hardcoded parameters
    embedding_dim = 768
    num_classes = 15
    M = 8
    t = 8
    num_sentences = 32
    frameaxis_dim = 10
    max_sentence_length = 52
    max_args_per_sentence = 10
    max_arg_length = 16
    test_size = 0.1
    epochs = 10
    num_negatives = 128
    accumulation_steps = 4
    batch_size = 8

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

    force_recalculate_srls = False
    force_recalculate_frameaxis = False
    sample_size = -1

    detect_anomaly = os.getenv("DETECT_ANOMALY", "False") == "True"
    debug = os.getenv("DEBUG", "False") == "True"

    logger = LoggerManager.get_logger(__name__)

    logger.info("Starting Optuna trial")

    # Ensure reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
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
        srl_embeddings_pooling=srl_embeddings_pooling,
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
        device=device,
        logger=logger,
        _debug=debug,
        _detect_anomaly=detect_anomaly,
    )

    if name_tokenizer == "roberta-base":
        tokenizer = RobertaTokenizerFast.from_pretrained(name_tokenizer)
    elif name_tokenizer == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(name_tokenizer)

    dim_names = dim_names.split(",")

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
        num_training_steps=len(train_dataloader) * epochs,
    )

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

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
        wandb_instance=wandb,
        accumulation_steps=accumulation_steps,
        mixed_precision="fp16",
    )

    logger.info("🏋️ Starting training")
    early_stopping = trainer.run_training(epochs=epochs, alpha=alpha)
    logger.info("🏁 Training finished")

    if early_stopping["early_stopped"]:
        logger.info("✴️ Early stopping triggered.")
        wandb.finish(early_stopping["stopping_code"])
    else:
        wandb.finish()

    return trainer.get_best_metric()


def main():
    welcome_message()

    wandb.login()

    wandb_kwargs = {"project": "slmuse-dlf"}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, callbacks=[wandbc])


if __name__ == "__main__":
    main()
