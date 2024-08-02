import argparse
import logging
import math
import os
from datetime import datetime
import random

import torch
import wandb
from datasets import load_dataset
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.integrations import WandbCallback


class RoBERTaMLM:
    def __init__(self, args):
        self.args = args

    class LogPerplexityCallback(WandbCallback):
        def __init__(self):
            super().__init__()

        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs.get("metrics", {})
            eval_loss = metrics.get("eval_loss")

            if eval_loss is not None:
                perplexity = math.exp(eval_loss)
                logging.info(f"Perplexity: {perplexity}")

                if "wandb" in args.report_to:
                    self._wandb.log({"perplexity": perplexity})

    class EarlyStoppingCallback(TrainerCallback):
        def __init__(self, patience=3):
            super().__init__()
            self.patience = patience
            self.best_perplexity = float("inf")
            self.wait = 0
            self.stopped_epoch = 0

        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs.get("metrics", {})
            eval_loss = metrics.get("eval_loss", None)
            if eval_loss is not None:
                perplexity = math.exp(eval_loss)
                if perplexity < self.best_perplexity:
                    self.best_perplexity = perplexity
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        print(
                            f"No improvement in perplexity for {self.patience} evaluations. Stopping training."
                        )
                        control.should_training_stop = True
                        self.stopped_epoch = state.epoch

    @staticmethod
    def load_data(data_path, tokenizer):
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        train_dataset = load_dataset(
            "text", data_files={"train": data_path + "train_data.txt"}
        )["train"]
        train_dataset = train_dataset.map(tokenize_function, batched=True)

        eval_dataset = load_dataset(
            "text", data_files={"test": data_path + "test_data.txt"}
        )["test"]
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        return train_dataset, eval_dataset

    def train_muse(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # welcome
        logging.info(
            """##################################################
               #                                                #
               #         Welcome to RoBERTa Fine-Runing         #
               #                                                #
               ##################################################"""
        )

        # gpu check
        logging.info("Checking for GPU")
        if not torch.cuda.is_available():
            raise ValueError("GPU not available")

        model_name = self.args.model_name

        if model_name not in ["roberta-base"]:
            raise ValueError("Model not supported")

        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForMaskedLM.from_pretrained(model_name)

        logging.info("Model and tokenizer loaded")

        train_dataset, eval_dataset = self.load_data(self.args.data_path, tokenizer)

        # log length of train and eval dataset
        logging.info(f"Train dataset length: {len(train_dataset)}")
        logging.info(f"Eval dataset length: {len(eval_dataset)}")

        logging.info("Data loaded")

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

        # generate wandb run name use current date and time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # generate random number between 1 and 99999
        random_number = str(random.randint(1, 99999))
        custom_run_name = f"{self.args.project_name}-{current_time}-{random_number}"

        wandb.login(key=self.args.wb_api_key)

        # Initialize wandb run here if you need to pass specific configuration
        wandb.init(
            project=self.args.project_name, name=custom_run_name, config=self.args
        )

        output_path_full = os.path.join(self.args.output_path, custom_run_name)

        # create the args.output_path if it does not exist
        if not os.path.exists(output_path_full):
            os.makedirs(output_path_full)

        logging.info("Setting up Trainer")

        training_args = TrainingArguments(
            output_dir=output_path_full,
            overwrite_output_dir=True,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            save_total_limit=5,
            report_to="wandb",
            fp16=True,
            run_name=self.args.project_name,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.args.learning_rate,
        )

        logging.info("Start Hyperparameter Optimization...")

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[
                self.LogPerplexityCallback(),
                self.EarlyStoppingCallback(patience=self.args.patience),
            ],
        )

        trainer.train()

        logging.info("Hyperparameter Optimization complete")

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MuSE model")

    parser.add_argument(
        "--model_name", type=str, default="roberta-base", help="Model name"
    )

    # base path for data
    parser.add_argument(
        "--data_path", type=str, default="data/mfc/", help="Base path for data"
    )

    # output path for model
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/roberta-mfc",
        help="Output path for model",
    )

    # wandb project name
    parser.add_argument(
        "--project_name",
        type=str,
        default="roberta-finefune",
        help="Wandb project name",
    )

    # training param

    # batch size
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )

    # learning rate
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training",
    )

    # patience
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )

    # epochs
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")

    # wandb api key
    parser.add_argument("--wb_api_key", type=str, default=None, help="Wandb api key")

    args = parser.parse_args()

    trainer = RoBERTaMLM(args)
    trainer.train_muse()
