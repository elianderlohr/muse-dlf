import argparse
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import load_dataset
import wandb

import math

import os


class LogPerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs.get("metrics", {}).get("eval_loss")
        if eval_loss is not None:
            perplexity = math.exp(eval_loss)
            print(f"Perplexity: {perplexity}")
            # If using W&B, you can log perplexity directly to it
            kwargs["model"].log({"perplexity": perplexity})


def load_data(data_path, tokenizer):

    def tokenize_function(
        examples,
    ):
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


def main():

    parser = argparse.ArgumentParser(description="Train MUSE model")

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

    # epochs
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    # wandb api key
    parser.add_argument("--wb_api_key", type=str, default=None, help="Wandb api key")

    args = parser.parse_args()

    model_name = args.model_name

    if model_name not in ["roberta-base"]:
        raise ValueError("Model not supported")

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name)

    train_dataset, eval_dataset = load_data(args.data_path, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    # wandb login
    if args.wb_api_key:
        wandb.login(key=args.wb_api_key)
    else:
        raise ValueError("Wandb api key not provided")

    # create the args.output_path if it does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=24,
        save_steps=5_000,
        eval_steps=2_000,
        logging_steps=100,
        save_total_limit=2,
        report_to="wandb",
        run_name=args.project_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[LogPerplexityCallback()],
    )

    trainer.train()


if __name__ == "__main__":
    main()
