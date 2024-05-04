import argparse
import numpy as np
from training.mlm import RoBERTaMLM


def main():
    parser = argparse.ArgumentParser(
        description="Multi-parameter sweep for MLM model training"
    )
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--data_path", type=str, help="Base path for data")
    parser.add_argument("--output_path", type=str, help="Output path for model")
    parser.add_argument("--project_name", type=str, help="Wandb project name")
    parser.add_argument("--patience", type=int, help="Patience for early stopping")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--wb_api_key", type=str, help="Wandb api key")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument(
        "--learning_rate_min", type=float, default=1e-6, help="Minimum learning rate"
    )
    parser.add_argument(
        "--learning_rate_max", type=float, default=1e-4, help="Maximum learning rate"
    )
    parser.add_argument(
        "--num_learning_rates",
        type=int,
        default=5,
        help="Number of learning rates to try",
    )

    args = parser.parse_args()

    # Generate learning rates
    learning_rates = np.linspace(
        args.learning_rate_min, args.learning_rate_max, args.num_learning_rates
    )

    # Iterate over each learning rate and instantiate RoBERTaMLM class
    for learning_rate in learning_rates:
        args.learning_rate = learning_rate
        trainer = RoBERTaMLM(args)
        trainer.train_muse()


if __name__ == "__main__":
    main()
