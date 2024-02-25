#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=10
#SBATCH --mem=128000
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com

# Install necessary packages
pip install datasets wandb

DATA_PATH="../data/mfc/"
OUTPUT_PATH="../models/roberta-base-finetune/"

# Set the W&B API key as an environment variable
WANDB_API_KEY="XXXX"

# Run the Python script with the W&B API key
python train_model.py --wb_api_key $WANDB_API_KEY --data_path $DATA_PATH --output_path $OUTPUT_PATH
