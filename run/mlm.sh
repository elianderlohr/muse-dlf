#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=10
#SBATCH --mem=128000
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com

# Install necessary packages
pip install datasets wandb

# Check if W&B API key is provided as a command-line argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 WANDB_API_KEY"
    exit 1
fi


DATA_PATH="../data/mfc/"
OUTPUT_PATH="../models/roberta-base-finetune/"

# Set the W&B API key as an environment variable
WANDB_API_KEY=$1

# Run the Python script with the W&B API key
python train_model.py --wb_api_key $WANDB_API_KEY --data_path $DATA_PATH --output_path $OUTPUT_PATH
