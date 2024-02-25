#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=10
#SBATCH --mem=128000
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com

# Load the environment variables from the .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

# Install necessary packages
pip install datasets wandb transformers

DATA_PATH="data/mfc/"
OUTPUT_PATH="models/roberta-base-finetune/"

# Set the W&B API key as an environment variable
# Check if W&B API key is set
if [ -z "${WANDB_API_KEY}" ]; then
    echo "WANDB_API_KEY is not set. Please check your .env file."
    exit 1
fi


# Run the Python script with the W&B API key
python src/training/mlm.py --wb_api_key $WANDB_API_KEY --data_path $DATA_PATH --output_path $OUTPUT_PATH
