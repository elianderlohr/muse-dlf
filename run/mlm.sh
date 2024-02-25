#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=10
#SBATCH --mem-per-gpu=94000mb
#SBATCH --cpus-per-gpu=10
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com
#SBATCH --gres=gpu:1

# Load the environment variables from the .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

# Check if W&B API key is set
if [ -z "${WANDB_API_KEY}" ]; then
    echo "WANDB_API_KEY is not set. Please check your .env file."
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and Install necessary packages within the virtual environment
echo "Installing necessary packages..."
pip install --upgrade pip
pip install datasets wandb==0.16.3 transformers accelerate

DATA_PATH="data/mfc/"
OUTPUT_PATH="models/roberta-base-finetune/"

# Run the Python script with the W&B API key
echo "Starting training script..."
python src/training/mlm.py --wb_api_key $WANDB_API_KEY --data_path $DATA_PATH --output_path $OUTPUT_PATH

# Deactivate the virtual environment
deactivate
