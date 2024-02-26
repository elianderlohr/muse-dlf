#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-gpu=10
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com
#SBATCH --gres=gpu:2

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
pip install datasets wandb==0.15.11 transformers accelerate

DATA_PATH="data/mfc/"

# create output path with timestamp subdir
OUTPUT_PATH="models/roberta-base-finetune/$(date +'%Y-%m-%d_%H-%M-%S')/"

# Capture the output of nvidia-smi
NVIDIA_SMI_OUTPUT=$(nvidia-smi)

# Log the output with echo
echo "Logging GPU status..."
echo "$NVIDIA_SMI_OUTPUT" > $OUTPUT_PATH/nvidia_smi_log.txt


# Run the Python script with the W&B API key
echo "Starting training script..."
python src/training/mlm.py --wb_api_key $WANDB_API_KEY --data_path $DATA_PATH --output_path $OUTPUT_PATH

# Deactivate the virtual environment
deactivate
