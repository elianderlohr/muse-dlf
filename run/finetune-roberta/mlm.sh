#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com
#SBATCH --gres=gpu:4

echo "===================== Job Details ====================="
# Activate the virtual environment
echo "Activating virtual environment..."
source run/venv/bin/activate

# Environment Setup
echo "Setting up environment..."
# Verify Python version
echo "Python version:"
python --version
echo "pip version:"
python -m pip --version

# Display installed package versions for verification
echo "Installed package versions:"
python -m pip list

# Load WANDB_API_KEY from .env file
echo "Loading WANDB_API_KEY from .env file..."
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

# Check if WANDB_API_KEY is loaded
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY is not set!"
    exit 1
else
    echo "WANDB_API_KEY successfully loaded."
fi

# Data and Output Configuration
echo "Configuring paths..."
DATA_PATH="data/mfc/"
OUTPUT_PATH="models/roberta-base-finetune/$(date +'%Y-%m-%d_%H-%M-%S')/"
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training Script Execution
echo "=================== Training Start ==================="
# echo "Setting up Accelerate configuration..."
# accelerate config --config_file run/accelerate_config.yaml

echo "Launching training script with Accelerate..."
accelerate launch --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision fp16 \
    --config_file run/accelerate_config.yaml src/training/mlm.py \
    --wb_api_key $WANDB_API_KEY \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --batch_size 32 \
    --epochs 100 \
    --patience 15

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="