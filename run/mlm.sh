#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@example.com
#SBATCH --gres=gpu:2

echo "===================== Job Details ====================="
# Activate the virtual environment
echo "Activating virtual environment..."
source py39venv/bin/activate

# Environment Setup
echo "Setting up environment..."
# Verify Python version
echo "Python version:"
python --version

# Verify and upgrade pip
echo "Ensuring pip is installed and updated..."
python -m ensurepip --default-pip
python -m pip install --upgrade pip
echo "pip version:"
python -m pip --version

# Display installed package versions for verification
echo "Installed package versions:"
python -m pip list

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
export CUDA_VISIBLE_DEVICES=0,1

# Training Script Execution
echo "=================== Training Start ==================="
echo "Setting up Accelerate configuration..."
accelerate config default

echo "Launching training script with Accelerate..."
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 src/training/mlm.py --wb_api_key $WANDB_API_KEY --data_path $DATA_PATH --output_path $OUTPUT_PATH --batch_size 32 --epochs 10

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="