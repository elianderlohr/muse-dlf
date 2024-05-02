#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=frameaxis
#SBATCH --gres=gpu:1

echo "===================== Job Details ====================="
# Activate the virtual environment
echo "Activating virtual environment..."
source run/srl/srl_venv/bin/activate

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

# Data and Output Configuration
echo "Configuring paths..."
DATA_PATH="data/mfc/data_prepared_cleaned.json"
OUTPUT_PATH="data/srls/mfc/mfc_labeled.pkl"

echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Training Script Execution
echo "=================== Training Start ==================="
# echo "Setting up Accelerate configuration..."

echo "Running srls.py..."
python src/srls.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH 

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="