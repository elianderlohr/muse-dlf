#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=frameaxis
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com
#SBATCH --gres=gpu:1

echo "===================== Job Details ====================="
# Activate the virtual environment
echo "Activating virtual environment..."
source run/muse-dlf/muse-dlf/bin/activate

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
DATA_PATH="data/mfc/immigration_articles.json"
PATH_ANTONYM_PAIRS="data/axis/mft.json"
DIM_NAMES="virtue,vice"
MODEL_PATH="models/roberta-base-finetune/2024-03-08_11-13-01/checkpoint-32454"
OUTPUT_PATH="data/frameaxis/mfc/"

echo "Data path: $DATA_PATH"
echo "Antonym pairs path: $PATH_ANTONYM_PAIRS"
echo "Model path: $MODEL_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Dimensions: $DIM_NAMES"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# Training Script Execution
echo "=================== Training Start ==================="
# echo "Setting up Accelerate configuration..."

echo "Running frameaxis.py..."
python src/other/frameaxis/frameaxis.py \
    --data_path $DATA_PATH \
    --path_antonym_pairs $PATH_ANTONYM_PAIRS \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_PATH \
    --dim_names $DIM_NAMES

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="