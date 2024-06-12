#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=mfc-frameaxis-1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

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

# Data and Output Configuration
echo "Configuring paths..."
DATA_PATH="data/mfc/immigration_labeled_preprocessed.json"
PATH_ANTONYM_PAIRS="data/axis/mft.json"
DIM_NAMES="virtue,vice"
MODEL_PATH="models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482"
OUTPUT_PATH="data/frameaxis/mfc/frameaxis_mft.pkl"
PATH_MICROFRAME="data/frameaxis/mfc/frameaxis_mft_microframes.pkl"


echo "Data path: $DATA_PATH"
echo "Antonym pairs path: $PATH_ANTONYM_PAIRS"
echo "Model path: $MODEL_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Dimensions: $DIM_NAMES"
echo "Microframe path: $PATH_MICROFRAME"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# Training Script Execution
echo "=================== Training Start ==================="
# echo "Setting up Accelerate configuration..."

echo "Running frameaxis.py..."
python src/frameaxis.py \
    --data_path $DATA_PATH \
    --path_antonym_pairs $PATH_ANTONYM_PAIRS \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_PATH \
    --dim_names $DIM_NAMES \
    --path_microframe $PATH_MICROFRAME \
    --word_blacklist "immigrant" "immigrants" "immigration" "illegal" "illegally" "illegals" "legally" "legalize" "legal"

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="