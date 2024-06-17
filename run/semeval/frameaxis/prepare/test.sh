#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=semeval-frameaxis-1
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
DATA_PATH="data/semeval/muse-dlf/semeval_train.json"
PATH_ANTONYM_PAIRS="data/axis/mft.json"
DIM_NAMES="virtue,vice"
MODEL_PATH="models/semeval-roberta-finetune/semeval-roberta-finetune-2024-06-11_08-49-35-57484/checkpoint-3710"
OUTPUT_PATH="data/frameaxis/semeval/frameaxis_semeval_mft_experiment.pkl"
PATH_MICROFRAME="data/frameaxis/semeval/frameaxis_semeval_mft_microframes_experiment.pkl"

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
python src/frameaxis_test.py \
    --data_path $DATA_PATH \
    --path_antonym_pairs $PATH_ANTONYM_PAIRS \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_PATH \
    --dim_names $DIM_NAMES \
    --path_microframe $PATH_MICROFRAME

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="