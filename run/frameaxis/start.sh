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
source py39venv/bin/activate

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
TFIDF_PATH="data/frameaxis/mfc/prepare/tf_idf_extended.json"
MODEL_PATH="roberta-base"
OUTPUT_PATH="data/frameaxis/prepare/embeddings.json"

echo "TFIDF path: $TFIDF_PATH"
echo "Model path: $MODEL_PATH"
echo "Output path: $OUTPUT_PATH"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0

# Training Script Execution
echo "=================== Training Start ==================="
# echo "Setting up Accelerate configuration..."

echo "Running frameaxis.py..."
python src/other/frameaxis/frameaxis.py \
    --tfidf_path $TFIDF_PATH \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_PATH

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="