#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=mfc-slmuse-dlf-sweep-4
#SBATCH --mem=32G
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
    export WANDB_API_KEY=$WANDB_API_KEY
fi

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# CUDA configuration

export PATH_DATA="data/mfc/immigration_labeled_preprocessed.json"
export SAVE_PATH="models/muse-dlf/$(date +'%Y-%m-%d_%H-%M-%S')/"
export NAME_TOKENIZER="roberta-base"
export PATH_NAME_BERT_MODEL="models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482"
export PATH_SRLS="data/srls/mfc/mfc_labeled.pkl"
export PATH_FRAMEAXIS="data/frameaxis/mfc/frameaxis_mft.pkl"
export PATH_ANTONYM_PAIRS="data/axis/mft.json"
export DIM_NAMES="virtue,vice"

# Training Script Execution
echo "=================== Training Start ==================="

CUDA_VISIBLE_DEVICES=0 python -m wandb agent --count 50 elianderlohr/slmuse-dlf/s5x0zyi8 &
CUDA_VISIBLE_DEVICES=1 python -m wandb agent --count 50 elianderlohr/slmuse-dlf/s5x0zyi8 &
CUDA_VISIBLE_DEVICES=2 python -m wandb agent --count 50 elianderlohr/slmuse-dlf/s5x0zyi8 &
CUDA_VISIBLE_DEVICES=3 python -m wandb agent --count 50 elianderlohr/slmuse-dlf/s5x0zyi8 &

# Wait for all background jobs to finish
wait

# Cleanup and Closeoutf
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="