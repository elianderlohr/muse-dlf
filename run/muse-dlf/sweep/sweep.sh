#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com
#SBATCH --gres=gpu:2

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

export PATH_DATA="data/mfc/data_prepared.json"
export SAVE_PATH="models/muse-dlf/$(date +'%Y-%m-%d_%H-%M-%S')/"
export NAME_TOKENIZER="roberta-base"
export PATH_NAME_BERT_MODEL="models/roberta-base-finetune/2024-03-08_11-13-01/checkpoint-32454"
export PATH_SRLS="data/srls/mfc/FRISS_srl.pkl"
export PATH_FRAMEAXIS="data/frameaxis/mfc/frameaxis_mft.pkl"
export PATH_ANTONYM_PAIRS="data/axis/mft.json"
export DIM_NAMES="virtue,vice"

# Training Script Execution
echo "=================== Training Start ==================="

python -m wandb agent --count 10 elianderlohr/muse-dlf/93c0g05f

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="