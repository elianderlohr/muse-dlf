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
fi

# Data and Output Configuration
echo "Configuring paths..."
DATA_PATH="data/mfc/data_prepared.json"
SAVE_PATH="models/muse-dlf/$(date +'%Y-%m-%d_%H-%M-%S')/"
echo "Data path: $DATA_PATH"
echo "Output path: $SAVE_PATH"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL


# Training Script Execution
echo "=================== Training Start ==================="

echo "Launching training script with Accelerate..."
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 --config_file run/muse-dlf/accelerate_config.yaml src/main.py \
    --wandb_api_key $WANDB_API_KEY \
    --path_data $DATA_PATH \
    --batch_size 32 \
    --epochs 25 \
    --name_tokenizer roberta-base \
    --path_name_bert_model models/roberta-base-finetune/2024-03-08_11-13-01/checkpoint-32454 \
    --path_srls data/srls/mfc/FRISS_srl.pkl \
    --path_frameaxis data/frameaxis/mfc/frameaxis_mft.pkl \
    --path_antonym_pairs data/axis/mft.json \
    --dim_names virtue,vice \
    --save_path $SAVE_PATH \
    --sample_size 100 \

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="