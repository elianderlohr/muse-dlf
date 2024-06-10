#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=semeval-roberta-finetune-8
#SBATCH --mem=32G
#SBATCH --gres=gpu:8

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
OUTPUT_PATH="models/semeval-roberta-finetune/"
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# Training Script Execution
echo "=================== Training Start ==================="
# echo "Setting up Accelerate configuration..."
echo "Launching training script with Accelerate..."
CUDA_VISIBLE_DEVICES=0 python src/mlm-sweep.py \
    --wb_api_key $WANDB_API_KEY \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --model_name roberta-base \
    --project_name "semeval-roberta-finetune" \
    --batch_size 8 \
    --epochs 20 \
    --patience 15 &
CUDA_VISIBLE_DEVICES=1 python src/mlm-sweep.py \
    --wb_api_key $WANDB_API_KEY \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --model_name roberta-base \
    --project_name "semeval-roberta-finetune" \
    --batch_size 16 \
    --epochs 20 \
    --patience 15 &
CUDA_VISIBLE_DEVICES=2 python src/mlm-sweep.py \
    --wb_api_key $WANDB_API_KEY \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --model_name roberta-base \
    --project_name "semeval-roberta-finetune" \
    --batch_size 24 \
    --epochs 20 \
    --patience 15 &
CUDA_VISIBLE_DEVICES=3 python src/mlm-sweep.py \
    --wb_api_key $WANDB_API_KEY \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --model_name roberta-base \
    --project_name "semeval-roberta-finetune" \
    --batch_size 32 \
    --epochs 20 \
    --patience 15 &

# Wait for all processes to complete
wait

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="