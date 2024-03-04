#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com
#SBATCH --gres=gpu:2

# Activate the virtual environment
source py39venv/bin/activate

# Verify Python version directly using the virtual environment's Python executable
echo "Verifying Python version..."
python --version

echo "Verifying pip version..."
python -m pip --version

# print package versions
echo "Verifying package versions..."
python -m pip list

DATA_PATH="data/mfc/"

# Create output path with timestamp subdir
OUTPUT_PATH="models/roberta-base-finetune/$(date +'%Y-%m-%d_%H-%M-%S')/"

# Print GPU status
echo "GPU status:"
nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1

# Run the Python script with the W&B API key
echo "Starting training script..."

echo "Set up accelerate config using default"
# Use accelerate config with Python 3.9
accelerate config

echo "Start training script with accelerate launch"
# Ensure accelerate uses Python 3.9
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 src/training/mlm.py --wb_api_key $WANDB_API_KEY --data_path $DATA_PATH --output_path $OUTPUT_PATH --batch_size 32 --epochs 10

# Deactivate the virtual environment
deactivate
