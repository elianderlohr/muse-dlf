#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com
#SBATCH --gres=gpu:2

# Create a virtual environment with Python 3.9
echo "Creating virtual environment with Python 3.9..."
python3.9 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Verify Python version
python --version
which python
type python

# Upgrade pip and install necessary packages within the virtual environment
echo "Installing necessary packages..."
pip install --upgrade pip
pip install -r run/requirements.txt

# List installed packages for verification
echo "Installed packages:"
pip list

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
