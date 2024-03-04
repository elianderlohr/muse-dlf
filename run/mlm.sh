#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com
#SBATCH --gres=gpu:2

# Load Python 3.9 module if necessary
# Uncomment the next line if you're on a cluster that uses modules to manage software
# module load python/3.9

# Load the environment variables from the .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

# Check if W&B API key is set
if [ -z "${WANDB_API_KEY}" ]; then
    echo "WANDB_API_KEY is not set. Please check your .env file."
    exit 1
fi

# Create a virtual environment specifically with Python 3.9
echo "Creating virtual environment with Python 3.9..."
python3.9 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# print available python version
python --version

# Upgrade pip and Install necessary packages within the virtual environment
echo "Installing necessary packages..."
pip install --upgrade pip
pip install -r run/requirements.txt

# echo versions of installed packages
echo "Installed packages:"
pip list

DATA_PATH="data/mfc/"

# create output path with timestamp subdir
OUTPUT_PATH="models/roberta-base-finetune/$(date +'%Y-%m-%d_%H-%M-%S')/"

# Print the output of nvidia-smi
echo "GPU status:"
nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1

# Run the Python script with the W&B API key
echo "Starting training script..."

echo "Set up accelerate config using default"

# accelerate config
accelerate config default --mixed_precision fp16

echo "Start training script with accelerate launch"

# launch accelerates training script with multi_gpu and num_processes
accelerate launch --multi_gpu --num_processes 2 --num_machines 2 --mixed_precision fp16 src/training/mlm.py --wb_api_key $WANDB_API_KEY --data_path $DATA_PATH --output_path $OUTPUT_PATH --batch_size 32 --epochs 10

# Deactivate the virtual environment
deactivate
