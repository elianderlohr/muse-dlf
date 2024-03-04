#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --job-name=roberta-base-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com
#SBATCH --gres=gpu:2

# Reset hash to ensure the latest versions of executables are used
hash -r

# Define the desired Python version
DESIRED_PYTHON_VERSION="3.9"

# Check if Python 3.9 is installed and accessible
if ! command -v python$DESIRED_PYTHON_VERSION &> /dev/null; then
    echo "Python $DESIRED_PYTHON_VERSION is not installed. Exiting."
    exit 1
fi

echo "Python $DESIRED_PYTHON_VERSION is installed. Proceeding with virtual environment creation."

# Create a virtual environment with Python 3.9
echo "Creating virtual environment with Python 3.9..."
python3.9 -m venv venv

# Reset hash to ensure the latest versions of executables are used
hash -r

# Activate the virtual environment
source venv/bin/activate

# Reset hash to ensure the latest versions of executables are used
hash -r

# Verify Python version directly using the virtual environment's Python executable
echo "Verifying Python version..."
./venv/bin/python --version
which python
type python

# Upgrade pip and install necessary packages within the virtual environment
echo "Installing necessary packages..."
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r run/requirements.txt

# List installed packages for verification
echo "Installed packages:"
./venv/bin/python -m pip list

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
