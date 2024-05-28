#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=save-model

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

# Training Script Execution
echo "=================== Training Start ==================="
# echo "Setting up Accelerate configuration..."
echo "Launching training script with Accelerate..."
python src/utils/save_file.py

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="