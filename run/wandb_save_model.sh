#!/bin/bash

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

# Parameters
PROJECT=$1
DIR_PATH=$2
ARTIFACT_NAME=$3
ARTIFACT_TYPE=$4
MODEL_TYPE=$5

# Run the Python script
python src/helper/wandb_save_model.py --project "$PROJECT" --dir_path "$DIR_PATH" --artifact_name "$ARTIFACT_NAME" --artifact_type "$ARTIFACT_TYPE" --model_type "$MODEL_TYPE"
