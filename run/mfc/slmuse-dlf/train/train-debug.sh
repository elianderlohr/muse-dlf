#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=train-debug-mfc-slmuse-dlf
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo "===================== Job Details ====================="
# Display job settings
echo "Job settings at start:"
scontrol show job $SLURM_JOB_ID

echo "===================== Job Setup ====================="

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

# Parse arguments
for arg in "$@"
do
    case $arg in
        tags=*)
        TAGS="${arg#*=}"
        shift
        ;;
        debug=true)
        DEBUG="--debug true"
        TAGS="$TAGS,debug"
        shift
        ;;
    esac
done

# Data and Output Configuration
echo "Configuring paths..."
DATA_PATH="data/mfc/immigration_labeled_preprocessed.json"
SAVE_PATH="models/slmuse-dlf/$(date +'%Y-%m-%d_%H-%M-%S')/"
echo "Data path: $DATA_PATH"
echo "Output path: $SAVE_PATH"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0

# Clear GPU memory function
function clear_gpu_memory {
    echo "Clearing GPU memory..."
    python -c "import torch; torch.cuda.empty_cache()"
}

# Clear GPU memory before starting
clear_gpu_memory

# Training Script Execution
echo "=================== Training Start ==================="

echo "Launching debug script..."
python src/start_train.py \
    --model_type slmuse-dlf \
    --path_data $DATA_PATH \
    --frameaxis_dim 10 \
    --name_tokenizer roberta-base \
    --path_name_bert_model models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482 \
    --path_srls data/srls/mfc/mfc_labeled.pkl \
    --path_frameaxis data/frameaxis/mfc/frameaxis_mft.pkl \
    --path_antonym_pairs data/axis/mft.json \
    --dim_names virtue,vice \
    --batch_size 32 \
    --num_sentences 32 \
    --max_sentence_length 64 \
    --max_args_per_sentence 10 \
    --max_arg_length 10 \
    --srl_embeddings_pooling mean \
    --seed 42 \
    --epochs 1 \
    --sample_size 200 \
    $DEBUG

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="
