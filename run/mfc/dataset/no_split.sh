#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=mfc-slmuse-dlf-dataset-1
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
echo "Data path: $DATA_PATH"

CLASS_COLUMN_NAMES="Capacity and Resources;Crime and Punishment;Cultural Identity;Economic;External Regulation and Reputation;Fairness and Equality;Health and Safety;Legality, Constitutionality, Jurisdiction;Morality;Other;Policy Prescription and Evaluation;Political;Public Sentiment;Quality of Life;Security and Defense"
echo "Class column names: $CLASS_COLUMN_NAMES"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0


# Training Script Execution
echo "=================== Training Start ==================="

echo "Launching dataset creation script..."

echo "______________________________________________________"

python src/create_dataset.py \
    --project_name slmuse-dlf \
    --wandb_api_key $WANDB_API_KEY \
    --path_data $DATA_PATH \
    --frameaxis_dim 10 \
    --name_tokenizer roberta-base \
    --path_name_bert_model models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482 \
    --path_srls data/srls/mfc/mfc_labeled.pkl \
    --path_frameaxis data/frameaxis/mfc/frameaxis_mft.pkl \
    --path_antonym_pairs data/axis/mft.json \
    --class_column_names "$CLASS_COLUMN_NAMES" \
    --dim_names virtue,vice \
    --num_sentences 24 \
    --max_sentence_length 64 \
    --max_args_per_sentence 10 \
    --max_arg_length 18 \
    --force_recalculate_srls False \
    --force_recalculate_frameaxis False \
    --path_frameaxis_microframe data/frameaxis/mfc/frameaxis_mft_microframes.pkl \
    --train_mode True \
    --artifact_name mfc-dataset-unsplitted \
    --split_train_test False
echo "______________________________________________________"

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="
