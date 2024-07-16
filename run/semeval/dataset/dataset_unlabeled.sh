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
DATA_PATH="data/semeval/muse-dlf/semeval_unlabeled_train.json"
echo "Data path: $DATA_PATH"

CLASS_COLUMN_NAMES="Capacity_and_resources;Crime_and_punishment;Cultural_identity;Economic;External_regulation_and_reputation;Fairness_and_equality;Health_and_safety;Legality_Constitutionality_and_jurisprudence;Morality;Policy_prescription_and_evaluation;Political;Public_opinion;Quality_of_life;Security_and_defense"
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
    --project_name muse-dlf \
    --wandb_api_key $WANDB_API_KEY \
    --path_data $DATA_PATH \
    --frameaxis_dim 10 \
    --name_tokenizer roberta-base \
    --path_name_bert_model models/semeval-roberta-finetune/semeval-roberta-finetune-2024-06-11_08-49-35-57484/checkpoint-3922 \
    --path_srls data/srls/semeval/semeval_unlabeled_train.pkl \
    --path_frameaxis data/frameaxis/semeval/frameaxis_semeval_unlabeled_mft.pkl \
    --path_antonym_pairs data/axis/mft.json \
    --class_column_names "$CLASS_COLUMN_NAMES" \
    --dim_names virtue,vice \
    --num_sentences 64 \
    --max_sentence_length 64 \
    --max_args_per_sentence 20 \
    --max_arg_length 16 \
    --force_recalculate_srls False \
    --force_recalculate_frameaxis False \
    --artifact_name semeval-unlabeled-dataset \
    --train_mode False \
    --path_frameaxis_microframe data/frameaxis/semeval/frameaxis_semeval_mft_microframes.pkl

echo "______________________________________________________"

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="
