#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=mfc-slmuse-dlf-train-4
#SBATCH --gres=gpu:4
#SBATCH --mem=48G

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

CLASS_COLUMN_NAMES="Capacity and Resources;Crime and Punishment;Cultural Identity;Economic;External Regulation and Reputation;Fairness and Equality;Health and Safety;Legality, Constitutionality, Jurisdiction;Morality;Other;Policy Prescription and Evaluation;Political;Public Sentiment;Quality of Life;Security and Defense"
echo "Class column names: $CLASS_COLUMN_NAMES"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Clear GPU memory function
function clear_gpu_memory {
    echo "Clearing GPU memory..."
    python -c "import torch; torch.cuda.empty_cache()"
}

# Clear GPU memory before starting
clear_gpu_memory

# Training Script Execution
echo "=================== Training Start ==================="

echo "Launching training script with Accelerate..."
accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision fp16 --config_file run/mfc/slmuse-dlf/train/accelerate_config.yaml src/start_train.py \
    --model_type slmuse-dlf \
    --project_name slmuse-dlf \
    --tags $TAGS \
    --wandb_api_key $WANDB_API_KEY \
    --path_data $DATA_PATH \
    --epochs 30 \
    --frameaxis_dim 10 \
    --name_tokenizer roberta-base \
    --path_name_bert_model models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482 \
    --path_srls data/srls/mfc/mfc_labeled.pkl \
    --path_frameaxis data/frameaxis/mfc/frameaxis_mft.pkl \
    --path_antonym_pairs data/axis/mft.json \
    --class_column_names "$CLASS_COLUMN_NAMES" \
    --dim_names virtue,vice \
    --save_path $SAVE_PATH \
    --embedding_dim 768 \
    --hidden_dim 2056 \
    --num_classes 15 \
    --dropout_prob 0.4 \
    --alpha 0.9 \
    --lambda_orthogonality 0.001626384818258435 \
    --lr 0.0003 \
    --M 8 \
    --t 8 \
    --batch_size 8 \
    --num_sentences 32 \
    --max_sentence_length 52 \
    --max_args_per_sentence 10 \
    --max_arg_length 10 \
    --muse_unsupervised_num_layers 2 \
    --muse_unsupervised_activation gelu \
    --muse_unsupervised_use_batch_norm True \
    --muse_unsupervised_matmul_input g \
    --muse_unsupervised_gumbel_softmax_log False \
    --muse_frameaxis_unsupervised_num_layers 2 \
    --muse_frameaxis_unsupervised_activation gelu \
    --muse_frameaxis_unsupervised_use_batch_norm True \
    --muse_frameaxis_unsupervised_matmul_input g \
    --muse_frameaxis_unsupervised_gumbel_softmax_log False \
    --num_negatives 128 \
    --supervised_concat_frameaxis False \
    --supervised_num_layers 2 \
    --supervised_activation gelu \
    --adamw_weight_decay 0.0001 \
    --optimizer adamw \
    --srl_embeddings_pooling mean \
    --tau_decay 0.0004682416233229908 \
    --tau_min 0.5 \
    --seed 42 \
    --mixed_precision fp16 \
    --accumulation_steps 1 \
    --alternative_supervised alt3 \
    $DEBUG

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="
