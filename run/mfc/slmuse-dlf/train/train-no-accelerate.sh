#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=mfc-slmuse-dlf-train-1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

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

# parse arguments

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
SAVE_PATH="models/muse-dlf/$(date +'%Y-%m-%d_%H-%M-%S')/"
echo "Data path: $DATA_PATH"
echo "Output path: $SAVE_PATH"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0


# Training Script Execution
echo "=================== Training Start ==================="

echo "Launching training script..."
python src/train.py \
    --tags $TAGS \
    --wandb_api_key $WANDB_API_KEY \
    --path_data $DATA_PATH \
    --epochs 10 \
    --frameaxis_dim 10 \
    --name_tokenizer roberta-base \
    --path_name_bert_model models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482 \
    --path_srls data/srls/mfc/mfc_labeled.pkl \
    --path_frameaxis data/frameaxis/mfc/frameaxis_mft.pkl \
    --path_antonym_pairs data/axis/mft.json \
    --dim_names virtue,vice \
    --save_path $SAVE_PATH \
    --embedding_dim 768 \
    --hidden_dim 768 \
    --num_classes 15 \
    --dropout_prob 0.3 \
    --alpha 0.1 \
    --lambda_orthogonality 0.01 \
    --lr 0.0001 \
    --M 8 \
    --t 8 \
    --batch_size 8 \
    --num_sentences 32 \
    --max_sentence_length 64 \
    --max_args_per_sentence 10 \
    --max_arg_length 10 \
    --muse_unsupervised_num_layers 2 \
    --muse_unsupervised_activation relu \
    --muse_unsupervised_use_batch_norm True \
    --muse_unsupervised_matmul_input g \
    --muse_unsupervised_gumbel_softmax_hard False \
    --muse_unsupervised_gumbel_softmax_log False \
    --muse_frameaxis_unsupervised_num_layers 2 \
    --muse_frameaxis_unsupervised_activation relu \
    --muse_frameaxis_unsupervised_use_batch_norm True \
    --muse_frameaxis_unsupervised_matmul_input g \
    --muse_frameaxis_unsupervised_concat_frameaxis True \
    --muse_frameaxis_unsupervised_gumbel_softmax_hard False \
    --muse_frameaxis_unsupervised_gumbel_softmax_log False \
    --supervised_concat_frameaxis True \
    --supervised_num_layers 2 \
    --supervised_activation relu \
    $DEBUG

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="