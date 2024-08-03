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
SAVE_BASE_PATH="models/slmuse-dlf/"
echo "Data path: $DATA_PATH"
echo "Output path: $SAVE_BASE_PATH"

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

# Identify network interface for NCCL
NCCL_IFNAME=$(ip -o -4 addr show up primary scope global | awk '{print $2}' | head -n 1)
echo "Using network interface: $NCCL_IFNAME"

# NCCL configuration
export NCCL_DEBUG=INFO 
export NCCL_SOCKET_IFNAME=$NCCL_IFNAME
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_TIMEOUT=1000

# Function to generate run name
generate_run_name() {
    verbs=(
        "flowing" "running" "jumping" "flying" "glowing" "shining" "burning"
        "exploding" "melting" "freezing" "crashing" "colliding" "breaking"
        "building" "growing" "shrinking" "expanding" "contracting" "twisting"
        "turning" "spinning" "rotating" "orbiting" "revolving" "circling"
        "swirling" "whirling" "whipping" "flipping" "flopping" "flapping"
        "fluttering" "flickering" "flaring" "blinking" "glinting" "gleaming"
        "glimmering" "glittering" "sparkling" "shimmering"
    )
    nouns=(
        "sound" "wave" "light" "shadow" "star" "planet" "house" "model" "car"
        "boat" "plane" "train" "bus" "truck" "bike" "motorcycle" "scooter"
        "skateboard" "surfboard" "snowboard" "skis" "helmet" "goggles" "gloves"
        "jacket" "coat" "shirt" "pants" "shorts" "shoes" "boots" "socks" "hat"
        "cap" "glasses" "watch" "ring" "necklace" "bracelet" "earrings" "belt"
        "tie" "scarf" "gloves" "mittens" "umbrella" "bag" "backpack" "purse"
        "wallet" "phone" "laptop"
    )

    random_verb=${verbs[$RANDOM % ${#verbs[@]}]}
    random_noun=${nouns[$RANDOM % ${#nouns[@]}]}
    random_num=$((1000 + RANDOM % 9000))

    run_name="${random_verb}-${random_noun}-${random_num}"
    echo "$run_name"
}

# Generate a run name
RUN_NAME=$(generate_run_name)

echo "Cleanup WANDB cache..."
# call wandb cache clean
wandb artifact cache cleanup 500m
echo "WANDB cache cleanup complete."

# Training Script Execution
echo "=================== Training Start ==================="

echo "Launching training script with Accelerate..."
echo "Using run name: $RUN_NAME"

echo "______________________________________________________"

accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision fp16 --config_file run/mfc/slmuse-dlf/train/accelerate_config.yaml src/start_train.py \
    --model_type slmuse-dlf \
    --project_name slmuse-dlf \
    --run_name $RUN_NAME \
    --tags $TAGS \
    --wandb_api_key $WANDB_API_KEY \
    --path_data $DATA_PATH \
    --epochs 20 \
    --planned_epochs 20 \
    --frameaxis_dim 10 \
    --name_tokenizer roberta-base \
    --path_name_bert_model models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482 \
    --path_srls data/srls/mfc/mfc_labeled.pkl \
    --path_frameaxis data/frameaxis/mfc/frameaxis_mft.pkl \
    --path_antonym_pairs data/axis/mft.json \
    --class_column_names "$CLASS_COLUMN_NAMES" \
    --dim_names virtue,vice \
    --save_base_path $SAVE_BASE_PATH \
    --embedding_dim 768 \
    --hidden_dim 768 \
    --num_classes 15 \
    --dropout_prob 0.3 \
    --alpha 0.5 \
    --lambda_orthogonality 1e-3 \
    --lr 0.0001 \
    --M 8 \
    --t 8 \
    --batch_size 32 \
    --num_sentences 24 \
    --max_sentence_length 64 \
    --max_args_per_sentence 10 \
    --max_arg_length 18 \
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
    --supervised_concat_frameaxis True \
    --supervised_num_layers 2 \
    --supervised_activation gelu \
    --optimizer adamw \
    --adamw_weight_decay 0.0001 \
    --ams_grad_options True \
    --sentence_pooling mean \
    --hidden_state second_to_last \
    --tau_decay 5e-4 \
    --tau_min 0.5 \
    --seed 42 \
    --mixed_precision fp16 \
    --accumulation_steps 1 \
    --alternative_supervised alt8 \
    --clip_value 1 \
    --focal_loss_gamma 2 \
    $DEBUG

echo "______________________________________________________"


# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="
