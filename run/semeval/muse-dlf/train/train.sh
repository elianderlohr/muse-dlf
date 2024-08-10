#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=semeval-muse-dlf-train-4
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

    echo "${random_verb}-${random_noun}-${random_num}"
}

# Generate a run name
RUN_NAME=$(generate_run_name)

# Default parameters
PARAMS=(
    --path_data "data/semeval/muse-dlf/semeval_train.json"
    --semeval_dev_path_data "data/semeval/muse-dlf/semeval_dev.json"
    --wandb_api_key "$WANDB_API_KEY"
    --project_name "muse-dlf"
    --run_name "$RUN_NAME"
    --model_type "muse-dlf"
    --name_tokenizer "roberta-base"
    --path_name_bert_model "models/semeval-roberta-finetune/semeval-roberta-finetune-2024-06-11_08-49-35-57484/checkpoint-3922"
    --path_srls "data/srls/semeval/semeval_train.pkl"
    --semeval_dev_path_srls "data/srls/semeval/semeval_dev.pkl"
    --path_frameaxis "data/frameaxis/semeval/semeval_train.pkl"
    --semeval_dev_path_frameaxis "data/frameaxis/semeval/semeval_dev.pkl"
    --path_antonym_pairs "data/axis/mft.json"
    --save_base_path "models/muse-dlf/"
    --class_column_names "Capacity_and_resources;Crime_and_punishment;Cultural_identity;Economic;External_regulation_and_reputation;Fairness_and_equality;Health_and_safety;Legality_Constitutionality_and_jurisprudence;Morality;Policy_prescription_and_evaluation;Political;Public_opinion;Quality_of_life;Security_and_defense"
    --dim_names "virtue,vice"
    --epochs 10
    --planned_epochs 10
    --frameaxis_dim 10
    --embedding_dim 768
    --hidden_dim 768
    --num_classes 14
    --dropout_prob 0.3
    --alpha 0.5
    --lambda_orthogonality 1e-3
    --lr 0.0001
    --M 8
    --t 8
    --batch_size 8
    --num_sentences 32
    --max_sentence_length 64
    --max_args_per_sentence 13
    --max_arg_length 18
    --muse_unsupervised_num_layers 2
    --muse_unsupervised_activation "relu"
    --muse_unsupervised_use_batch_norm "True"
    --muse_unsupervised_matmul_input "g"
    --muse_unsupervised_gumbel_softmax_log "False"
    --muse_frameaxis_unsupervised_num_layers 2
    --muse_frameaxis_unsupervised_activation "relu"
    --muse_frameaxis_unsupervised_use_batch_norm "True"
    --muse_frameaxis_unsupervised_matmul_input "g"
    --muse_frameaxis_unsupervised_gumbel_softmax_log "False"
    --num_negatives 64
    --supervised_concat_frameaxis "True"
    --supervised_num_layers 2
    --supervised_activation "gelu"
    --optimizer "adamw"
    --adamw_weight_decay 0.000001
    --ams_grad_options "True"
    --sentence_pooling "mean"
    --hidden_state "second_to_last"
    --tau_decay 5e-4
    --tau_min 0.5
    --mixed_precision "no"
    --accumulation_steps 1
    --alternative_supervised "alt1"
    --seed 42
    --clip_value 1
    --focal_loss_gamma 2
    --early_stopping_patience 20
    --save_metric "f1_micro"
    --save_threshold 0.5
    --asymmetric_loss_clip 0.05
    --asymmetric_loss_gamma_neg 4
    --asymmetric_loss_gamma_pos 1
    --asymmetric_loss_scaler 100
)

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            PARAMS+=(--debug "true")
            shift
            ;;
        --tags)
            PARAMS+=(--tags "$2")
            shift 2
            ;;
        *)
            PARAMS+=("$1")
            if [[ $2 != --* && $2 != "" ]]; then
                PARAMS+=("$2")
                shift
            fi
            shift
            ;;
    esac
done

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

# TORCH DEBUG
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Cleanup WANDB cache..."
# call wandb cache clean
wandb artifact cache cleanup 500m
echo "WANDB cache cleanup complete."

# Training Script Execution
echo "=================== Training Start ==================="

echo "Launching training script with Accelerate..."
echo "Using run name: $RUN_NAME"

echo "______________________________________________________"

accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision fp16 --config_file run/semeval/muse-dlf/train/accelerate_config.yaml src/start_train.py "${PARAMS[@]}"

echo "______________________________________________________"

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="