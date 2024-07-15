#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=semeval-muse-dlf-train-1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

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
DATA_PATH="data/semeval/muse-dlf/semeval_train.json"
SAVE_BASE_PATH="models/muse-dlf/"
echo "Data path: $DATA_PATH"
echo "Output path: $SAVE_BASE_PATH"

CLASS_COLUMN_NAMES="Capacity_and_resources;Crime_and_punishment;Cultural_identity;Economic;External_regulation_and_reputation;Fairness_and_equality;Health_and_safety;Legality_Constitutionality_and_jurisprudence;Morality;Policy_prescription_and_evaluation;Political;Public_opinion;Quality_of_life;Security_and_defense"
echo "Class column names: $CLASS_COLUMN_NAMES"

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

# Training Script Execution
echo "=================== Training Start ==================="

echo "Launching training script..."
echo "Using run name: $RUN_NAME"

echo "______________________________________________________"

python src/start_train.py \
    --model_type muse-dlf \
    --project_name muse-dlf \
    --run_name $RUN_NAME \
    --tags $TAGS \
    --wandb_api_key $WANDB_API_KEY \
    --path_data $DATA_PATH \
    --epochs 10 \
    --frameaxis_dim 10 \
    --name_tokenizer roberta-base \
    --path_name_bert_model models/semeval-roberta-finetune/semeval-roberta-finetune-2024-06-11_08-49-35-57484/checkpoint-3922 \
    --path_srls data/srls/semeval/semeval_train.pkl \
    --path_frameaxis data/frameaxis/semeval/frameaxis_semeval_mft.pkl \
    --path_antonym_pairs data/axis/mft.json \
    --class_column_names "$CLASS_COLUMN_NAMES" \
    --dim_names virtue,vice \
    --save_base_path $SAVE_BASE_PATH \
    --embedding_dim 768 \
    --hidden_dim 768 \
    --num_classes 14 \
    --dropout_prob 0.3 \
    --alpha 0.9 \
    --lambda_orthogonality 0.003 \
    --lr 0.0005 \
    --M 8 \
    --t 8 \
    --batch_size 8 \
    --num_sentences 64 \
    --max_sentence_length 64 \
    --max_args_per_sentence 20 \
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
    --sentence_pooling mean \
    --hidden_state second_to_last \
    --tau_decay 0.0005 \
    --tau_min 0.5 \
    --seed 42 \
    --mixed_precision fp16 \
    --accumulation_steps 1 \
    --alternative_supervised default \
    $DEBUG

echo "______________________________________________________"

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="
