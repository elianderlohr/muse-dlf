#!/bin/bash

# SLURM Directives
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=muse-dlf-train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elias.anderlohr@gmail.com
#SBATCH --gres=gpu:2

echo "===================== Job Details ====================="
# Activate the virtual environment
echo "Activating virtual environment..."
source run/muse-dlf/muse-dlf/bin/activate

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
        desc=*)
        DESC="${arg#*=}"
        shift # Remove from processing
        ;;
        kw=*)
        KW="${arg#*=}"
        shift # Remove from processing
        ;;
    esac
done

# Data and Output Configuration
echo "Configuring paths..."
DATA_PATH="data/mfc/data_prepared.json"
SAVE_PATH="models/muse-dlf/$(date +'%Y-%m-%d_%H-%M-%S')/"
echo "Data path: $DATA_PATH"
echo "Output path: $SAVE_PATH"

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL


# Training Script Execution
echo "=================== Training Start ==================="

echo "Launching training script with Accelerate..."
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 --config_file run/muse-dlf/train/accelerate_config.yaml src/train.py \
    --description $DESC \
    --keywords $KW \
    --wandb_api_key $WANDB_API_KEY \
    --path_data $DATA_PATH \
    --batch_size 32 \
    --epochs 25 \
    --frameaxis_dim 5 \
    --name_tokenizer roberta-base \
    --path_name_bert_model models/roberta-base-finetune/2024-03-08_11-13-01/checkpoint-32454 \
    --path_srls data/srls/mfc/FRISS_srl.pkl \
    --path_frameaxis data/frameaxis/mfc/frameaxis_contextualized_mft.pkl \
    --path_antonym_pairs data/axis/mft.json \
    --dim_names virtue,vice \
    --save_path $SAVE_PATH \
    --D_h 512 \
    --dropout_prob 0.41813713193324464 \
    --alpha 0.29939853249854825 \
    --lambda_orthogonality 0.00016402662815016467 \
    --lr 0.0018359455575357815 \
    --M 13 \
    --t 13 

# Cleanup and Closeout
echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="