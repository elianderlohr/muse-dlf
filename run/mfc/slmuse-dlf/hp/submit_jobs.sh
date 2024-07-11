#!/bin/bash

# Define the path to the combination files
combination_files_path="run/mfc/slmuse-dlf/hp"

# Submit jobs for each combination file
for file in ${combination_files_path}/new_combinations_*
do
    split_id=${file##*_} # Extract the split identifier
    sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=mfc-slmuse-dlf-batch-${split_id}
#SBATCH --gres=gpu:4
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --partition=single

echo "===================== Job Details ====================="
echo "Job settings at start:"
scontrol show job \$SLURM_JOB_ID

echo "===================== Job Setup ====================="

# Activate the virtual environment
echo "Activating virtual environment..."
source run/venv/bin/activate

# Environment Setup
echo "Setting up environment..."
echo "Python version:"
python --version
echo "pip version:"
python -m pip --version
echo "Installed package versions:"
python -m pip list

echo "Loading WANDB_API_KEY from .env file..."
if [ -f ".env" ]; then
    export \$(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

if [ -z "\$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY is not set!"
    exit 1
else
    echo "WANDB_API_KEY successfully loaded."
fi

# Clear GPU memory function
function clear_gpu_memory {
    echo "Clearing GPU memory..."
    python -c "import torch; torch.cuda.empty_cache()"
}

# GPU Setup and Verification
echo "GPU status:"
nvidia-smi

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Iterate through combinations and run them sequentially
index=0
while read lr dropout_prob activation pooling weight_decay
do
    index=\$((index + 1))
    JOB_NAME="mfc-slmuse-dlf-train-4-split${split_id}-idx\${index}-lr\${lr}-dropout\${dropout_prob}-activation\${activation}-pooling\${pooling}"
    TAGS="split=${split_id},index=\${index},lr=\${lr},dropout_prob=\${dropout_prob},supervised_activation=\${activation},srl_embeddings_pooling=\${pooling}"

    echo "=================== Training Start ==================="
    echo "Launching training script with Accelerate..."
    accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision fp16 --config_file run/mfc/slmuse-dlf/train/accelerate_config.yaml src/start_train.py \
        --model_type slmuse-dlf \
        --project_name slmuse-dlf \
        --tags \$TAGS \
        --wandb_api_key \$WANDB_API_KEY \
        --path_data data/mfc/immigration_labeled_preprocessed.json \
        --epochs 3 \
        --planned_epochs 10 \
        --frameaxis_dim 10 \
        --name_tokenizer roberta-base \
        --path_name_bert_model models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482 \
        --path_srls data/srls/mfc/mfc_labeled.pkl \
        --path_frameaxis data/frameaxis/mfc/frameaxis_mft.pkl \
        --path_antonym_pairs data/axis/mft.json \
        --class_column_names "Capacity and Resources;Crime and Punishment;Cultural Identity;Economic;External Regulation and Reputation;Fairness and Equality;Health and Safety;Legality, Constitutionality, Jurisdiction;Morality;Other;Policy Prescription and Evaluation;Political;Public Sentiment;Quality of Life;Security and Defense" \
        --dim_names virtue,vice \
        --save_path models/slmuse-dlf/\$(date +'%Y-%m-%d_%H-%M-%S')/ \
        --embedding_dim 768 \
        --hidden_dim 768 \
        --num_classes 15 \
        --dropout_prob \$dropout_prob \
        --alpha 0.9 \
        --lambda_orthogonality 0.001626384818258435 \
        --lr \$lr \
        --M 8 \
        --t 8 \
        --batch_size 8 \
        --num_sentences 32 \
        --max_sentence_length 52 \
        --max_args_per_sentence 10 \
        --max_arg_length 10 \
        --muse_unsupervised_num_layers 1 \
        --muse_unsupervised_activation gelu \
        --muse_unsupervised_use_batch_norm True \
        --muse_unsupervised_matmul_input g \
        --muse_unsupervised_gumbel_softmax_log False \
        --muse_frameaxis_unsupervised_num_layers 1 \
        --muse_frameaxis_unsupervised_activation gelu \
        --muse_frameaxis_unsupervised_use_batch_norm True \
        --muse_frameaxis_unsupervised_matmul_input g \
        --muse_frameaxis_unsupervised_gumbel_softmax_log False \
        --num_negatives 32 \
        --supervised_concat_frameaxis false \
        --supervised_num_layers 1 \
        --supervised_activation \$activation \
        --optimizer adamw \
        --adamw_weight_decay \$weight_decay \
        --adam_weight_decay \$weight_decay \
        --sentence_pooling \$pooling \
        --tau_decay 0.0004682416233229908 \
        --tau_min 0.5 \
        --seed 42 \
        --mixed_precision fp16 \
        --accumulation_steps 2 \
        \$DEBUG

    # Clear GPU memory after each run
    clear_gpu_memory

done < $file

echo "Deactivating virtual environment..."
deactivate
echo "==================== Job Complete ===================="
sleep $((RANDOM % 120))
EOT
done
