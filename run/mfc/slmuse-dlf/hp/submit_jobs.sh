#!/bin/bash

# Define the path to the combination files
combination_files_path="run/mfc/slmuse-dlf/hp"

# Submit jobs for each combination file
for file in ${combination_files_path}/combination_*
do
    split_id=${file##*_} # Extract the split identifier
    sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=mfc-slmuse-dlf-batch-${split_id}
#SBATCH --gres=gpu:4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
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
while read lr dropout_prob weight_decay clip_value batch_size focal_gamma
do
    index=\$((index + 1))
    run_name="run-split${split_id}-idx\${index}"
    TAGS="split=${split_id},index=\${index},lr=\${lr},dropout_prob=\${dropout_prob},weight_decay=\${weight_decay},clip_value=\${clip_value},batch_size=\${batch_size},focal_gamma=\${focal_gamma}"

    echo "Cleanup WANDB cache..."
    wandb artifact cache cleanup 500m
    echo "WANDB cache cleanup complete."

    echo "=================== Training Start ==================="
    echo "Launching training script with Accelerate..."
    accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision fp16 --config_file run/mfc/slmuse-dlf/train/accelerate_config.yaml src/start_train.py \
        --model_type slmuse-dlf \
        --project_name slmuse-dlf \
        --run_name \$run_name \
        --tags \$TAGS \
        --wandb_api_key \$WANDB_API_KEY \
        --path_data data/mfc/immigration_labeled_preprocessed.json \
        --epochs 10 \
        --planned_epochs 20 \
        --frameaxis_dim 10 \
        --name_tokenizer roberta-base \
        --path_name_bert_model models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482 \
        --path_srls data/srls/mfc/mfc_labeled.pkl \
        --path_frameaxis data/frameaxis/mfc/frameaxis_mft.pkl \
        --path_antonym_pairs data/axis/mft.json \
        --class_column_names "Capacity and Resources;Crime and Punishment;Cultural Identity;Economic;External Regulation and Reputation;Fairness and Equality;Health and Safety;Legality, Constitutionality, Jurisdiction;Morality;Other;Policy Prescription and Evaluation;Political;Public Sentiment;Quality of Life;Security and Defense" \
        --dim_names virtue,vice \
        --save_base_path models/slmuse-dlf/ \
        --embedding_dim 768 \
        --hidden_dim 768 \
        --num_classes 15 \
        --dropout_prob \$dropout_prob \
        --focal_loss_gamma \$focal_gamma \
        --lambda_orthogonality 1e-3 \
        --lr \$lr \
        --M 8 \
        --t 8 \
        --batch_size \$batch_size \
        --num_sentences 24 \
        --max_sentence_length 64 \
        --max_args_per_sentence 10 \
        --max_arg_length 18 \
        --muse_unsupervised_num_layers 2 \
        --muse_unsupervised_activation relu \
        --muse_unsupervised_use_batch_norm True \
        --muse_unsupervised_matmul_input g \
        --muse_unsupervised_gumbel_softmax_log False \
        --muse_frameaxis_unsupervised_num_layers 2 \
        --muse_frameaxis_unsupervised_activation relu \
        --muse_frameaxis_unsupervised_use_batch_norm True \
        --muse_frameaxis_unsupervised_matmul_input g \
        --muse_frameaxis_unsupervised_gumbel_softmax_log False \
        --num_negatives 64 \
        --supervised_concat_frameaxis True \
        --supervised_num_layers 2 \
        --supervised_activation gelu \
        --optimizer adamw \
        --adamw_weight_decay \$weight_decay \
        --sentence_pooling mean \
        --hidden_state second_to_last \
        --tau_decay 0.0005 \
        --tau_min 0.5 \
        --seed 42 \
        --mixed_precision fp16 \
        --accumulation_steps 1 \
        --alternative_supervised alt6 \
        --clip_value \$clip_value \
        --save_model False \
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