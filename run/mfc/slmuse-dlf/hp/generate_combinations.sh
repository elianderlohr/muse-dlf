#!/bin/bash

# Define ranges for hyperparameters
dropout_probs=(0.2 0.3 0.4)
learning_rates=(0.001 0.0005 0.0001)
weight_decays=(1e-5 1e-4 1e-3)
batch_sizes=(8 16 32)
ams_grad_options=(True False)
focal_loss_alphas=(0.75 0.85 0.95)
focal_loss_gammas=(2.0 2.5 3.0)

# Output file to store combinations
output_file="new_combinations.txt"
> $output_file

# Generate all combinations and write them to a file
for dropout in "${dropout_probs[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for weight_decay in "${weight_decays[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                for ams_grad in "${ams_grad_options[@]}"; do
                    for focal_alpha in "${focal_loss_alphas[@]}"; do
                        for focal_gamma in "${focal_loss_gammas[@]}"; do
                            echo "$lr $dropout $weight_decay $batch_size $ams_grad $focal_alpha $focal_gamma" >> $output_file
                        done
                    done
                done
            done
        done
    done
done
