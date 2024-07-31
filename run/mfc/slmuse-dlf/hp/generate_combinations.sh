#!/bin/bash

# Define ranges for hyperparameters
dropout_probs=(0.3 0.4 0.5 0.6)
learning_rates=(2e-4 1e-4 5e-5)
weight_decays=(1e-6 5e-6 1e-5)
clip_values=(0.5 1.0 2.0)
batch_sizes=(16 32 64)
focal_loss_gammas=(1.5 2.0 2.5)

# Output file to store combinations
output_file="hyperparameter_combinations.txt"
> $output_file

# Generate all combinations and write them to a file
for dropout in "${dropout_probs[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for weight_decay in "${weight_decays[@]}"; do
            for clip_value in "${clip_values[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for focal_gamma in "${focal_loss_gammas[@]}"; do
                        echo "$lr $dropout $weight_decay $clip_value $batch_size $focal_gamma" >> $output_file
                    done
                done
            done
        done
    done
done

echo "Hyperparameter combinations have been written to $output_file"

# Count the number of combinations
num_combinations=$(wc -l < $output_file)
echo "Total number of combinations: $num_combinations"