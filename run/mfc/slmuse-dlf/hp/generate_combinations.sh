#!/bin/bash

dropout_probs=(0.2 0.3 0.4)
learning_rates=(0.001 0.0005 0.0001)
weight_decays=(1e-5 1e-4 1e-3)
batch_sizes=(8 16 32)
ams_grad_options=(True)

# Generate all combinations and write them to a file
output_file="new_combinations.txt"
> $output_file

for dropout in "${dropout_probs[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for weight_decay in "${weight_decays[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                for ams_grad in "${ams_grad_options[@]}"; do
                    echo "$lr $dropout $weight_decay $batch_size $ams_grad" >> $output_file
                done
            done
        done
    done
done
