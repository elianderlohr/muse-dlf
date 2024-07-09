#!/bin/bash

srl_embeddings_pooling=("mean" "cls")
dropout_probs=(0.1 0.2 0.3 0.4 0.5)
supervised_activations=("gelu" "relu")

# Generate all combinations and write them to a file
output_file="new_combinations.txt"
> $output_file

for pooling in "${srl_embeddings_pooling[@]}"; do
    for dropout in "${dropout_probs[@]}"; do
        for activation in "${supervised_activations[@]}"; do
            lr=0.0003
            weight_decay=0.0001
            echo "$lr $dropout $activation $pooling $weight_decay" >> $output_file
        done
    done
done

