#!/bin/bash

learning_rates=(0.0001 0.0003 0.0005 0.001)
srl_embeddings_pooling=("mean" "cls")
dropout_probs=(0.1 0.2 0.3 0.4 0.5)
supervised_activations=("gelu" "relu")

# Generate all combinations and write them to a file
output_file="combinations.txt"
> $output_file

for lr in "${learning_rates[@]}"; do
    for pooling in "${srl_embeddings_pooling[@]}"; do
        for dropout in "${dropout_probs[@]}"; do
            for activation in "${supervised_activations[@]}"; do
                echo "$lr $pooling $dropout $activation" >> $output_file
            done
        done
    done
done
