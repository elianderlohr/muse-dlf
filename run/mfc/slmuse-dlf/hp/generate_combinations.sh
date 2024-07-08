#!/bin/bash

learning_rates=(0.0001 0.0003 0.0005 0.001)
supervised_concat_frameaxis_values=(false)
supervised_num_layers_values=(1 2 3)
supervised_activations=("gelu" "relu")
optimizers=("adam" "adamw")

# Generate all combinations and write them to a file
output_file="combinations.txt"
> $output_file

for lr in "${learning_rates[@]}"; do
    for concat_frameaxis in "${supervised_concat_frameaxis_values[@]}"; do
        for num_layers in "${supervised_num_layers_values[@]}"; do
            for activation in "${supervised_activations[@]}"; do
                for optimizer in "${optimizers[@]}"; do
                    if [ "$optimizer" == "adam" ]; then
                        weight_decay=0.0001
                    else
                        weight_decay=0.0001
                    fi
                    echo "$lr $concat_frameaxis $num_layers $activation $optimizer $weight_decay" >> $output_file
                done
            done
        done
    done
done
