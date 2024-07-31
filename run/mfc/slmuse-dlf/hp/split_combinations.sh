#!/bin/bash

# Number of splits/jobs
num_splits=5

# Split the combinations file
split -l $(( $(wc -l < hyperparameter_combinations.txt) / num_splits )) hyperparameter_combinations.txt combination_
