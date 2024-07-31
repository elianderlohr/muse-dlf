#!/bin/bash

# Number of splits/jobs
num_splits=5

# Split the combinations file
split -l $(( $(wc -l < new_combinations.txt) / num_splits )) new_combinations.txt combination_
