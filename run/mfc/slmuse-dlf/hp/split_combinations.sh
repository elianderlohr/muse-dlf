#!/bin/bash

# Number of splits/jobs
num_splits=4

# Split the combinations file
split -l $(( $(wc -l < combinations.txt) / num_splits )) combinations.txt new_combinations_
