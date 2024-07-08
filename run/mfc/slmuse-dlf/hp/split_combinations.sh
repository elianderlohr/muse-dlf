#!/bin/bash

# Number of splits/jobs
num_splits=5

# Split the combinations file
split -l $(( $(wc -l < combinations.txt) / num_splits )) combinations.txt combination_
