#!/bin/bash

# Base directory containing all the experiment folders
BASE_DIR="."

# Find all directories containing best_model.pt
find "$BASE_DIR" -name "best_model.pt" | while read -r model_path; do
    # Get the model folder (directory containing best_model.pt)
    model_folder=$(dirname "$model_path")
    
    # Skip if done.txt doesn't exist (training not completed)
    if [ ! -f "$model_folder/done.txt" ]; then
        echo "Skipping $model_folder (training not completed)"
        continue
    fi
    
    # Extract param hash folder name
    param_hash=$(basename "$model_folder")
    
    # Extract experiment name (parent directory of param hash folder)
    experiment_dir=$(dirname "$model_folder")
    experiment_name=$(basename "$experiment_dir")
    
    # Skip if experiment name contains "RB"
    if [[ "$experiment_name" == *RB* ]]; then
        echo "Skipping $experiment_name (contains RB)"
        continue
    fi
    
    # Extract problem type from the experiment name
    if [[ "$experiment_name" == max_cut* ]]; then
        problem_type="max_cut"
    elif [[ "$experiment_name" == vertex_cover* ]]; then
        problem_type="vertex_cover"
    elif [[ "$experiment_name" == sat* ]]; then
        problem_type="sat"
    else
        problem_type="unknown"
    fi
    
    # Create descriptive test prefix with experiment name
    test_prefix="${experiment_name}_${param_hash:0:8}_Eval"
    
    echo "===================================="
    echo "Running test on: $model_folder"
    echo "Experiment: $experiment_name"
    echo "Param hash: $param_hash"
    echo "Problem type: $problem_type"
    echo "Test prefix: $test_prefix"
    
    # Run the test script
    python test.py --model_folder="$model_folder" --model_file="best_model.pt" --test_prefix="$test_prefix" --problem_type="$problem_type"
    
    echo "Test completed for $experiment_name ($param_hash)"
    echo "===================================="
done