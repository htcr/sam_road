#!/bin/bash

# List of directories to process, add more as needed
output_dirs=(
    "spacenet_toponet_8x8"
    "spacenet_toponet_4x4"
)

# Base directory where the output directories are located
base_dir="save"

# Loop through each output_dir in the list
for output_dir in "${output_dirs[@]}"; do
    # Construct the full path to the output directory
    full_path="$base_dir/$output_dir"
    
    # Run the apls.bash script with the full path to the output directory
    bash apls.bash "$full_path"
    
    # Run the topo.bash script with the full path to the output directory
    bash topo.bash "$full_path"
done