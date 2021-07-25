#!/bin/bash

# Run Multi-layer neural network on all configuration files
for file in ./DL_configs/*; do 
    if [ -f "$file" ]; then
        echo "************************* Neural Network on $file *************************" 
        neural-network --dl-config "$file" 
    fi
done

# Run 1D CNN on GLCM configuration files
for file in ./DL_configs/GLCM*; do 
    if [ -f "$file" ]; then
        echo "************************* 1D CNN on $file *************************" 
        CNN --dl-config "$file" 
    fi
done