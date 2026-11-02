#!/bin/bash

# Set variables similar to pretrain.sh
OUTPUT_DIR="./rg_atten_pad_dataset"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the attention visualization script with parameters
CUDA_VISIBLE_DEVICES=2 python ./visual_atten/mol2token_atten_map.py \
    --model_path "./Mol2Token/checkpoints/qm9_3"\
    --model_base "./Projects/Mol2Token/llm_ckpt"\
    --use_task "rg"\
    --use_dataset "qm9"\
    --batch_size 4\

echo "Attention visualization completed. Results saved to $OUTPUT_DIR"