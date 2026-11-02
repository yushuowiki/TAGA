#!/bin/bash

# Set variables similar to pretrain.sh
OUTPUT_DIR="./cls_clintox_viz"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the attention visualization script with parameters
CUDA_VISIBLE_DEVICES=1 python ./visual_atten/mol2token_atten_map.py \
    --model_path "./checkpoints/clintox/adaptive_entropy/"\
    --model_base "./Projects/Mol2Token/llm_ckpt"\
    --use_task "cls"\
    --use_dataset "clintox"\
    --batch_size 2\

echo "Attention visualization completed. Results saved to $OUTPUT_DIR"