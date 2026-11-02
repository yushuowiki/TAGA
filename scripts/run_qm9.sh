#!/bin/bash
# Train and evaluate on QM9 regression task
export WANDB_MODE=offline

echo "=============================================="
echo "QM9 Regression Training (adaptive_entropy)"
echo "=============================================="

# Training
CUDA_VISIBLE_DEVICES=0 python train/train_mem.py \
    --model_name_or_path ./llm_ckpt \
    --version v1 \
    --cache_dir ../../checkpoint \
    --pretrained_embedding_type 20_m_4096 \
    --tune_mm_mlp_adapter True \
    --tune_graph_tower True \
    --mm_use_graph_start_end False \
    --mm_use_graph_patch_token False \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 2500 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none \
    --mm_projector_type linear \
    --fp_fusion_mode adaptive_entropy \
    --use_task rg \
    --use_dataset qm9 \
    --graph_pretrained ./model/graph_encoder/models/graphconv_bio_epoch_10.pt \
    --output_dir ./checkpoints/qm9/adaptive_entropy

if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "QM9 Regression Inference"
echo "=============================================="

mkdir -p ./results/qm9
CUDA_VISIBLE_DEVICES=0 python eval/qm9_regression.py \
    --model-path ./checkpoints/qm9/adaptive_entropy \
    --model-base ./llm_ckpt \
    --test-file ./dataset/qm9/qm9_test.jsonl \
    --answers-file ./results/qm9/answers.jsonl \
    --batch-size 8

echo "=============================================="
echo "QM9 Done!"
echo "=============================================="
