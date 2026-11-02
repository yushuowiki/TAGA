#!/bin/bash
# Run training and inference for all cls tasks
export WANDB_MODE=offline

# dataset list
DATASETS="bace bbbp hiv sider clintox tox21 muv cyp450"

# common args
COMMON_ARGS="--model_name_or_path ./llm_ckpt \
    --version v1 \
    --cache_dir ../../checkpoint \
    --pretrained_embedding_type 20_m_4096 \
    --tune_mm_mlp_adapter True \
    --tune_graph_tower True \
    --mm_use_graph_start_end False \
    --mm_use_graph_patch_token False \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy epoch \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none \
    --mm_projector_type linear \
    --fp_fusion_mode adaptive_entropy \
    --use_task cls \
    --graph_pretrained ./model/graph_encoder/models/graphconv_bio_epoch_10.pt"

echo "=============================================="
echo "Begin to run all cls task (adaptive_entropy)"
echo "Dataset: $DATASETS"
echo "=============================================="

for DATASET in $DATASETS; do
    echo ""
    echo "=============================================="
    echo "[$DATASET] start training..."
    echo "=============================================="

    # training
    CUDA_VISIBLE_DEVICES=0 python train/train_mem.py \
        $COMMON_ARGS \
        --output_dir ./checkpoints/${DATASET}/adaptive_entropy \
        --use_dataset $DATASET

    if [ $? -ne 0 ]; then
        echo "[$DATASET] failed!"
        continue
    fi

    echo ""
    echo "[$DATASET] begin to evaluate..."

    # evaluate
    mkdir -p ./results/${DATASET}
    CUDA_VISIBLE_DEVICES=0 python eval/moleculenet_classification.py \
        --model-path ./checkpoints/${DATASET}/adaptive_entropy \
        --model-base ./llm_ckpt \
        --test-file ./dataset/${DATASET}/${DATASET}_test.jsonl \
        --answers-file ./results/${DATASET}/answers.jsonl

    echo "[$DATASET] done!"
    echo ""
done

echo "=============================================="
echo "Completed!"
echo "=============================================="
