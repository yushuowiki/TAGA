#!/bin/bash
SEED=42
ALL_DATASETS="bbbp bace hiv sider clintox tox21 muv cyp450"

echo "Dataset,Baseline_AUC,Entropy_AUC,Delta,Delta_Pct"

for DATASET in $ALL_DATASETS; do
    mkdir -p ./results/${DATASET}

    # Find baseline checkpoint
    if [ -f "./checkpoints/${DATASET}/baseline_none/mm_projector.bin" ]; then
        BASELINE_CKPT="./checkpoints/${DATASET}/baseline_none"
    elif [ -f "./checkpoints/${DATASET}/baseline/mm_projector.bin" ]; then
        BASELINE_CKPT="./checkpoints/${DATASET}/baseline"
    else
        BASELINE_CKPT=""
    fi

    ENTROPY_CKPT="./checkpoints/${DATASET}/adaptive_entropy"

    # Evaluate baseline
    if [ -n "$BASELINE_CKPT" ] && [ -f "${BASELINE_CKPT}/mm_projector.bin" ]; then
        BASELINE_AUC=$(CUDA_VISIBLE_DEVICES=0 python eval/moleculenet_classification.py \
            --model-path ${BASELINE_CKPT} \
            --model-base ./llm_ckpt \
            --test-file ./dataset/${DATASET}/${DATASET}_test.jsonl \
            --answers-file ./results/${DATASET}/eval_baseline.jsonl \
            --seed ${SEED} 2>&1 | grep "ROC-AUC (prob-based)" | sed 's/.*: //')
    else
        BASELINE_AUC="N/A"
    fi

    # Evaluate adaptive_entropy
    if [ -f "${ENTROPY_CKPT}/mm_projector.bin" ]; then
        ENTROPY_AUC=$(CUDA_VISIBLE_DEVICES=0 python eval/moleculenet_classification.py \
            --model-path ${ENTROPY_CKPT} \
            --model-base ./llm_ckpt \
            --test-file ./dataset/${DATASET}/${DATASET}_test.jsonl \
            --answers-file ./results/${DATASET}/eval_entropy.jsonl \
            --seed ${SEED} 2>&1 | grep "ROC-AUC (prob-based)" | sed 's/.*: //')
    else
        ENTROPY_AUC="N/A"
    fi

    # Calculate delta
    if [ "$BASELINE_AUC" != "N/A" ] && [ -n "$BASELINE_AUC" ] && [ "$ENTROPY_AUC" != "N/A" ] && [ -n "$ENTROPY_AUC" ]; then
        DELTA=$(python3 -c "b=$BASELINE_AUC; e=$ENTROPY_AUC; print(f'{e-b:.4f}')")
        DELTA_PCT=$(python3 -c "b=$BASELINE_AUC; e=$ENTROPY_AUC; print(f'{(e-b)/b*100:.2f}')")
    else
        DELTA="N/A"
        DELTA_PCT="N/A"
    fi

    echo "${DATASET},${BASELINE_AUC},${ENTROPY_AUC},${DELTA},${DELTA_PCT}%"
done
