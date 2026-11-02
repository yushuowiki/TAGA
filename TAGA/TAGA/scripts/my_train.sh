#!/bin/bash

max_len=4096

model=${1:-"vicuna"}
task=${2:-"cls"}
dataset=${3-"bace-bbbp"}
bs=${4:-16}
tower=${5:-"neuralfp"}
pre_emb_shape=${6:-"20_m_4096"}


if [ ${model} = "vicuna" ]; then
  projector_type="linear"
  prefix=llaga-vicuna-7b-${tower}-${projector_type}-projector
  model_base=./llm_ckpt
  mode="v1"
elif [ ${model} = "vicuna_2layer" ]; then
  projector_type="2-layer-mlp"
  prefix=llaga-vicuna-7b-${tower}-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "llama" ]; then
  projector_type="linear"
  prefix=llaga-llama-2-7b-hf-${tower}-${projector_type}--projector
  model_base=meta-llama/Llama-2-7b-hf
  mode="llaga_llama_2"
elif [ ${model} = "opt_2.7b" ]; then
  projector_type="linear"
  prefix=llaga-opt-2.7b-${tower}-${projector_type}-projector
  model_base=facebook/opt-2.7b
  max_len=1536
  mode="v1"
fi

echo "PREFIX:  ${prefix}"

export WANDB_MODE=offline

python train/train_mem.py \
--model_name_or_path ${model_base} \
--version ${mode} \
--cache_dir  ../../checkpoint \
--pretrained_embedding_type ${pre_emb_shape} \
--tune_mm_mlp_adapter True \
--tune_graph_tower True \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--bf16 True \
--output_dir  ./checkpoints/${dataset}/${prefix}_${task} \
--num_train_epochs 1 \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length ${max_len} \
--gradient_checkpointing True \
--lazy_preprocess True \
--report_to none \
--mm_projector_type ${projector_type} \
--use_task ${task} \
--use_dataset ${dataset} \
--graph_pretrained ./model/graph_encoder/models/graphconv_bio_epoch_10.pt
