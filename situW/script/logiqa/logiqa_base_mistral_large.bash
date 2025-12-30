#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2

python /data3/KJE/code/SituW/situW/src/main_seoha.py \
  --data_path /data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/data/ThinkAgent/LogiQA2.0/logiqa_val.json \
  --save_path /data3/KJE/code/SituW/situW/output \
  --model_name mistral_large_instruct \
  --mode logiqa_val \
  --batch_size 16 \
  --max_new_tokens 16
