#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2

python /data3/KJE/code/SituW/situW/src/main_seoha.py \
  --data_path /data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/data/ThinkAgent/FOLIO/folio_val.json \
  --save_path /data3/KJE/code/SituW/situW/output \
  --model_name qwen_72b_instruct \
  --mode folio_val \
  --max_new_tokens 16
