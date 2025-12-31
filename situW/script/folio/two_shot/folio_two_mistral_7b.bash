#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3,4,5

python /data3/KJE/code/SituW/situW/src/main_seoha.py \
  --data_path /data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/data/ThinkAgent/FOLIO/folio_val.json \
  --save_path /data3/KJE/code/SituW/situW/output \
  --model_name mistral_7b_instruct \
  --mode folio_val \
  --shot two \
  --max_new_tokens 16
