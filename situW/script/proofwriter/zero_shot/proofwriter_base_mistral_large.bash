#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3,4,5

python /data3/KJE/code/SituW/situW/src/main_seoha.py \
  --data_path /data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/data/ThinkAgent/ProofWriter/proofwriter_val.json \
  --save_path /data3/KJE/code/SituW/situW/output \
  --model_name mistral_large_instruct \
  --mode proofwriter_val \
  --batch_size 32 \
  --max_new_tokens 16
