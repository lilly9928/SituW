#!/bin/bash
set -e

python SituW/situW/src/main_seoha.py \
  --data_path data/ThinkAgent/ProofWriter/proofwriter_val.json \
  --save_path SituW/situW/output \
  --model_name qwen_72b_instruct \
  --mode proofwriter_val \
  --batch_size 64 \
  --max_new_tokens 128 \
  --cache_dir /home/sclab_kje/hg_weight