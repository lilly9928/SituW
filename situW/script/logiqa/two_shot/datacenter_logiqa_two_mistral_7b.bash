#!/bin/bash
set -e

python SituW/situW/src/main_seoha.py \
  --data_path data/ThinkAgent/LogiQA2.0/logiqa_val.json \
  --save_path SituW/situW/output \
  --model_name mistral_7b_instruct \
  --mode logiqa_val \
  --shot two \
  --batch_size 16 \
  --max_new_tokens 16 \
  --cache_dir /scratch/sclab_kje/hg_weight/hub
