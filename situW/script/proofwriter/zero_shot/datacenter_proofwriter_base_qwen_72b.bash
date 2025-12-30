#!/bin/bash
set -e

python SituW/situW/src/main_seoha_datacenter.py \
  --data_path data/ThinkAgent/ProofWriter/proofwriter_val.json \
  --save_path SituW/situW/output \
  --model_name qwen_72b_instruct \
  --mode proofwriter_val \
  --batch_size 64 \
  --max_new_tokens 128 \
  --cache_dir /data3/hg_weight/hg_weight