#!/bin/bash
set -e

python SituW/situW/src/main_seoha.py \
  --data_path data/ThinkAgent/ProofWriter/proofwriter_val.json \
  --save_path SituW/situW/output \
  --model_name mistral_7b_instruct \
  --mode proofwriter_val \
  --shot two \
  --batch_size 64 \
  --max_new_tokens 16 \
  --cache_dir /scratch/sclab_kje/hg_weight/hub
