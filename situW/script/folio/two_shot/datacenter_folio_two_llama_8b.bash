#!/bin/bash
set -e

python SituW/situW/src/main_seoha.py \
  --data_path data/ThinkAgent/FOLIO/folio_val.json \
  --save_path SituW/situW/output \
  --model_name llama_31_8b_instruct \
  --mode folio_val \
  --shot two \
  --batch_size 16 \
  --max_new_tokens 16 \
  --cache_dir /scratch/sclab_kje/hg_weight/hub
