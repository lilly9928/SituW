#!/bin/bash -i
set -euo pipefail

# ----- Key loading -----
KEY_FILE="$(dirname "$0")/openai.key"
# KEY_FILE="$HOME/.openai.key"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "ERROR: key file not found: $KEY_FILE" >&2
  exit 1
fi

api_key="$(tr -d ' \t\r\n' < "$KEY_FILE")"

if [[ -z "$api_key" ]]; then
  echo "ERROR: api_key is empty (check $KEY_FILE)" >&2
  exit 1
fi
# -----------------------

model_name="gpt-5-nano"
data_path="/data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/data/ThinkAgent/FOLIO"
dataset_name="FOLIO"   # FOLIO / LogicalDeduction / AR-LSAT / logiqa
split="train"

stage1_file="/data3/KJE/code/SituW/situW/output/situation_memory_260101/Ours_CoT_FOLIO_train_gpt-5-mini_event_situation_models.json"
save_path="/data3/KJE/code/SituW/situW/output/stage2_predictions"

use_llm_conclusion="--use_llm_conclusion"
use_relevance="--use_relevance"

max_chain_iter=10
max_ground_constants=50
topk_focus=10

binary=""  # e.g., set to "--binary" if you want 2-class output
binary_positive="entailed"  # entailed | not_contradicted

echo "$dataset_name"
echo "$model_name"
echo "stage1_file=$stage1_file"

python src/stage2_reasoner.py \
  --api_key "$api_key" --model_name "$model_name" \
  --data_path "$data_path" --dataset_name "$dataset_name" \
  --split "$split" \
  --stage1_files "$stage1_file" \
  --save_path "$save_path" \
  $use_llm_conclusion \
  $use_relevance \
  --max_chain_iter "$max_chain_iter" \
  --max_ground_constants "$max_ground_constants" \
  --topk_focus "$topk_focus" \
  $binary --binary_positive "$binary_positive"
