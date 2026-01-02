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

# ----- Config -----
model_name="gpt-5-nano"

data_path="/data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/data/ThinkAgent/LogiQA2.0"
dataset_name="LogiQA2.0"          # FOLIO / LogicalDeduction / AR-LSAT / LogiQA2.0
split="train"

save_path="/data3/KJE/code/SituW/situW/output/distill_memo"

# prompts
memory_step_prompt_path="/data3/KJE/code/SituW/situW/utils/prompt/memory_step.txt"
final_reason_prompt_path="/data3/KJE/code/SituW/situW/utils/prompt/final_reasoning_from_memory_logiqa.txt"

# batch & controls
batch_size=10
max_steps=0              # 0 = no limit
save_all="--save_all"    # optional: save all outputs json too (remove if not needed)

echo "dataset_name=$dataset_name"
echo "split=$split"
echo "model_name=$model_name"
echo "data_path=$data_path"
echo "save_path=$save_path"
echo "memory_step_prompt_path=$memory_step_prompt_path"
echo "final_reason_prompt_path=$final_reason_prompt_path"
# -------------------

python src/situW_ver2.py \
  --api_key "$api_key" \
  --model_name "$model_name" \
  --data_path "$data_path" \
  --dataset_name "$dataset_name" \
  --split "$split" \
  --save_path "$save_path" \
  --memory_step_prompt_path "$memory_step_prompt_path" \
  --final_reason_prompt_path "$final_reason_prompt_path" \
  --batch_size "$batch_size" \
  --max_steps "$max_steps" \
  $save_all
