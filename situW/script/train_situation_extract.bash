#!/bin/bash -i
set -euo pipefail

# ----- Key loading -----
KEY_FILE="$(dirname "$0")/openai.key"   
# KEY_FILE="$HOME/.openai.key"          

if [[ ! -f "$KEY_FILE" ]]; then
  echo "ERROR: key file not found: $KEY_FILE" >&2
  exit 1
fi

# 파일 내용에서 공백/개행 제거해서 api_key에 저장
api_key="$(tr -d ' \t\r\n' < "$KEY_FILE")"

if [[ -z "$api_key" ]]; then
  echo "ERROR: api_key is empty (check $KEY_FILE)" >&2
  exit 1
fi
# -----------------------

model_name="gpt-5"
data_path="/data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/data/ThinkAgent/FOLIO"
dataset_name="FOLIO" #FOLIO / LogicalDeduction / AR-LSAT /logiqa
mode="Ours_CoT"
split="train"

echo "$dataset_name"
echo "$model_name"

python src/extract_situation.py \
  --api_key "$api_key" --model_name "$model_name" \
  --data_path "$data_path" --dataset_name "$dataset_name" \
  --split "$split" --mode "$mode"
