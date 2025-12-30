
set -e

export CUDA_VISIBLE_DEVICES=0,1,2

python test_llm.py \
  --model_name llama_33_70b_instruct\
