#!/bin/bash -i


api_key=""
model_name="gpt-5"
data_path="/data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/data/ThinkAgent/FOLIO"
dataset_name="FOLIO" #FOLIO / LogicalDeduction / AR-LSAT /logiqa
mode="Ours_CoT"
split="train"
# echo $dataset_name
# echo $model_name
# echo "EVAL"
# python3 evaluate.py --dataset_name $dataset_name --model_name $model_name --split dev


echo $dataset_name
echo $model_name

# python3 symbcot.py \
# --api_key $api_key --model_name $model_name  --data_path $data_path --dataset_name $dataset_name --split dev

python src/extract_situation.py \
--api_key $api_key --model_name $model_name  --data_path $data_path --dataset_name $dataset_name --split $split --mode $mode