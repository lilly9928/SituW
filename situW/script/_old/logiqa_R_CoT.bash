#!/bin/bash -i


api_key=""
model_name="gpt-4o-mini"
data_path="/data3/KJE/code/WIL_DeepLearningProject_2/NS_Parser/SITUM_EMNLP/data/LogiQA2.0/logiqa2nli/DATA/QA2NLI"
dataset_name="logiqa" #FOLIO / LogicalDeduction / AR-LSAT /logiqa
mode="Ours_CoT"
split="test"
# echo $dataset_name
# echo $model_name
# echo "EVAL"
# python3 evaluate.py --dataset_name $dataset_name --model_name $model_name --split dev


echo $dataset_name
echo $model_name

# python3 symbcot.py \
# --api_key $api_key --model_name $model_name  --data_path $data_path --dataset_name $dataset_name --split dev

python src/main.py \
--api_key $api_key --model_name $model_name  --data_path $data_path --dataset_name $dataset_name --split $split --mode $mode
