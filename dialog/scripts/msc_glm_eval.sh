#! /usr/bin/env bash
DATASET_PATH=/data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/data/Dialog/MSC

MODEL_PATH=THUDM/chatglm3-6b
SUMMARIZER=/data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/base/LD-Agent/model/summarizer
EXTRACTOR=/data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/base/LD-Agent/model/extractor
GENERATOR=/data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/base/LD-Agent/model/summarizer

DATESTR=`date +%Y%m%d-%H%M%S`
LOG_NAME=MSC_EVAL-${DATESTR}.log


python -u main.py --dataset msc --data_path ${DATASET_PATH} --data_name sequential_test.json \
        --client chatglm --model ${MODEL_PATH} \
        --summary_model ${SUMMARIZER} --persona_model ${EXTRACTOR} --generation_model ${GENERATOR} \
        --usr_name SPEAKER_1 --agent_name SPEAKER_2 \
        --test_num 501 --gpus 0

