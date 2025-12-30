#!/bin/bash
set -e

bash /data3/KJE/code/SituW/situW/script/proofwriter/proofwriter_base_llama_70b_33.bash
bash /data3/KJE/code/SituW/situW/script/proofwriter/proofwriter_base_llama_70b.bash
bash /data3/KJE/code/SituW/situW/script/proofwriter/proofwriter_base_mistral_7b.bash
bash /data3/KJE/code/SituW/situW/script/proofwriter/proofwriter_base_mistral_large.bash

bash /data3/KJE/code/SituW/situW/script/logiqa/logiqa_base_llama_70b_33.bash
bash /data3/KJE/code/SituW/situW/script/logiqa/logiqa_base_llama_70b.bash
bash /data3/KJE/code/SituW/situW/script/logiqa/logiqa_base_mistral_large.bash

bash /data3/KJE/code/SituW/situW/script/proofwriter/proofwriter_base_qwen_72b.bash

bash /data3/KJE/code/SituW/situW/script/logiqa/logiqa_base_qwen_72b.bash

bash /data3/KJE/code/SituW/situW/script/folio/folio_base_qwen_72b.bash

bash /data3/KJE/code/SituW/situW/script/prontoqa/prontoqa_base_qwen_72b.bash