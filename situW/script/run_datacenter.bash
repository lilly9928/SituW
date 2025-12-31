#!/bin/bash
set -e

bash SituW/situW/script/logiqa/two_shot/datacenter_logiqa_two_llama_8b.bash
bash SituW/situW/script/logiqa/two_shot/datacenter_logiqa_two_mistral_7b.bash

bash SituW/situW/script/folio/two_shot/datacenter_folio_two_llama_8b.bash
bash SituW/situW/script/folio/two_shot/datacenter_folio_two_mistral_7b.bash

bash SituW/situW/script/proofwriter/two_shot/datacenter_proofwriter_two_mistral_7b.bash
