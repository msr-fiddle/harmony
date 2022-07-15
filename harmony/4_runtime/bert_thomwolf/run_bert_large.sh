#!/bin/bash
MODEL_DIR="../../results"
MODEL="bert_large"
CONFIG="../../../model_lib/bert_configs/bert-large-uncased.json"
OUT_DIR="./logs/${MODEL}"
mkdir -p ${OUT_DIR}
# --------------------------------------------
for SCHEDULED in "D32_vDP_N2_Top1" "D32_vPP_N2_Top1" 
do
echo "Clean Python Processes"
sleep 3s && pkill -9 python3 && pkill -9 python && sleep 3s

echo "${SCHEDULED}"
python3 main.py \
--bert_config_path ${CONFIG} \
--bert_model "" \
--synthetic_data \
--module_dir ${MODEL_DIR} \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_iter 4 \
--output_dir ${OUT_DIR}
done