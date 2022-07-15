#!/bin/bash
MODEL="gpt2_xl"
CONFIG="../../../model_lib/gpt2_configs/gpt2-xl-config.json"
OUT_DIR="./logs/${MODEL}"
mkdir -p ${OUT_DIR}
# --------------------------------------------
for SCHEDULED in "D32_vDP_N2_Top1" "D32_vPP_N2_Top1" 
do
echo "Clean Python Processes"
sleep 3s && pkill -9 python3 && pkill -9 python && sleep 3s

echo "${SCHEDULED}"
python3 main.py \
--gpt2_config_path ${CONFIG} \
--gpt2_model "" \
--synthetic_data \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_iter 4 \
--output_dir ${OUT_DIR} \
# |& tee ${OUT_DIR}/${SCHEDULED}.txt
done