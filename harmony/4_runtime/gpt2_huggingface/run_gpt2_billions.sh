#!/bin/bash
for BILLION in 10 # 20 30 40
do

MODEL="gpt2_${BILLION}b"
CONFIG="../../../model_lib/gpt2_configs/gpt2-${BILLION}-billion.json"
OUT_DIR="./logs/${MODEL}"
mkdir -p ${OUT_DIR}
# --------------------------------------------
for SCHEDULED in "D32_vPP_N8_Ufwd1_Ubwd1_P2" "D32_vDP_N8_Ufwd1_Ubwd1_P2" 
do

echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 10s

echo "${MODEL} (${SCHEDULED})"
python3 main.py \
--gpt2_config_path ${CONFIG} \
--gpt2_model "" \
--synthetic_data \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_iters 3 \
--no_pin_model \
--no_pin_x \
--no_pin_data \
--output_dir ${OUT_DIR} \
# |& tee ${OUT_DIR}/${SCHEDULED}.txt
done

done

