#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_DIR="../../results"
MODEL="gpt2_xl"
CONFIG="../../../model_lib/gpt2_configs/gpt2-xl-config.json"
OUT_DIR="./logs/${MODEL}"
mkdir -p ${OUT_DIR}
# --------------------------------------------
for SCHEDULED in "D16_vDP_N4_Top1" "D16_vPP_N4_Top1" "D64_vDP_N4_Top1" "D64_vPP_N4_Top1" "D256_vDP_N4_Top1" "D256_vPP_N4_Top1"
do
echo "Clean Python Processes"
sleep 3s && pkill -9 python3 && pkill -9 python && sleep 3s

echo "${SCHEDULED}"
numactl --cpunodebind=0 --membind=0 \
python3 -OO main.py \
--gpt2_config_path ${CONFIG} \
--gpt2_model "" \
--synthetic_data \
--module_dir ${MODEL_DIR} \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_iter 4 \
--output_dir ${OUT_DIR} \
# |& tee ${OUT_DIR}/${SCHEDULED}__fig10_fig12.txt
done