#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_DIR="../../results"
MODEL="bert_96"
CONFIG="../../../model_lib/bert_configs/bert-large-uncased-96.json"
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
--bert_config_path ${CONFIG} \
--bert_model "" \
--synthetic_data \
--module_dir ${MODEL_DIR} \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_iter 4 \
--output_dir ${OUT_DIR} \
# |& tee ${OUT_DIR}/${SCHEDULED}__fig10.txt
done
