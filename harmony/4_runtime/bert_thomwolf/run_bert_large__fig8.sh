#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_DIR="../../results"
MODEL="bert_large"
CONFIG="../../../model_lib/bert_configs/bert-large-uncased.json"
OUT_DIR="./logs/${MODEL}"
mkdir -p ${OUT_DIR}
# --------------------------------------------
for SCHEDULED in "D600_vPP_N4_Ufwd30_Ubwd15_P4" "D600_vPP_N4_Ufwd20_Ubwd10_P7" 
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
# |& tee ${OUT_DIR}/${SCHEDULED}__fig8.txt
done
