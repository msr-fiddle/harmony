#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_DIR="../../results"
MODEL="bert_large"
SEQLEN=128
SUFFIX="_seqlen${SEQLEN}"
CONFIG="../../../model_lib/bert_configs/bert-large-uncased.json"
INIT_MODEL="/workspace/.pretrained_models/BERT-Large-Uncased_Seed111"
SEED=3
# -------------------- Train ------------------------
for SCHEDULED in "D32_vDP_N1_Ufwd8_Ubwd8_P4" "D32_vPP_N4_Ufwd8_Ubwd8_P4" "D32_vDP_N4_Ufwd8_Ubwd8_P4" # "D32_vDP_N1_Ufwd8_Ubwd8_P4" "D32_vDP_N2_Ufwd8_Ubwd8_P4" "D32_vPP_N2_Ufwd8_Ubwd8_P4"
do
echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

OUT_DIR="./logs/finetune_${MODEL}${SUFFIX}/${SCHEDULED}"
mkdir -p ${OUT_DIR}
echo "${SCHEDULED}"
numactl --cpunodebind=0 --membind=0 \
python3 main.py \
--bert_data_dir "/data/glue/MRPC" \
--bert_seq_length ${SEQLEN} \
--bert_config_path ${CONFIG} \
--bert_model ${INIT_MODEL} \
--module_dir ${MODEL_DIR} \
--module_name ${MODEL} \
--suffix ${SUFFIX} \
--schedule_fname ${SCHEDULED} \
--num_epochs 3 \
--seed ${SEED} \
--output_dir ${OUT_DIR} \
--save_final_model \
# |& tee ${OUT_DIR}/log.txt
done


# NOTE:
# -. Initial models can be downloaded here (https://1drv.ms/u/s!ApfNYtXZyxcLcKg9KbddiUGFp9E?e=WqcHe2).
# 
# -. In case of hardware randomness, use following flags (at cost of speed):
#       export CUDA_LAUNCH_BLOCKING=1 # single-GPU & Harmony DP only
#       --seed_cudnn
#       --no_all_prefetch_offload
#
# -. Losses need to be moving-averaged.
