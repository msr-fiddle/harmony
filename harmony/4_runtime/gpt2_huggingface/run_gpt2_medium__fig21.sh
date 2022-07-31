#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_DIR="../../results"
MODEL="gpt2_medium"
CONFIG="../../../model_lib/gpt2_configs/gpt2-medium-config.json"
INIT_MODEL="/workspace/.pretrained_models/GPT2-Medium_Seed42"
SEED=42
# -------------------- Train ------------------------
for SCHEDULED in "D256_vDP_N1_Ufwd2_Ubwd2_P7" "D256_vPP_N4_Ufwd2_Ubwd2_P7" "D256_vDP_N4_Ufwd2_Ubwd2_P7" # "D32_vDP_N2_Ufwd2_Ubwd2_P7" "D32_vPP_N2_Ufwd2_Ubwd2_P7"
do
echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

OUT_DIR="./logs/finetune_${MODEL}/${SCHEDULED}"
mkdir -p ${OUT_DIR}
echo "${SCHEDULED}"
numactl --cpunodebind=0 --membind=0 \
python3 main.py \
--gpt2_train_file "/home/wikitext-103-tokens/wiki.train.tokens" \
--gpt2_config_path ${CONFIG} \
--gpt2_model ${INIT_MODEL} \
--learning_rate 5e-5 \
--warmup_steps 0 \
--adam_epsilon 1e-8 \
--module_dir ${MODEL_DIR} \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_epochs 4 \
--seed ${SEED} \
--output_dir ${OUT_DIR} \
--save_final_model \
# |& tee ${OUT_DIR}/log.txt
done


# NOTE:
# In case of hardware randomness, use following flags (at cost of speed):
#   export CUDA_LAUNCH_BLOCKING=1 # single-GPU & Harmony DP only
#   --seed_cudnn
#   --no_all_prefetch_offload
