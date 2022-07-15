#!/bin/bash
MODEL="gpt2_medium"
CONFIG="../../../model_lib/gpt2_configs/gpt2-medium-config.json"
OUT_DIR="./logs/${MODEL}"
mkdir -p ${OUT_DIR}

# ----------------- Synthetic Data + NVPROF --------------------------
for SCHEDULED in "D32_vDP_N2_Ufwd2_Ubwd2_P7" "D32_vPP_N2_Ufwd2_Ubwd2_P7" 
do
echo "Clean Python Processes"
sleep 3s && pkill -9 python3 && pkill -9 python && sleep 3s

echo "${SCHEDULED}"
nvprof --profile-child-processes \
-fo "${OUT_DIR}/${SCHEDULED}_pid%p.nvvp" \
--track-memory-allocations on \
--unified-memory-profiling off \
--profile-from-start off \
--print-summary-per-gpu \
python3 main.py \
--gpt2_config_path ${CONFIG} \
--gpt2_model "" \
--synthetic_data \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_iter 4 \
--output_dir ${OUT_DIR} \
--nvprof \
--nvprof_iter "last" \
# |& tee ${OUT_DIR}/${SCHEDULED}__syn_nvprof.txt
done

# -------------------- Real Data + Fullspeed ------------------------
for SCHEDULED in "D32_vDP_N2_Ufwd2_Ubwd2_P7" "D32_vPP_N2_Ufwd2_Ubwd2_P7"
do

echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

echo "${SCHEDULED}"
python3 -OO main.py \
--gpt2_train_file "/home/wikitext-2-tokens/wiki.train.tokens" \
--gpt2_config_path ${CONFIG} \
--gpt2_model "" \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_iter 4 \
--output_dir ${OUT_DIR} \
# |& tee ${OUT_DIR}/${SCHEDULED}__real_fullspd.txt
done

# -------------------- Real Data + Train ------------------------
INIT_MODEL="/workspace/.pretrained_models/GPT2-Medium"
SCHEDULED="D256_vPP_N4_Ufwd2_Ubwd2_P7"
OUT_DIR="./logs/finetune_${MODEL}_wiki103t/${SCHEDULED}"

echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

mkdir -p ${OUT_DIR}
echo "${OUT_DIR}"
python3 main.py \
--gpt2_train_file "/home/wikitext-103-tokens/wiki.train.tokens" \
--gpt2_config_path ${CONFIG} \
--gpt2_model ${INIT_MODEL} \
--learning_rate 5e-5 \
--warmup_steps 0 \
--adam_epsilon 1e-8 \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_epochs 4 \
--seed 42 \
--output_dir ${OUT_DIR} \
--save_final_model \
# |& tee ${OUT_DIR}/log.txt