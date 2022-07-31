#!/bin/bash
MODEL="bert_large"
SEQLEN=128
SUFFIX="_seqlen${SEQLEN}"
CONFIG="../../../model_lib/bert_configs/bert-large-uncased.json"
OUT_DIR="./logs/${MODEL}${SUFFIX}"
mkdir -p ${OUT_DIR}

# ----------------- Synthetic Data + NVPROF --------------------------
for SCHEDULED in "D32_vDP_N2_Ufwd8_Ubwd8_P4" "D32_vPP_N2_Ufwd8_Ubwd8_P4" 
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
--bert_seq_length ${SEQLEN} \
--bert_config_path ${CONFIG} \
--bert_model "" \
--synthetic_data \
--module_name ${MODEL} \
--suffix ${SUFFIX} \
--schedule_fname ${SCHEDULED} \
--num_iter 4 \
--output_dir ${OUT_DIR} \
--nvprof \
--nvprof_iter "last" \
# |& tee ${OUT_DIR}/${SCHEDULED}__syn_nvprof.txt
done

# -------------------- Real Data + Fullspeed ------------------------
for SCHEDULED in "D32_vDP_N2_Ufwd8_Ubwd8_P4" "D32_vPP_N2_Ufwd8_Ubwd8_P4"
do

echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

echo "${SCHEDULED}"
python3 -OO main.py \
--bert_data_dir "/data/glue/MRPC" \
--bert_seq_length ${SEQLEN} \
--bert_config_path ${CONFIG} \
--bert_model "" \
--module_name ${MODEL} \
--suffix ${SUFFIX} \
--schedule_fname ${SCHEDULED} \
--num_iter 4 \
--output_dir ${OUT_DIR} \
# |& tee ${OUT_DIR}/${SCHEDULED}__real_fullspd.txt
done
