#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

echo "Clean Python Processes"
pkill -9 python3 & pkill -9 python & sleep 1s

# -------------- BERT Large (128) ---------------
MODEL_DIR="../../results"
MODEL="bert_large"
CONFIG_PATH="../../../model_lib/bert_configs/bert-large-uncased.json"
SEQ_LEN=128
SUFFIX="_seqlen128"

for PROBE_WHAT in "FWD" "BWD" 
do
echo "Probe ${PROBE_WHAT}"
numactl --cpunodebind=0 --membind=0 \
python3 main.py \
 --bert_seq_length ${SEQ_LEN} \
 --bert_config_path ${CONFIG_PATH} \
 --module_dir ${MODEL_DIR} \
 --module_name ${MODEL} \
 --mode "probe" \
 --probe_what ${PROBE_WHAT} \
 --outname_suffix ${SUFFIX}
done

echo "Profile normally"
numactl --cpunodebind=0 --membind=0 \
python3 main.py \
 --bert_seq_length ${SEQ_LEN} \
 --bert_config_path ${CONFIG_PATH} \
 --module_dir ${MODEL_DIR} \
 --module_name ${MODEL} \
 --mode "normal" \
 --ubatchsize_step 10 \
 --num_trials 4 \
 --outname_suffix ${SUFFIX} \
# --fwd_umax 800 \
# --bwd_umax 400 \