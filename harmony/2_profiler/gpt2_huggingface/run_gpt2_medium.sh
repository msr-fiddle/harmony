#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

echo "Clean Python Processes"
pkill -9 python3 & pkill -9 python & sleep 1s

# --------------- GPT2 Medium ---------------
MODEL_DIR="../../results"
MODEL="gpt2_medium"
CONFIG_PATH="../../../model_lib/gpt2_configs/gpt2-medium-config.json"

for PROBE_WHAT in "FWD" "BWD" 
do
echo "Probe ${PROBE_WHAT}"
numactl --cpunodebind=0 --membind=0 \
python3 main.py \
 --gpt2_config_path ${CONFIG_PATH} \
 --module_dir ${MODEL_DIR} \
 --module_name ${MODEL} \
 --mode "probe" \
 --probe_what ${PROBE_WHAT}
done

echo "Profile normally"
numactl --cpunodebind=0 --membind=0 \
python3 main.py \
 --gpt2_config_path ${CONFIG_PATH} \
 --module_dir ${MODEL_DIR} \
 --module_name ${MODEL} \
 --mode "normal" \
 --ubatchsize_step 1 \
 --num_trials 4 \
# --fwd_umax 18 \
# --bwd_umax 9 \
