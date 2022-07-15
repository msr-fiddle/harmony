#!/bin/bash
MODEL_DIR="../results"
MODEL="bert_large"
# -------------- Search ---------------------------
for D in 32
do
  for MODE in 'vDP' 'vPP'
  do
    for N in 2
    do
    echo "Search"
    python3 scheduler.py \
    --packing_method_fwd 'balanced_time' 'reuse' \
    --packing_method_bwd 'balanced_time' \
    --topk 1 \
    --rank_fit_normally \
    --module_dir ${MODEL_DIR} \
    --module_name ${MODEL} \
    --minibatchsize ${D} \
    --mode ${MODE} \
    --num_gpus ${N} 
    done
  done
done
