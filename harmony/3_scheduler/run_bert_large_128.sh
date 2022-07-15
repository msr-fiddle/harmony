#!/bin/bash
MODEL_DIR="../results"
MODEL="bert_large"
SUFFIX="_seqlen128"
# -------------- Manual ---------------------------
for D in 32
do
  for MODE in "vDP" "vPP"
  do
    for N in 4
    do
    echo "Manual"
    python3 scheduler.py \
    --manual \
    --manual_ufwd 8 \
    --manual_ubwd 8 \
    --manual_packsize 4 \
    --module_dir ${MODEL_DIR} \
    --module_name ${MODEL} \
    --suffix ${SUFFIX} \
    --minibatchsize ${D} \
    --mode ${MODE} \
    --num_gpus ${N} \
    --simulation 
    done
  done
done
# -------------- Search ---------------------------
for D in 32
do
  for MODE in 'vDP' 'vPP'
  do
    for N in 4
    do
    echo "Search"
    python3 scheduler.py \
    --packing_method_fwd 'balanced_time' 'reuse' \
    --packing_method_bwd 'balanced_time' \
    --topk 10 \
    --dedup \
    --rank_fit_normally \
    --module_dir ${MODEL_DIR} \
    --module_name ${MODEL} \
    --suffix ${SUFFIX} \
    --minibatchsize ${D} \
    --mode ${MODE} \
    --num_gpus ${N} 
    done
  done
done
