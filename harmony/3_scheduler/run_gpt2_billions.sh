#!/bin/bash
MODEL_DIR="../results"
for MODEL in "gpt2_10b" "gpt2_20b" "gpt2_30b" "gpt2_40b"
do
  
  # -------------- Manual ---------------------------
  for D in 32
  do
    for MODE in "vDP" "vPP"
    do
      for N in 8
      do
      echo "Manual"
      python3 scheduler.py \
      --manual \
      --manual_ufwd 1 \
      --manual_ubwd 1 \
      --manual_packsize 2 \
      --module_dir ${MODEL_DIR} \
      --module_name ${MODEL} \
      --minibatchsize ${D} \
      --mode ${MODE} \
      --num_gpus ${N} \
      --simulation
      done
    done
  done

done