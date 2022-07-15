#!/bin/bash
# --------------- BERT Large (128) ---------------
MODEL="bert_large"
SUFFIX="_seqlen128"
CONFIG="Ufwd101_Ubwd101_P2"
DIR="${PWD}/analysis/${MODEL}${SUFFIX}/${CONFIG}"

pushd . && cd ..
python3 analyze_profiles.py \
 --module_name ${MODEL} \
 --suffix ${SUFFIX} \
 --mode "model" \
 --analysis_dir ${DIR} \
  |& tee "${DIR}/model.txt"
python3 analyze_profiles.py \
 --module_name ${MODEL} \
 --suffix ${SUFFIX} \
 --mode "data" \
 --analysis_dir ${DIR}
popd