#!/bin/bash
MODEL="bert_large"
SEQLEN=128
SUFFIX="_seqlen${SEQLEN}"
OUT_DIR="../bert_thomwolf/logs/${MODEL}${SUFFIX}"

SCHEDULED="D32_vPP_N2_Ufwd8_Ubwd8_P4"
PIDS=(2260 2292 2317) # continuous
RANKS=(-1 0 1)
WORLD=2
MODE="vPP"

echo "Convert to chrome-trace.json"
NVVPS=(); for p in ${PIDS[@]}; do NVVPS+=("${SCHEDULED}_pid${p}.nvvp"); done
python3 nvprof2json.py \
--dir_nvvps "${OUT_DIR}" \
--nvvps ${NVVPS[@]} \
--ranks ${RANKS[@]} \
--world_size ${WORLD} \
--mode ${MODE} \
--dir_jsons "${OUT_DIR}" \
--json "${SCHEDULED}_pid${PIDS[0]}_${PIDS[1]}-${PIDS[-1]}.json.gz" \
 |& tee ${OUT_DIR}/${SCHEDULED}_view.txt
# --unify_swap \

