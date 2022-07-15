#!/bin/bash
MODEL="resnet1026"
OUT_DIR="${PWD}/../../results/${MODEL}"
mkdir -p ${OUT_DIR}

# ----- ResNet1026 -----
echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

echo "Creating Graph"
python3 main.py \
--data "/data/imagenet" \
--arch ${MODEL} \
--workers 0 \
--max_steps 1 \
--batch-size 1 \
--print-freq 1 \
--seed 42 \
--no_cuda \
--graph_dir ${OUT_DIR} \

echo
echo "Generating Code"
pushd . && cd ..
python3 code_generator.py \
--input_dir ${OUT_DIR} \
--output_dir ${OUT_DIR} \
--arch ${MODEL} 
popd