#!/bin/bash
# ==== GPT2-Billion-1Layer ====
MODEL="gpt2_b_1layer"
OUT_DIR="${PWD}/../../results/${MODEL}"
mkdir -p ${OUT_DIR}

echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

echo "Creating Graph"
python3 main.py \
--train_data_file "/home/wikitext-103-tokens/wiki.train.tokens" \
--model_type "gpt2" \
--config_name "../../../model_lib/gpt2_configs/gpt2-billion-1layer.json" \
--tokenizer_name "gpt2-xl" \
--cache_dir "/home/pretrained" \
--do_train \
--per_gpu_train_batch_size 1 \
--max_steps 1 \
--logging_steps 1 \
--save_steps -1 \
--no_cuda \
--output_dir ${OUT_DIR} \
--overwrite_output_dir \
--graph_dir ${OUT_DIR} \

echo
echo "Generating Code"
pushd . && cd ..
python3 code_generator.py \
--input_dir ${OUT_DIR} \
--output_dir ${OUT_DIR} \
--arch ${MODEL}
popd

# ==== GPT2-Billions ====
for BILLION in 10 # 20 30 40
do

MODEL="gpt2_${BILLION}b"
OUT_DIR="${PWD}/../../results/${MODEL}"
mkdir -p ${OUT_DIR}

echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

echo "Creating Graph"
python3 main.py \
--train_data_file "/home/wikitext-103-tokens/wiki.train.tokens" \
--model_type "gpt2" \
--config_name "../../../model_lib/gpt2_configs/gpt2-${BILLION}-billion.json" \
--tokenizer_name "gpt2-xl" \
--cache_dir "/home/pretrained" \
--do_train \
--per_gpu_train_batch_size 1 \
--max_steps 1 \
--logging_steps 1 \
--save_steps -1 \
--no_cuda \
--output_dir ${OUT_DIR} \
--overwrite_output_dir \
--graph_dir ${OUT_DIR} \

echo
echo "Generating Code"
pushd . && cd ..
python3 code_generator.py \
--input_dir ${OUT_DIR} \
--output_dir ${OUT_DIR} \
--arch ${MODEL}
popd

done


