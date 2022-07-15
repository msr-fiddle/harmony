#!/bin/bash
MODEL="gpt2_xl"
OUT_DIR="${PWD}/../../results/${MODEL}"
mkdir -p ${OUT_DIR}

# ----- GPT2-XL -----
echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

echo "Creating Graph"
python3 main.py \
--train_data_file "/home/wikitext-103-tokens/wiki.train.tokens" \
--model_type "gpt2" \
--config_name "../../../model_lib/gpt2_configs/gpt2-xl-config.json" \
--tokenizer_name "gpt2-xl" \
--model_name_or_path "gpt2-xl" \
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