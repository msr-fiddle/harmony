#!/bin/bash
MODEL="bert_large"
OUT_DIR="${PWD}/../../results/${MODEL}"
mkdir -p ${OUT_DIR}

# ----- BERT Large -----
echo "Clean Python Processes"
sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

echo "Creating Graph"
python3 main.py \
--data_dir "/data/glue/MRPC" \
--task_name "MRPC" \
--bert_tokenizer 'bert-large-uncased' \
--bert_model 'bert-large-uncased' \
--do_train \
--do_lower_case \
--train_batch_size 1 \
--max_steps 1 \
--no_cuda \
--output_dir ${OUT_DIR} \
--graph_dir ${OUT_DIR}

echo
echo "Generating Code"
pushd . && cd ..
python3 code_generator.py \
--input_dir ${OUT_DIR} \
--output_dir ${OUT_DIR} \
--arch ${MODEL}
popd
