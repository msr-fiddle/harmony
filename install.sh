#!/bin/bash
echo "Within nvidia's pytorch container, install following requirements"
set -x
# For MKL
conda install mkl mkl-include -y
conda install -c mingfeima mkldnn -y 
# For dot graph
pip install graphviz # torchviz
# For GPT2 
pip install tokenizers dataclasses