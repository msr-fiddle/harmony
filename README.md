# Harmony

This repository contains the source code implementation of the following papers:

- ''[Harmony: Overcoming the hurdles of GPU memory capacity to train massive DNN models on commodity servers](https://www.microsoft.com/en-us/research/publication/harmony-overcoming-the-hurdles-of-gpu-memory-capacity-to-train-massive-dnn-models-on-commodity-servers/)'', which appeared at VLDB 2022.
  
- ''[Doing more with less: Training large DNN models on commodity servers for the masses](https://www.microsoft.com/en-us/research/publication/doing-more-with-less-training-large-dnn-models-on-commodity-servers-for-the-masses/)'', which appeared at HotOS 2021.

This work was done as part of Microsoft Research's [Project Fiddle](https://aka.ms/msr-fiddle). This source code is available under the [MIT License](./LICENSE.txt).

## Directory Structure

- `harmony`: the Harmony source code, with detailed instructions, various example scripts, as well as previous results.

- `model_lib`: the model libary containing model code that is not included in pytorch, such as the transformer library from [huggingface](https://huggingface.co/). 

- `util_lib`: the customized utility libary.

## Setup

To run Harmony, the easiest way is to use the standard nvidia's container (nvcr.io/nvidia/pytorch:20.03-py3) which satisfies most dependencies. It can be launched by:

```bash
./launch.sh
```

Once getting into the container, the remaining dependencies can be satisified by running:

```bash
./install.sh
```

### Note: 

- Harmony was developed in the environment of Python 3.6.9, PyTorch 1.5.0a0, CUDA 10.1.243, cuDNN 7.6.3, NCCL 2.4.8, Nvidia driver 418, Ubuntu 18.04.3 LTS. 

- Harmony was developed with Nivida GPUs. 

- Harmony does not modfiy PyTorch library and may remain portable to different versions.  

## Dataset

- GLUE (including MRPC): It can be downloaded by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpacked to a directorary `/data/glue/MRPC`.

- WikiText-2 and WikiText-103: It can be downloaded from [here](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) and unpacked to a directorary `/data/wikitext-2-tokens` and `/data/wikitext-103-tokens`.

- ImageNet: The ImageNet ILSVC 2012 can be downloaded by running [this script](https://github.com/msr-fiddle/pipedream/blob/pipedream/scripts/download_imagenet.py) and unpacked to a directory `/data/imagenet/`.

## End-to-end Workflow

The end-to-end workflow of Harmony can be illustrated by the figure below:

<img src="Overview3.jpg" alt="drawing" width="80%"/>

For example, to run a BERT-Large with Harmony, we can go through following steps:

### Decompose model into per-layer code
```bash
cd harmony/1_decomposer/bert_thomwolf && ./run_bert_large.sh
```

### Profile each layer
```bash
cd ../../2_profiler/bert_thomwolf && ./run_bert_large.sh
```

### Search the best schedule
```bash
cd ../../3_scheduler && ./run_bert_large.sh
```

### Run the best schedule
```bash
cd ../4_runtime/bert_thomwolf && ./run_bert_large.sh
```

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT License](./LICENSE.txt).

## Reference

If you find the code helpful, citing our papers would be appreciated : )
```bibtex
@article{VLDB22Harmony,
    title = {{Harmony: Overcoming the Hurdles of GPU Memory Capacity to Train Massive DNN Models on Commodity Servers}}, 
    author = {Youjie Li and Amar Phanishayee and Derek Murray and Jakub Tarnawski and Nam Sung Kim},
    journal = {The 48th International Conference on Very Large Databases (VLDB'22)},
    year = {2022},
    address = {Sydney, Australia},
    month = sep
}

@inproceedings{HotOS21Harmony,
    title = {{Doing More with Less: Training Large DNN Models on Commodity Servers for the Masses}},
    author = {Youjie Li and Amar Phanishayee and Derek Murray and Nam Sung Kim},
    booktitle = {Workshop on Hot Topics in Operating Systems (HotOS’21)},
    year = {2021},
    address = {Ann Arbor, MI, USA},
    month = jun
}
```