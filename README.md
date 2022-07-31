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

More examples can be found under `harmony/1_decomposer`, `harmony/2_profiler`, `harmony/3_scheduler`, and `harmony/4_runtime`.

## Experiments
To conduct the experiments in the VLDB paper, the scripts are provided as below:

- Figure 8
    ```bash
    cd harmony/4_runtime/bert_thomwolf && ./run_bert_large__fig8.sh
    ```

- Figure 10
    ```bash
    cd harmony/4_runtime/bert_thomwolf && ./run_bert96__fig10.sh
    cd harmony/4_runtime/gpt2_huggingface && ./run_gpt2_xl__fig10_fig12.sh
    cd harmony/4_runtime/vgg_resnet_torch && ./run_vgg416__fig10.sh
    cd harmony/4_runtime/vgg_resnet_torch && ./run_resnet1026__fig10.sh
    ```

- Figure 12
    ```bash
    cd harmony/4_runtime/gpt2_huggingface && ./run_gpt2_xl__fig10_fig12.sh
    ```

- Figure 13
    ```bash
    cd harmony/4_runtime/bert_thomwolf && ./run_bert_large__fig13.sh
    ```

- Figure 17 and Figure 18
    ```bash
    cd harmony/4_runtime/gpt2_huggingface && ./run_gpt2_billions__fig17_fig18.sh
    ```
   
- Figure 21
    ```bash
    cd harmony/4_runtime/gpt2_huggingface && ./run_gpt2_medium__fig21.sh
    ```

- Table 1
    ```bash
    cd harmony/3_scheduler && ./run_four_models__tab1.sh
    ```

### Note

For experiments of Figure 17 and Figure 18, three prerequisits exist to run largest models saturating the CPU memory capacity. (Tested on Ubuntu 18.04.)

- [Raise the limitation of pinned memory](https://linux.die.net/man/5/limits.conf)
  
    Step 1: open /etc/security/limits.conf
    ```bash
    sudo vim /etc/security/limits.conf
    ```

    Step 2: make memlock unlimited
    ```
    #<domain>      <type>  <item>         <value>
    #

    #*               soft    core            0
    #root            hard    core            100000
    #*               hard    rss             10000
    #@student        hard    nproc           20
    #@faculty        soft    nproc           20
    #@faculty        hard    nproc           50
    #ftp             hard    nproc           0
    #ftp             -       chroot          /ftp
    #@student        -       maxlogins       4

    *              -       memlock         unlimited
    root           -       memlock         unlimited

    # End of file
    ```

    Step 3: verify 
    ```
    ulimit -a
    ```

- [Max out shared memory](https://masukkhan.wordpress.com/2015/12/09/resize-devshm-filesystem-in-linux/)
  
    Step 1: Open /etc/fstab
    ```bash
    sudo vim /etc/fstab 
    ```

    Step 2: Locate /dev/shm and use the tmpfs size option to specify max size
    ```
    # /etc/fstab: static file system information.
    #
    # Use 'blkid' to print the universally unique identifier for a
    # device; this may be used with UUID= as a more robust way to name devices
    # that works even if disks are added and removed. See fstab(5).
    #
    # <file system> <mount point>   <type>  <options>       <dump>  <pass>
    # / was on /dev/sda1 during installation
    UUID=4e3b7d44-77c9-4cc8-be72-fa2ff836ac2f /               ext4    errors=remount-ro 0       1
    /swapfile                                 none            swap    sw              0       0
    # resize /dev/shm
    tmpfs /dev/shm tmpfs defaults,size=750g 0 0
    ```

    Step 3: To make change effective immediately, remount the /dev/shm filesystem:
    ```bash
    mount -o remount /dev/shm
    ```

    Step 4: Verify
    ```bash 
    df -h
    ```

- [Disable swapping to disk](https://askubuntu.com/questions/1357/how-to-empty-swap-if-there-is-free-ram)

    Step 1: Open sysctl.conf
    ```bash
    sudo vim /etc/sysctl.conf
    ```

    Step 2: Add this line vm.swappiness = 0
    ```
    ###################################################################
    # Protected links
    #
    # Protects against creating or following links under certain conditions
    # Debian kernels have both set to 1 (restricted) 
    # See https://www.kernel.org/doc/Documentation/sysctl/fs.txt
    #fs.protected_hardlinks=0
    #fs.protected_symlinks=0

    vm.swappiness = 0
    ```

    Step 3: Restart machine
    ```
    sudo reboot now
    ```

    After all experiments, restore swapping to disk
    ```
    # vm.swappiness = 0 # comment out
    ```

- Setup Container

    Finally, we need to unlock the resource limitation of container by setting options in `launch.sh` as below. Assume that the machine has 750GB CPU memory and 8 GPUs. 

    ```bash
    nvidia-docker run \
        ...
        --memory=750g \
        --memory-swap=750g \
        --memory-swappiness=0 \
        --memory-reservation=750g \
        --shm-size=750g \
        --ulimit memlock=750000000000:750000000000 \
        --gpus '"device=0,1,2,3,4,5,6,7"' \
        ...
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