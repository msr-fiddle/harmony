# Harmony Runtime

This directory contains code to run the actual training by consuming upstream's output in `../results/<model>` (i.e., code from `1_decomposer`, profiles from `2_profiler`, and task graph from `3_scheduler`) via follow modules: 

- `runtime.py` (main): initialize Harmony Runtime and launch multiple worker processes (`worker.py`), where each worker ties to one GPU and executes the task graph for the training loop with each operation implemented in a separate module as below.

- `local_model_gpu.py`: wrap each layer's parameters on a GPU, handling any operations regarding to the layer's parameters: swap-in, swap-out, prefetech, allocation, deletion, etc.

- `shared_optim_cpu.py`: wrap each layer's optimizer states on CPU, handling any operations regarding to the layer's optimizer states: optimizer state sharing, weight update, pinned memory copying, etc.
  
- `p2p.py`: peer-2-peer transfer activations, activation gradients, as well as allreduces of weight gradients. It uses double buffering for nonblocking receiving and nonblocking sending. 

- `swp_x.py`: CPU-to-GPU swap activations and activation gradients. It uses double buffering for nonblocking prefetch and nonblocking offload. 
   
- `msg_stash_x.py`: CPU-to-CPU message pass stashing activations. 

- `ubatchsize_converter.py`: convert tensors of different microbatch sizes between forward and backward pipelines. 
  
- `decompose.py`: spliting a minibatch into multiple microbatches.
  
- `tensor_helper.py` and `utils.py`: helps other modules.

- `viewer` (optional): visualize the runtime in `chrome-trace` (chrome://tracing/) via [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview).

## Example Scripts

Example scripts (`run_<model>.sh`) are provided in each model directory (`bert_thomwolf`, `gpt2_huggingface`, and `vgg_resnet_torch`). 