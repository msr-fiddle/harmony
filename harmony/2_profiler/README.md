# Harmony Profiler

This directory contains code to profile per-layer code generated from Harmony Decomposer (`1_decomposer`) via following modules:

- `profiler.py` (main): profile each layer on a single GPU to obtain the characteristics (compute time, memory footprint, activation size) under different microbatch sizes, forward pass and backward pass. These profiled characteristices are stored in a customized data structure specified in `prof_data_struct.py`.

- `analyze_profiles.py` (optional): analyze the generated profiles from `profiler.py` and visualize them.

## Example Scripts

Example scripts (`run_<model>.sh`) are provided in each model directory (`bert_thomwolf`, `gpt2_huggingface`, and `vgg_resnet_torch`). 
Then the generated profiles will be saved to `../results/<model>`, which will be used by downstreams of Harmony. 