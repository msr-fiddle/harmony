# Harmony

This directory contains four components of Harmony: 

- `1_decomposer`: decompose user model into per-layer code

- `2_profiler`: profile the per-layer code

- `3_scheduler`: find the best schedule based on per-layer profile

- `4_runtime`: run the best schedule on real hardware 

- `results` (optional): store the intermediate results from the above components

Detailed instructions and scripts are under each directory.

## Note

Abbreviations:
- vDP and vPP: Harmony DP and Harmony PP
- vLayer: Harmony layer
- R: number of vLayers
- N: number of GPUs
- D: minibatch size
- U: microbatch size
- P: layer-pack size