# Harmony Decomposer

This directory contains code to decompose different PyTorch models from entire model into per-layer code, which goes through two steps (each is a python file):

- graph creation (`graph_creator.py`): parse the user given model via a dry run of forward pass and create a model graph (`Graph`) consisting of many layers (`Node`).

- code generation (`code_generator.py`): augment the model graph to create a sequential graph (i.e., simplifing pipeline training), and then generate per-layer code.
  
## User Models
Example of user models are provided in directories of `bert_thomwolf`, `gpt2_huggingface`, and `vgg_resnet_torch`, in which user code are in `main.py` with necessary instrumentations and user's model libraries are moved to `../../model_lib` for convenience for downstreams of Harmony.

## Example Scripts
Example scripts (`run_<model>.sh`) are provided in each model directory (`bert_thomwolf`, `gpt2_huggingface`, and `vgg_resnet_torch`). 
Then the created graph and generated code will be saved to `../results/<model>`, which will be used by downstreams of Harmony. 

## Note
- The graph related python modules are derived from [PipeDream](https://github.com/msr-fiddle/pipedream). 
