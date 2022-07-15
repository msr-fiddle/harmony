# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import json
from collections import OrderedDict as ODict

import torch

import sys
sys.path.append("../../../model_lib")
sys.path.append("../../../util_lib")

sys.path.append("../../2_profiler")

def add_args(parser):
    ### GPT2
    parser.add_argument("--gpt2_config_path", type=str, 
                        default="../../../model_lib/gpt2_configs/gpt2-xl-config.json", 
                        help="")
    return parser


def synthetic_data(args): # "gpt2_huggingface" and "gpt2_2bw"
    data_names = ["input0"]
    target_names = ["labels"]
    
    tensor_shapes = {"input0": [args.config.n_positions], "labels": [args.config.n_positions]}
    tensor_dtypes = {"input0": torch.int64, "labels": torch.int64}

    data_tensors = []
    for name in data_names:
        data_tensors.append(torch.ones(tuple(tensor_shapes[name]), dtype=tensor_dtypes[name]))
    
    target_tensors = []
    for name in target_names:
        target_tensors.append(torch.ones(tuple(tensor_shapes[name]), dtype=tensor_dtypes[name]))   

    return data_names, data_tensors, target_names, target_tensors
    
def create_model(args):
    sys.path.append(args.module_dir)
    import importlib; module = importlib.import_module(args.module_name + ".code")

    from gpt2_huggingface import GPT2Config
    config = GPT2Config.from_json_file(args.gpt2_config_path)
    # print("gpt2config({}): num_hidden_layers={}, hidden_size={}, num_attention_heads={}, max_seq_length={}".format(
    #     os.path.basename(args.gpt2_config_path), config.num_hidden_layers,config.hidden_size,config.num_attention_heads,config.n_positions))
    criterion = torch.nn.CrossEntropyLoss()
    model = module.model(config, criterion)
    
    args.config = config

    return model

def create_optimizer(model):
    optimizer = []
    from gpt2_huggingface.optimization2 import AdamW
    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) == 0:
            optimizer.append(None)
        else:
            for param in vlayer.parameters():
                assert not param.is_cuda
            optim = AdamW(vlayer.parameters(), lr=3e-5, weight_decay=0.0)
            optimizer.append(optim)
    return optimizer

def compute_loss(last_vlayer, named_tensors, X_names, named_targets):
    output = named_tensors[X_names[0]][..., :-1, :].contiguous()
    output = output.view(-1, output.size(-1))
    shift_labels = named_targets["labels"][..., 1:].contiguous()
    loss = last_vlayer(output, shift_labels.view(-1))
    return [loss]

if __name__ == "__main__":    
    import prof_args
    args = prof_args.initialize_args(custom_args_provider=add_args)

    import profiler
    profiler.run(args, synthetic_data, create_model, create_optimizer, compute_loss=compute_loss)
