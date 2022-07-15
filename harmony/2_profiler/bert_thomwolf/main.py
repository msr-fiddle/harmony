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
    ### BERT
    parser.add_argument("--bert_seq_length", type=int, default=512, 
                        help="data sample length")
    parser.add_argument("--bert_config_path", type=str, 
                        default="../../../model_lib/bert_configs/bert-large-uncased.json", 
                        help="always use this to config bert")
    return parser

def synthetic_data(args):
    data_names = ["input0", "input1", "input2"]
    target_names = ["labels"]
    
    tensor_shapes = {"input0": [args.bert_seq_length], "input1": [args.bert_seq_length], "input2": [1, 1, args.bert_seq_length], "labels": [] }
    tensor_dtypes = {"input0": torch.int64, "input1": torch.int64, "input2": torch.float32, "labels": torch.int64}
    
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

    from bert_thomwolf.modeling2 import BertConfig
    config = BertConfig.from_json_file(args.bert_config_path)
    criterion = torch.nn.CrossEntropyLoss()
    model = module.model(config, criterion)
    
    return model

def create_optimizer(model):
    optimizer = []
    from bert_thomwolf.optimization2 import BertAdam
    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) == 0:
            optimizer.append(None)
        else:
            for param in vlayer.parameters():
                assert not param.is_cuda
            optim = BertAdam(vlayer.parameters(), lr=3e-5, weight_decay_rate=0.01)
            optimizer.append(optim)
    return optimizer

def compute_loss(last_vlayer, named_tensors, X_names, named_targets):
    logits = named_tensors[X_names[0]]
    loss = last_vlayer(logits.view(-1, 2), named_targets["labels"].view(-1))
    return [loss]

if __name__ == "__main__":    
    import prof_args
    args = prof_args.initialize_args(custom_args_provider=add_args)

    import profiler
    profiler.run(args, synthetic_data, create_model, create_optimizer, compute_loss=compute_loss)
