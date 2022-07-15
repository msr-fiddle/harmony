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

def synthetic_data(args):
    data_names = ["input0"]
    target_names = ["target"]
    
    data_tensors = torch.autograd.Variable(torch.rand(3,224,224)).type(torch.float32)
    target_tensors = torch.Tensor(1).random_(0, 1000)[0].type(torch.int64)

    return data_names, data_tensors, target_names, target_tensors
    
def create_model(args):
    sys.path.append(args.module_dir)
    import importlib; module = importlib.import_module(args.module_name + ".code")

    criterion = torch.nn.CrossEntropyLoss()
    model = module.model(criterion) 
    
    return model

def create_optimizer(model):
    optimizer = []
    from torch.optim import SGD
    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) == 0:
            optimizer.append(None)
        else:
            for param in vlayer.parameters():
                assert not param.is_cuda
            optim = SGD(vlayer.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            optimizer.append(optim)
    return optimizer

if __name__ == "__main__":    
    import prof_args
    args = prof_args.initialize_args()

    import profiler
    profiler.run(args, synthetic_data, create_model, create_optimizer)
