# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os

@torch.no_grad()
def check_grad_sum_baseline(cpu_named_params):
    gs = 0.
    for name, param in cpu_named_params:
        assert param.grad is not None
        gs += torch.sum(param.grad.data).item()
    return gs

@torch.no_grad()
def check_grad_sum_harmony(share_model):
    gs = 0.
    for vlayer, _, _ in share_model:
        if len(list(vlayer.parameters())) == 0:
            pass
        else:
            for name, param in vlayer.named_parameters():
                assert param.grad is not None
                gs += torch.sum(param.grad.data).item()
    return gs

@torch.no_grad()
def check_grad_sum_vPP(share_model, rank, TASKS):
    assert False
    gs = 0.
    for id, (vlayer, _, _) in enumerate(share_model):
        if len(list(vlayer.parameters())) == 0:
            pass
        else:
            for name, param in vlayer.named_parameters():
                assert param.grad is not None
                gs += torch.sum(param.grad.data).item()
    return gs

def diff_model_checkpoints(path1, path2):
    assert os.path.exists(path1) and os.path.exists(path2)
    state_dict1 = torch.load(path1)
    state_dict2 = torch.load(path2)
    if len(state_dict1) != len(state_dict2):
        return False
    for v1, v2 in zip(state_dict1.values(), state_dict2.values()):
        if not torch.equal(v1, v2):
            return False
    return True
