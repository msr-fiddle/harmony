# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import sys
from collections import OrderedDict as ODict

import torch

from split_cat import split_minibatch, split_DP_minibatch

from prof_data_struct import ConstMeta, TensorMeta, XMeta, TMeta
from tensor_helper import pin_named_tensors

def _decompose_batch_local(batch, bnames, ubatchszs_fwd, ubatchszs_bwd, XMETA, TMETA, pin_memory=True):
    """ for single GPU or vPP """
    data_ubatches = [] # [ {named_tensors}, {named_tensors}, ... ]
    target_ubatches = [] # [ {named_tensors}, {named_tensors}, ... ]
    ### partition batch into data and target
    batch_data, batch_target = [], []
    batch_dnames, batch_tnames = [], []
    assert len(batch) == len(bnames["is_data"]) and len(batch) == len(bnames["name"])
    for tensor, is_data, name in zip(batch, bnames["is_data"], bnames["name"]):
        if is_data:
            batch_data.append(tensor)
            batch_dnames.append(name)
        else:
            batch_target.append(tensor)
            batch_tnames.append(name)
    ### split data and target into ubatches
    ubatches = split_minibatch(tuple(batch_data), ubatchszs_fwd)
    for tensor_tuple in ubatches:
        named_tensors = ODict()
        assert len(tensor_tuple) == len(batch_dnames)
        for t, n in zip(tensor_tuple, batch_dnames):
            named_tensors[n] = t
        data_ubatches.append(named_tensors)
    ubatches = split_minibatch(tuple(batch_target), ubatchszs_bwd)
    for tensor_tuple in ubatches:
        named_tensors = ODict()
        assert len(tensor_tuple) == len(batch_tnames)
        for t, n in zip(tensor_tuple, batch_tnames):
            named_tensors[n] = t
        target_ubatches.append(named_tensors)
    ### move to pinned memory
    if pin_memory:
        [pin_named_tensors(named_tensors) for named_tensors in data_ubatches]
        [pin_named_tensors(named_tensors) for named_tensors in target_ubatches]
    ### match XMETA, TMETA
    assert len(data_ubatches) == len(ubatchszs_fwd)
    for named_tensors, u in zip(data_ubatches, ubatchszs_fwd):
        for n, t in named_tensors.items(): 
            meta = XMETA.get(u, vlayer_id=0)[n]
            assert t.shape == meta.shape and t.dtype == meta.dtype
    assert len(target_ubatches) == len(ubatchszs_bwd)
    for named_tensors, u in zip(target_ubatches, ubatchszs_bwd):
        for n, t in named_tensors.items(): 
            meta = TMETA.get(u, TMETA.last_vlayer_id)[n]
            assert t.shape == meta.shape and t.dtype == meta.dtype
    
    return data_ubatches, target_ubatches

def decompose_minibatch(minibatch, bnames, ubatchszs_fwd_local, ubatchszs_bwd_local, XMETA, TMETA, CONFIGS, rank, pin_memory=True):
    """ decompose a global minibatch into local microbatches """
    assert isinstance(minibatch, tuple)
    if CONFIGS["N"] == 1 or CONFIGS["mode"] == 'vPP':
        return _decompose_batch_local(minibatch, bnames, ubatchszs_fwd_local, ubatchszs_bwd_local, XMETA, TMETA, pin_memory)
    else: # multi-GPU vDP
        assert sum(CONFIGS['ubatchszs_fwd'][0]) == sum(CONFIGS['ubatchszs_bwd'][0])
        batch_local = split_DP_minibatch(minibatch, CONFIGS['ubatchszs_fwd'], rank)
        return _decompose_batch_local(batch_local, bnames, ubatchszs_fwd_local, ubatchszs_bwd_local, XMETA, TMETA, pin_memory)


       
        



   
