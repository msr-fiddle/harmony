# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

def split_minibatch_size(minibatch_size, ubatch_size):
    """ split a minibatch size into a list of microbatch sizes """
    assert isinstance(minibatch_size, int) and isinstance(ubatch_size, int)
    assert minibatch_size >= ubatch_size
    if minibatch_size % ubatch_size == 0:
        ubatch_sizes = [ubatch_size] * int(minibatch_size/ubatch_size)
    else:
        ubatch_sizes = [ubatch_size] * int(minibatch_size/ubatch_size) \
                        + [minibatch_size%ubatch_size ]
    assert sum(ubatch_sizes) == minibatch_size
    
    return ubatch_sizes

def _split_tensor(t, split_size):
    assert t.ndim > 0, "scalar tensor cannot be split'ed" # dim=0 must be ubatchsize
    return torch.split(t, split_size, dim=0) # share the same underlying memory # inherit tensor's device
    # tensor will be split into equally sized chunks (if possible). 
    # Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.
    # return (t1,t2) or (t1,res) or (t1,) or (res,)

def split_minibatch(minibatch, ubatch_sizes):
    """ split a minibatch into a list of microbatches """
    assert isinstance(minibatch, tuple)
    # input_ids, input_mask, segment_ids, label_ids = minibatch
    
    ### split minibatch
    splits = [_split_tensor(t, ubatch_sizes[0]) for t in minibatch]
    for split_t in splits:
        assert len(split_t) == len(ubatch_sizes)
     
    ### make microbatches
    ubatches = [] # [ubatch#0, ubatch#1, ...] 
    for i, u in enumerate(ubatch_sizes):
        ubatch = tuple(split_t[i] for split_t in splits)
        ubatches.append(ubatch)
        
        for t in ubatch:
            assert t.shape[0] == u
    
    return ubatches

def split_DP_minibatch_size(num_gpus, minibatch_size, ubatch_size):
    """ split the global minibatch size into a list of per-GPU microbatch sizes """
    per_gpu_ubatch_sizes = []
    for n in range(num_gpus):
        # ----- find per-GPU microbatch sizes -----
        DD = int(float(minibatch_size)/num_gpus)
        if minibatch_size % num_gpus != 0: # uneven batch size across GPUs
            if n < minibatch_size % num_gpus:
                DD += 1
        ubszs = split_minibatch_size(DD, ubatch_size)
        per_gpu_ubatch_sizes.append(ubszs)
    
    return per_gpu_ubatch_sizes

def split_DP_minibatch(minibatch, per_gpu_ubatch_sizes, rank):
    """ split the global minibatch into a local batch """
    assert isinstance(minibatch, tuple)
    # input_ids, input_mask, segment_ids, label_ids = minibatch
    minibatchsize0 = sum(per_gpu_ubatch_sizes[0])
    ### split minibatch across GPUs
    splits = [_split_tensor(t, minibatchsize0) for t in minibatch]
    for split_t in splits:
        assert len(split_t) == len(per_gpu_ubatch_sizes)
    ### choose which split by rank
    batch_local = tuple(split_t[rank] for split_t in splits)
    
    return batch_local

