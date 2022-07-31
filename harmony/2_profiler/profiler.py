# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division, print_function
import os
import sys
import argparse
import json
import numpy as np
import gc
from copy import deepcopy
from collections import OrderedDict as ODict

import torch
from torch.autograd import Variable

from time import perf_counter as pc

from prof_data_struct import *

def realize_TensorMeta(meta, ubatchsize=-1, requires_grad=False, force_dtype=None, device="cuda:0", use_rand=True):
    assert type(meta) is TensorMeta
    meta.add_ubatch(ubatchsize) # in-place add ubatchsize to shape if not there
    dtype = force_dtype if force_dtype is not None else meta.dtype
    if dtype == torch.float32:
        tensor = Variable(torch.rand(meta.shape, dtype=torch.float32, device=device)) if use_rand else Variable(torch.ones(meta.shape, dtype=torch.float32, device=device))
    elif dtype == torch.int64:
        tensor = Variable(torch.randint(low=0,high=2,size=meta.shape,dtype=torch.int64, device=device)) if use_rand else Variable(torch.ones(meta.shape,dtype=torch.int64, device=device))
    else:
        raise ValueError("unknown X.dtype={}".format(meta.dtype))
    tensor.requires_grad_(requires_grad)
    if requires_grad:
        tensor.retain_grad()
    return tensor

def realize_X(XMETA, ubatchsize, vlayer_id, names, requires_grad=False, device="cuda:0", use_rand=True):
    named_tensors = ODict() # { name : tensor or const or [tensor,tensor] }
    for name in names: # [XMETA.get(ubatchsize, vlayer_id)[name] for name in X_names]
        meta = XMETA.get(ubatchsize, vlayer_id)[name]
        if name.startswith("input"): # in ["input0","input1","input2","input_ids"]: # TODO: add identity chain to input
            requires_grad = False
        if type(meta) is TensorMeta:
            named_tensors[name] = realize_TensorMeta(meta, ubatchsize, requires_grad, device=device, use_rand=use_rand)
        elif type(meta) is ConstMeta: # output of size(int)
            named_tensors[name] = meta.const
        elif type(meta) is list: # output tuple of bert pretrainhead 
            named_tensors[name] = [realize_TensorMeta(m, ubatchsize, requires_grad, device=device, use_rand=use_rand) for m in meta]
        else:
            raise ValueError("unknown meta={}".format(meta))
    return named_tensors

def realize_D(TMETA, ubatchsize, device="cuda:0", use_rand=True): 
    return realize_X(TMETA, ubatchsize, 0, TMETA.get_names(ubatchsize, vlayer_id=0), requires_grad=False, device=device, use_rand=use_rand)

def realize_T(TMETA, ubatchsize, device="cuda:0", use_rand=True):
    return realize_X(TMETA, ubatchsize, TMETA.last_vlayer_id, TMETA.target_names, requires_grad=False, device=device, use_rand=use_rand)
    
def realize_dX(XMETA, ubatchsize, vlayer_id, names, device="cuda:0", use_rand=True): # excluding T
    named_gradients = ODict() # { name : tensor or None or [tensor,tensor] }
    for name in names: # [XMETA.get(ubatchsize, vlayer_id)[name] for name in X_names]
        meta = XMETA.get(ubatchsize, vlayer_id)[name]
        # if name in ["input0","input1","input2"]: # TODO: add identity chain to input
        #     requires_grad = False
        if type(meta) is TensorMeta:
            assert meta.is_ubatch
            named_gradients[name] = realize_TensorMeta(meta, requires_grad=False, force_dtype=torch.float32, device=device, use_rand=use_rand)
        elif type(meta) is ConstMeta: # output of size(int)
            named_gradients[name] = None
        elif type(meta) is list: # output tuple of bert pretrainhead 
            named_gradients[name] = [realize_TensorMeta(m, requires_grad=False, force_dtype=torch.float32, device=device, use_rand=use_rand) for m in meta]
        else:
            raise ValueError("unknown meta={}".format(meta))
    return named_gradients


class Profiler(object):
    def __init__(self, model, optimizer=None, compute_loss=None, offload_optim=True, device='cuda:0', verbose=False):
        self.model = model
        self.optimizer = optimizer
        # NOTE: safe to self.model and self.optimizer? 
        #       - yes for profile_forward and profile_backward (stateless)
        #       - no for profile_update (modified model and optimizer state) (so leave update to last phase)
        self.compute_loss = compute_loss
        self.offload_optim = offload_optim
        self.device = device
        self.verbose = verbose
        
        # clean up model grad and graph
        self.del_model_grad()

    def _save_Y_tensors_to_named(self, Y_names, Y_tensors, named_tensors):
        assert type(Y_tensors) is list
        if len(Y_names) == 1 and len(Y_tensors) > 1:
            named_tensors[Y_names[0]] = Y_tensors
        else:
            for name, tensor in zip(Y_names, Y_tensors):
                named_tensors[name] = tensor

    @torch.no_grad()
    def _swapin_param_buf(self, vlayer, requires_grad=False): 
        vlayer.cuda()
        if len(list(vlayer.parameters())) != 0:
            for param in vlayer.parameters():
                if param is not None:
                    assert param.grad is None and (not param.requires_grad), \
                    "swapin requires no grad for both FWD and BWD (param={}, param.grad={}, param.requires_grad={})".format(param, param.grad, param.requires_grad) 
                    param.requires_grad_(requires_grad)

    @torch.no_grad()
    def _del_grad(self, vlayer, manual_gc=False):
        if len(list(vlayer.parameters())) != 0:
            for param in vlayer.parameters(): 
                if param is not None:
                    if param.grad is not None:
                        param.grad = None
                    param.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
                    assert not param.requires_grad
        if manual_gc:
            gc.collect(); torch.cuda.empty_cache()

    def del_model_grad(self):
        for vlayer, _, _ in self.model:
            self._del_grad(vlayer, manual_gc=True)

    def _vlayer_forward_an_ubatch(self, ubatchsize, vlayer_id, vlayer, X_names, Y_names, tid, TIME, XMETA, TMETA, requires_grad=False):
        if vlayer_id != len(self.model)-1: # not criterion yet
            # In {X}
            named_tensors = realize_X(XMETA, ubatchsize, vlayer_id, X_names, requires_grad, self.device)
            # Forward on GPU
            torch.cuda.synchronize(self.device)
            t_start = pc() 
            Y_tensors = vlayer(*[named_tensors[name] for name in X_names])
            torch.cuda.synchronize(self.device)
            t_end = pc() 
            # Result
            TIME.add('FWD' if not requires_grad else 'BWD', ubatchsize, vlayer_id, 'GPU', t_end-t_start) 
            # print("\t\t\tforward'ed trial:{}".format(tid))
            if not isinstance(Y_tensors, tuple):
                Y_tensors = (Y_tensors,)
            Y_tensors = list(Y_tensors)
            # Save {Y}
            self._save_Y_tensors_to_named(Y_names, Y_tensors, named_tensors)
            # Out {Y} && {stashX}
            XMETA.set(ubatchsize, vlayer_id+1, Y_names, Y_tensors)
        else: # criterion
            assert Y_names == ["loss"]
            # In {X}
            named_tensors = realize_X(XMETA, ubatchsize, vlayer_id, X_names, requires_grad, self.device)
            # In {T}
            named_targets = realize_T(TMETA, ubatchsize, self.device)
            # Forward on GPU
            torch.cuda.synchronize(self.device)
            t_start = pc()
            if self.compute_loss is not None:
                Y_tensors = self.compute_loss(vlayer, named_tensors, X_names, named_targets)
            else:
                Y_tensors = [vlayer(named_tensors[name],named_targets["target"]) for name in X_names]
                Y_tensors = [sum(Y_tensors)]
            torch.cuda.synchronize(self.device)
            t_end = pc() 
            # Result
            TIME.add('FWD' if not requires_grad else 'BWD', ubatchsize, vlayer_id, 'GPU', t_end-t_start) 
            # print("\t\t\tforward'ed trial:{} loss = {}".format(tid, Y_tensors))
            # Save {Y}
            self._save_Y_tensors_to_named(Y_names, Y_tensors, named_tensors)
            del named_targets
        # Clean up
        del Y_tensors
        # return for backward pass
        if requires_grad:
            return named_tensors
        else:
            del named_tensors

    def _vlayer_backward_an_ubatch(self, ubatchsize, vlayer_id, vlayer, X_names, Y_names, tid, TIME, XMETA, TMETA, named_tensors):
        # In {dY}
        if vlayer_id == len(self.model)-1: # criterion
            assert Y_names == ['loss'] and isinstance(named_tensors['loss'], (torch.Tensor,Variable))
            named_gradients = ODict({ 'loss': None })
            assert named_tensors['loss'].requires_grad
        else:
            named_gradients = realize_dX(XMETA, ubatchsize, vlayer_id+1, self.model[vlayer_id+1][1], self.device)
        # Backward on GPU
        Y_tensors = []
        Y_gradients = [] 
        for name in Y_names: # only tensor & required_grad can run autograd
            Y = named_tensors[name]
            if (type(Y) in [torch.Tensor, Variable]) and (Y.requires_grad):
                Y_tensors.append(Y)
                Y_gradients.append(named_gradients[name])
            elif type(Y) is list: # output tuple of bert pretrainheader
                for i, y in enumerate(Y):
                    if (type(y) in [torch.Tensor, Variable]) and (y.requires_grad):
                        Y_tensors.append(y)
                        Y_gradients.append(named_gradients[name][i])
        torch.cuda.synchronize(self.device)
        t_start = pc() 
        torch.autograd.backward(tuple(Y_tensors), grad_tensors=tuple(Y_gradients))
        torch.cuda.synchronize(self.device)
        t_end = pc() 
        # Result
        TIME.add('BWD', ubatchsize, vlayer_id, 'GPU', t_end-t_start) 
        # print("\t\t\tbackward'ed trial:{}".format(tid))
        # Clean up {X,Y,dX,dY}
        del named_tensors; del named_gradients; del Y_tensors; del Y_gradients

    def profile_forward(self, ubatchsize_range, num_trials, TIME, MEMORY, XMETA, TMETA):
        # NOTE: no OoM allowed in this function
        if self.verbose: print("forward ...")
        for ubatchsize in range(*ubatchsize_range):
            print("\tubatchsize {} ...".format(ubatchsize))
            for vlayer_id, (vlayer, X_names, Y_names) in enumerate(self.model):
                if self.verbose: print("\t\tvlayer_id {}".format(vlayer_id))
                # Clean start
                gc.collect(); torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)
                assert torch.cuda.memory_reserved()==0, "vlayer begin w/ alloc = {} B, resrv = {} B".format(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
                torch.cuda.reset_peak_memory_stats() 
                # Swap-in model {W,B}
                self._swapin_param_buf(vlayer, requires_grad=False)
                # First iteration for MEMORY
                with torch.no_grad():
                    self._vlayer_forward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, 0, TIME, XMETA, TMETA, requires_grad=False)
                    gc.collect()
                MEMORY.set('FWD', ubatchsize, vlayer_id, torch.cuda.max_memory_allocated())
                # Then iterations for TIME
                TIME.reset('FWD', ubatchsize, vlayer_id, 'GPU')
                for tid in range(0, num_trials): # each trial is one microbatch 
                    with torch.no_grad():
                        self._vlayer_forward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, tid, TIME, XMETA, TMETA, requires_grad=False)
                        gc.collect()
                # Swap-out {W,B}
                vlayer.cpu()            

    def profile_backward(self, ubatchsize_range, num_trials, TIME, MEMORY, XMETA, TMETA):
        # NOTE: no OoM allowed in this function
        if self.verbose: print("backward (with recompute) ...")
        for ubatchsize in range(*ubatchsize_range):
            print("\tubatchsize {} ...".format(ubatchsize))
            for vlayer_id, (vlayer, X_names, Y_names) in reversed(list(enumerate(self.model))): # reverse all vlayer (layer)
                if self.verbose: print("\t\tvlayer_id {}".format(vlayer_id))
                # Clean start
                gc.collect(); torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)
                assert torch.cuda.memory_reserved()==0, "vlayer begin w/ alloc = {} B, resrv = {} B".format(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
                torch.cuda.reset_peak_memory_stats() 
                # Swap-in model {W,B}
                self._swapin_param_buf(vlayer, requires_grad=True)
                # First iteration for MEMORY
                named_tensors = self._vlayer_forward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, 0, TIME, XMETA, TMETA, requires_grad=True) # X/Y { name : tensor or const or [tensor,tensor] }
                self._vlayer_backward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, 0, TIME, XMETA, TMETA, named_tensors)
                del named_tensors # very important!
                gc.collect()
                MEMORY.set('BWD', ubatchsize, vlayer_id, torch.cuda.max_memory_allocated())
                # Then iterations for TIME
                TIME.reset('BWD', ubatchsize, vlayer_id, 'GPU')
                for tid in range(0, num_trials): # each trial is one microbatch 
                    named_tensors = self._vlayer_forward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, tid, TIME, XMETA, TMETA, requires_grad=True) # X/Y { name : tensor or const or [tensor,tensor] }
                    self._vlayer_backward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, tid, TIME, XMETA, TMETA, named_tensors)
                    del named_tensors # very important!
                    gc.collect()
                # Swap-out model {dW,W,B}
                self._del_grad(vlayer)
                vlayer.cpu()

    @torch.no_grad()
    def profile_update(self, num_trials, TIME):
        if self.offload_optim:
            for vlayer_id, ((vlayer, _, _), optim) in enumerate(zip(self.model, self.optimizer)):
                if optim is not None:
                    if self.verbose: print("\tvlayer_id {}".format(vlayer_id))
                    # Traverse all trials; each trial is one vlayer update
                    for tid in range(0, num_trials):
                        for param in vlayer.parameters():
                            param.requires_grad_(True)
                            param.grad = torch.rand(param.data.shape, dtype=torch.float32, device="cpu")
                        # compute updated weight
                        t_start = pc() 
                        optim.step()
                        optim.zero_grad()
                        t_end = pc() 
                        TIME.add('UPD', None, vlayer_id, 'CPU', t_end-t_start) 
                        # print("\t\tupdated on trial:{}".format(tid))
        else:
            raise NotImplementedError("update on GPU")
        # print("update done")

    def initial_iteration(self, umax, data_names, data_tensors, target_names, target_tensors):
        ubatchsize_range = [umax, umax + 1, 1]
        TIME = Time(ubatchsize_range, ubatchsize_range, len(self.model))
        MEMORY = Memory(ubatchsize_range, ubatchsize_range, len(self.model))
        XMETA = XMeta(ubatchsize_range, len(self.model))
        TMETA = TMeta(ubatchsize_range, len(self.model))
        XMETA.init_data_meta_on_vlayer0(data_names, data_tensors)
        TMETA.init_target_meta_on_last_vlayer(target_names, target_tensors)
        self.profile_forward(ubatchsize_range, -1, TIME, MEMORY, XMETA, TMETA)
        self.profile_backward(ubatchsize_range, -1, TIME, MEMORY, XMETA, TMETA)
        print("initial iteration finished at batchsize {}".format(umax))

    def probe_max_ubatchsize(self, type, data_names, data_tensors, target_names, target_tensors):
        """ 
        Probe max microbatch size by multiplicative-increase
        
        NOTE: additive-increase/decrease is not used in practice, as it causes repeated program rerun 
        (esp., model's initialization overhead). This is due to the limitation of PyTorch: 
        Each OoM causes memory leak (https://github.com/pytorch/pytorch/issues/27600), and 
        rerun is the only way to recover full GPU memory after OoM.
        
        NOTE: Forward probing, backward probing, normal profiling need three seperate runs of 
        the entire python program, due to the above limitation. 
        """
        assert type in ('FWD', 'BWD')
        
        print("\n----- probing {}'s max microbatch size -----".format(type))
        
        ubatchsize, umax = 1, -1
        while True:
            # print("{}: try ubatchsize {} ...".format(type, ubatchsize))
            try:
                ubatchsize_range = [ubatchsize, ubatchsize + 1, 1]
                TIME = Time(ubatchsize_range, ubatchsize_range, len(self.model))
                MEMORY = Memory(ubatchsize_range, ubatchsize_range, len(self.model))
                XMETA = XMeta(ubatchsize_range, len(self.model))
                TMETA = TMeta(ubatchsize_range, len(self.model))
                XMETA.init_data_meta_on_vlayer0(data_names, data_tensors)
                TMETA.init_target_meta_on_last_vlayer(target_names, target_tensors)
                if type == 'FWD':
                    self.profile_forward(ubatchsize_range, 1, TIME, MEMORY, XMETA, TMETA)
                elif type == 'BWD':
                    self.profile_forward(ubatchsize_range, 1, TIME, MEMORY, XMETA, TMETA)
                    self.profile_backward(ubatchsize_range, 1, TIME, MEMORY, XMETA, TMETA)
                umax = ubatchsize
            except Exception as e: 
                if 'CUDA out of memory' in str(e):
                    del e
                    break
                elif 'an illegal memory access' in str(e):
                    print(e)
                    del e
                    break
                else:
                    raise e
            ubatchsize *= 2
        
        print("--- {}'s max microbatch size = {} ---\n".format(type, umax))
        return umax

def run(args, synthetic_data, create_model, create_optimizer, compute_loss=None):
    
    assert torch.cuda.is_available()
    torch.cuda.set_device(0) # control by CUDA_VISIBLE_DEVICES
    device = "cuda:0"

    """ Initialize model. """
    model = create_model(args)
    print("model created")

    """ Initialize data. """
    data_names, data_tensors, target_names, target_tensors = synthetic_data(args)
    
    """ Initialize Harmony. """
    p = Profiler(model, compute_loss=compute_loss, offload_optim=not args.no_offload_optim, device=device, verbose=args.verbose)

    """ Modes to profile. """ 
    if args.mode == "probe":
        
        umax = p.probe_max_ubatchsize(args.probe_what, data_names, data_tensors, target_names, target_tensors)
        assert umax > 0, "[Error] Invalid {}'s max microbatch size = {}. Likely that even microabatch size = 1 explodes the GPU memory.".format(args.probe_what, umax)
        save_prof_data_struct(umax, args.output_dir, 'probe_{}_umax{}'.format(args.probe_what, args.outname_suffix))
    
    elif args.mode == "normal":
        
        if 'FWDBWD' in args.what:
            
            # get probed ubatchsize
            fwd_umax = load_prof_data_struct(args.output_dir, 'probe_{}_umax{}'.format('FWD', args.outname_suffix)) if args.fwd_umax == -1 else args.fwd_umax
            bwd_umax = load_prof_data_struct(args.output_dir, 'probe_{}_umax{}'.format('BWD', args.outname_suffix)) if args.bwd_umax == -1 else args.bwd_umax
            assert fwd_umax >= bwd_umax, "fwd_umax:{} v.s. bwd_umax:{}".format(fwd_umax, bwd_umax)
            
            # run initial iteration for starting cuda context
            p.initial_iteration(bwd_umax, data_names, data_tensors, target_names, target_tensors)
            
            # set ubatchsize_range for FWD and BWD 
            if args.ubatchsize_step >= 1.0:
                ubatchsize_step = int(args.ubatchsize_step)
            else:
                ubatchsize_step = max(int(float(args.ubatchsize_step) * min(fwd_umax, bwd_umax)), 1)
            fwd_ubatchsize_range = [1, fwd_umax + 1, ubatchsize_step]
            bwd_ubatchsize_range = [1, bwd_umax + 1, ubatchsize_step]
            print("\n----- normal profiling -----")
            print("forward microbatch sizes: [{}, {}) with a step size {}".format(fwd_ubatchsize_range[0], fwd_ubatchsize_range[1], fwd_ubatchsize_range[2]))
            print("backward microbatch sizes: [{}, {}) with a step size {}".format(bwd_ubatchsize_range[0], bwd_ubatchsize_range[1], bwd_ubatchsize_range[2]))
            print("-------------------------------\n")

            # profile FWD and BWD
            TIME = Time(fwd_ubatchsize_range, bwd_ubatchsize_range, len(p.model))
            MEMORY = Memory(fwd_ubatchsize_range, bwd_ubatchsize_range, len(p.model))
            XMETA = XMeta(fwd_ubatchsize_range, len(p.model))
            TMETA = TMeta(fwd_ubatchsize_range, len(p.model))
            XMETA.init_data_meta_on_vlayer0(data_names, data_tensors)
            TMETA.init_target_meta_on_last_vlayer(target_names, target_tensors)
            
            print("\n----- profiling forward -----")
            p.profile_forward(fwd_ubatchsize_range, args.num_trials, TIME, MEMORY, XMETA, TMETA)
            print("\n----- profiling backward -----")
            p.profile_backward(bwd_ubatchsize_range, args.num_trials, TIME, MEMORY, XMETA, TMETA)
            
            # save results
            TIME.avg_trials(args.num_trials)
            
            if args.verbose:
                print(TIME)
                print(MEMORY)
                print(XMETA)
                print(TMETA)
            
            print()
            save_prof_data_struct(TIME, args.output_dir, "prof_TIME_FWDBWD{}".format(args.outname_suffix))
            save_prof_data_struct(MEMORY, args.output_dir, "prof_MEMORY_FWDBWD{}".format(args.outname_suffix))
            save_prof_data_struct(XMETA, args.output_dir, "prof_XMETA{}".format(args.outname_suffix)) # NOTE: data shape is ubatched
            save_prof_data_struct(TMETA, args.output_dir, "prof_TMETA{}".format(args.outname_suffix)) # NOTE: target shape is ubatched
            print()

        if 'UPD' in args.what:
            if not args.no_offload_optim:

                """ Initialize optimizer. """
                p.optimizer = create_optimizer(p.model)
                print("optimizer created on CPU")

                # profile UPD
                TIME = Time(None, None, len(p.model))
                WMETA = WMeta(p.model)
                BMETA = BMeta(p.model)
                KMETA = KMeta(p.model, p.optimizer)
                
                print("\n----- profiling update -----")
                p.profile_update(args.num_trials, TIME)

                # save results
                TIME.avg_trials(args.num_trials)

                if args.verbose:
                    print(TIME)
                    print(WMETA)
                    print(BMETA)
                    print(KMETA)

                print()
                save_prof_data_struct(TIME, args.output_dir, "prof_TIME_UPD{}".format(args.outname_suffix))
                save_prof_data_struct(WMETA, args.output_dir, "prof_WMETA{}".format(args.outname_suffix))
                save_prof_data_struct(BMETA, args.output_dir, "prof_BMETA{}".format(args.outname_suffix))
                save_prof_data_struct(KMETA, args.output_dir, "prof_KMETA{}".format(args.outname_suffix))
                print()

            else:
                raise NotImplementedError("Update on GPU")
    else:
        raise ValueError
