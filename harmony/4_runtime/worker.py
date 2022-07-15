# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division, print_function
import argparse
import os
import sys
import json
import numpy as np
import gc
from collections import OrderedDict as ODict
from copy import deepcopy

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
import torch.distributed as dist

import torch.cuda.profiler as cuda_profiler 
from torch.cuda.nvtx import mark as nvtx_mark 
from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 
from viewer.probe_cuda_mem import ProbeCudaMem 
from time import perf_counter as pc

import seeding, checker

from profiler import realize_X, realize_dX, realize_T
from task_data_struct import Medium, vTask
import shared_optim_cpu, local_model_gpu, msg_stash_x, ubatchsize_converter, swp_x, p2p
from decompose import decompose_minibatch
from tensor_helper import * 
from utils import *

class Worker(object): # each rank process
    def __init__(self, args, real_dataset, shared_model, shared_optimizer, empty_model, get_lr_sched, compute_loss, save_model, XMETA, TMETA, rTASKS, CONFIGS, rank):
        self.args = args
        self.shared_model = shared_model
        self.shared_optimizer = shared_optimizer
        self.compute_loss = compute_loss
        self.save_model = save_model
        self.XMETA, self.TMETA = XMETA, TMETA
        self.rTASKS, self.CONFIGS = rTASKS, CONFIGS
        self.rank, self.world_size = rank, CONFIGS["N"]
        self.verbose, self.nvprof = args.verbose, args.nvprof

        # worker process must re-seed
        seeding.seed(args.seed, args.seed_cudnn) 
        self.rand_state_train = seeding.RandState()

        # per-rank configs
        if CONFIGS['mode'] == 'vPP':
            self.ubatchszs_fwd_local = CONFIGS['ubatchszs_fwd']
            self.ubatchszs_bwd_local = CONFIGS['ubatchszs_bwd']
            self.minibatchsize_local = CONFIGS['D']
        elif CONFIGS['mode'] == 'vDP':
            self.ubatchszs_fwd_local = CONFIGS['ubatchszs_fwd'][self.rank]
            self.ubatchszs_bwd_local = CONFIGS['ubatchszs_bwd'][self.rank]
            self.minibatchsize_local = sum(self.ubatchszs_fwd_local)
            assert self.minibatchsize_local == sum(self.ubatchszs_bwd_local)
        else:
            raise ValueError
        self.is_convert_ubs = True if CONFIGS["u_fwd"] != CONFIGS["u_bwd"] else False
        
        # Initialize the Gloo world first
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = str(args.master_port)
        dist.init_process_group(backend="gloo", rank=self.rank, world_size=self.world_size)      
        assert dist.get_rank() == self.rank and dist.get_world_size() == self.world_size
        print("rank%d (pid %d): initialized Gloo world. world_size %d" % (self.rank, os.getpid(), self.world_size))
        
        # Set up GPU
        torch.cuda.set_device(self.rank)
        
        # initialize dataset (must be local to be pinned)
        if args.synthetic_data:
            self.data_loader = list(range(args.num_iters))
            self.data_ubatches, self.target_ubatches = synthesize_data(XMETA, TMETA, self.ubatchszs_fwd_local, self.ubatchszs_bwd_local, pin_memory=not args.no_pin_data)
        else:
            self.data_loader, _, self.is_skip_minibatch, self.preprocess_minibatch, self.bnames, self.fdim, self.is_copy_minibatch = real_dataset(args, CONFIGS["D"], args.data_workers)
            self.data_ubatches, self.target_ubatches = None, None
        
        # initialize shared optimizer locally
        self.pcm = PrintCPUMem()
        self.pcm.print("rank%d: before initializing optimizer" % self.rank)
        lr_scheduler = []
        for id, optim in enumerate(shared_optimizer):
            optim.init_in_subproc(self.rank, no_pin_model=args.no_pin_model, no_pin_grad_buf=args.no_pin_grad_buf)
            if get_lr_sched is not None: # "gpt2_huggingface"      
                lr_scheduler.append(None if optim.shared_optimizer is None else 
                                    get_lr_sched(args, optim.shared_optimizer))
        self.pcm.print("rank%d: optimizer initialized" % self.rank)
        
        # initialize local model GPU 
        self.local_model = []
        for vlayer_id, (optim, (_,X_names,Y_names), empty_vlayer) in enumerate(zip(shared_optimizer, shared_model, empty_model)):
            local_vlayer = local_model_gpu.LocalModelGPU(optim.pinned_model, optim.shared_model, empty_vlayer, vlayer_id, X_names, Y_names, self.rank, self.world_size, no_pin_model=args.no_pin_model, no_pin_grad_buf=args.no_pin_grad_buf) 
            local_vlayer.train() # shared_model/pinned_train.train() not necessary
            self.local_model.append(local_vlayer)
        self.pcm.print("rank%d: local model initialized" % self.rank)
        
        # initialize MSG stashing X on CPU
        layer_X_names = ODict()
        for vlayer_id, (_,X_names,_) in enumerate(shared_model):
            layer_X_names[vlayer_id] = X_names
        msg_stashx = msg_stash_x.MSGStashX(self.rank, rTASKS, layer_X_names, XMETA, self.ubatchszs_bwd_local, 'pack-by-pack', pin_memory=not args.no_pin_x, nvprof=self.nvprof)
        swapout_stashx_output_fn = msg_stashx.isend
        if self.is_convert_ubs: # initialize Optional UBatchSize Converter on CPU
            stashx_ubs_converter = ubatchsize_converter.UBatchSizeConverter(self.rank, self.minibatchsize_local, CONFIGS['u_fwd'], self.ubatchszs_fwd_local, CONFIGS['u_bwd'], self.ubatchszs_bwd_local, msg_stashx.isend, pack_ordering=False, pin_memory=not args.no_pin_x, nvprof=self.nvprof)
            swapout_stashx_output_fn = stashx_ubs_converter.isend
        
        # initialize SWP locally
        local_x = None
        if (CONFIGS['mode'] == 'vPP' and CONFIGS['N'] == 1) or (CONFIGS['mode'] == 'vDP'):
            local_x = msg_stash_x.LocalX(self.rank, list(range(CONFIGS['R'])))
            swapout_localx_output_fn = local_x.isend
            if self.is_convert_ubs: # initialize Optional UBatchSize Converter on CPU
                localx_ubs_converter = ubatchsize_converter.UBatchSizeConverter(self.rank, self.minibatchsize_local, CONFIGS['u_fwd'], self.ubatchszs_fwd_local, CONFIGS['u_bwd'], self.ubatchszs_bwd_local, local_x.isend, pack_ordering=False, pin_memory=not args.no_pin_x, nvprof=self.nvprof)
                swapout_localx_output_fn = localx_ubs_converter.isend
        
        # initialize P2P
        self.p2px_handler, self.p2pm_handler = None, None
        if CONFIGS['mode'] == 'vPP' and CONFIGS['N'] > 1:
            self.p2px_handler = p2p.P2PX(self.rank, self.world_size, CONFIGS['reverse_bwd'], verbose=self.verbose, nvprof=self.nvprof)
        elif CONFIGS['mode'] == 'vDP' and CONFIGS['N'] > 1:
            self.p2pm_handler = p2p.P2PModel(self.rank, self.world_size, verbose=self.verbose)

        # Get default cuda stream (already initialized by local_model_gpu)
        self.default_stream = torch.cuda.default_stream(self.rank)

        # initialize Update in Background thread
        self.update_handler = shared_optim_cpu.UpdateInBkgd(self.default_stream, shared_optimizer, lr_scheduler, self.rank, nvprof=self.nvprof)

        # initialize Prefetch Model background thread
        syncpin_handler = shared_optim_cpu.SyncPinModelInBkgd(shared_optimizer, self.rank, nvprof=self.nvprof)
        self.prefetch_model_handler = local_model_gpu.PrefetchLocalModelGPU(syncpin_handler, self.local_model, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof)
        
        # initialize SwapIn background thread
        self.swapin_stashx_handler = swp_x.SwapIn(msg_stashx.recv, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof) 
        self.swapin_localx_handler = swp_x.SwapIn(local_x.recv, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof) if local_x is not None else None
        
        # initialize SwapOut background thread
        swapout_stream = torch.cuda.Stream(device=self.rank)
        self.swapout_stashx_handler = swp_x.SwapOut(swapout_stashx_output_fn, self.rank,    
                                    swapout_stream=swapout_stream,
                                    compute_stream=self.default_stream, 
                                    blocking=True if args.no_offload_stashx else False,
                                    pin_memory=not args.no_pin_x,
                                    nvprof=self.nvprof)
        self.swapout_localx_handler = swp_x.SwapOut(swapout_localx_output_fn, self.rank, 
                                    swapout_stream=swapout_stream, 
                                    compute_stream=self.default_stream, 
                                    blocking=True if args.no_offload_localx else False,
                                    pin_memory=not args.no_pin_x,
                                    nvprof=self.nvprof) \
                                    if local_x is not None else None
        
        # initialize MSG X on CPU # NOTE: tentatively only for last FWD to first BWD
        msg_x = msg_stash_x.MSGX(self.rank, rTASKS, layer_X_names, XMETA, self.ubatchszs_bwd_local, 'pack-by-pack', pin_memory=not args.no_pin_x, nvprof=self.nvprof)
        swapout_msgx_output_fn = msg_x.isend
        self.swapout_msgx_handler, self.swapin_msgx_handler = None, None
        if msg_x.has_no_send() and msg_x.has_no_recv():
            del msg_x; msg_x = None
        elif not msg_x.has_no_send() and msg_x.has_no_recv(): # sender only
            if self.is_convert_ubs:
                msgx_ubs_converter = ubatchsize_converter.UBatchSizeConverter(self.rank, self.minibatchsize_local, CONFIGS['u_fwd'], self.ubatchszs_fwd_local, CONFIGS['u_bwd'], self.ubatchszs_bwd_local, msg_x.isend, pack_ordering=False, pin_memory=not args.no_pin_x, nvprof=self.nvprof)
                swapout_msgx_output_fn = msgx_ubs_converter.isend
            self.swapout_msgx_handler = swp_x.SwapOut(swapout_msgx_output_fn, self.rank, 
                                        swapout_stream=swapout_stream, 
                                        compute_stream=self.default_stream, 
                                        blocking=True if args.no_offload_msgx else False,
                                        pin_memory=not args.no_pin_x,
                                        nvprof=self.nvprof)
        elif msg_x.has_no_send() and not msg_x.has_no_recv(): # recver only
            self.swapin_msgx_handler = swp_x.SwapIn(msg_x.recv, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof)
        else:
            raise NotImplementedError
        
        # initialize succesor info for all prefetch
        self.sucinfo = SucInfoForPrefetch(self.rank, rTASKS, XMETA)

    ################### Initial Iteration ###################
    def _initial_a_pack_forward_an_ubatch(self, vt, ubatch_idx, ubatch_size, requires_grad=False, verbose=False, nvprof=False):
        if not vt.has_criterion:
            ### In {X}
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names
            X_named_tensors = realize_X(self.XMETA, ubatch_size, l, X_names, requires_grad, "cuda:%d"%self.rank, use_rand=False)
            ### Compute forward pass on GPU
            if nvprof: nvtx_range_push("task{}({}) {}(#{})".format(vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx)) 
            Y_tensors = [X_named_tensors[name] for name in X_names]
            for l in vt.layers:
                Y_tensors = self.local_model[l](*Y_tensors)
                if not isinstance(Y_tensors, tuple):
                    Y_tensors = (Y_tensors,)
                Y_tensors = list(Y_tensors)
            if verbose: print("\trank{}: task{}({}) {}(#{})".format(self.rank, vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx))
            if nvprof: nvtx_range_pop()
            ### Save Y
            l = vt.layers[-1]
            Y_names = self.local_model[l].Y_names
            Y_named_tensors = make_tensors_named(Y_names, Y_tensors)
            ### Clean up
            del Y_tensors
            if not requires_grad:
                del X_named_tensors; del Y_named_tensors
            else:
                return X_named_tensors, Y_named_tensors
        else: # criterion pack
            assert requires_grad
            ### In {X}
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names
            X_named_tensors = realize_X(self.XMETA, ubatch_size, l, X_names, requires_grad, "cuda:%d"%self.rank, use_rand=False)
            ### Recompute on GPU
            if nvprof: nvtx_range_push("task{}({}) Recompute(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            if len(vt.layers) > 1: # packed
                Y_tensors = [X_named_tensors[name] for name in X_names]
                for l in vt.layers[:-1]:
                    Y_tensors = self.local_model[l](*Y_tensors)
                    if not isinstance(Y_tensors, tuple):
                        Y_tensors = (Y_tensors,)
                    Y_tensors = list(Y_tensors)
                Y_names = self.local_model[vt.layers[-2]].Y_names
                Y_named_tensors = make_tensors_named(Y_names, Y_tensors)
            else: # only last vlayer
                Y_names = X_names
                Y_named_tensors = X_named_tensors
            ### In {T}
            T_named_tensors = realize_T(self.TMETA, ubatch_size, "cuda:%d"%self.rank, use_rand=False)
            ### Compute loss on GPU
            assert vt.layers[-1] == self.CONFIGS['R']-1
            last_vlayer = self.local_model[self.CONFIGS['R']-1]
            if self.compute_loss is not None: 
                Y_tensors = self.compute_loss(last_vlayer, Y_named_tensors, Y_names, T_named_tensors)
            else:
                Y_tensors = [last_vlayer(Y_named_tensors[name],T_named_tensors["target"]) for name in Y_names]
                Y_tensors = [sum(Y_tensors)]
            if verbose: print("\trank{}: task{}({}) Recompute(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            if nvprof: nvtx_range_pop()
            ### Save Y
            Y_named_tensors = make_tensors_named(['loss'], Y_tensors)
            ### Clean up
            del T_named_tensors; del Y_tensors; 
            return X_named_tensors, Y_named_tensors

    def _initial_a_pack_backward_an_ubatch(self, vt, ubatch_idx, ubatch_size, X_named_tensors, Y_named_tensors, verbose=False, nvprof=False):
        ### In {dY}
        if vt.has_criterion:
            dY_named_tensors = ODict({ 'loss': None })
            assert Y_named_tensors['loss'].requires_grad
        else:
            l, m = vt.layers[-1], vt.In['dY'][vt.layers[-1]]
            dY_named_tensors = realize_dX(self.XMETA, ubatch_size, l+1, self.local_model[l+1].X_names, device="cuda:%d"%self.rank, use_rand=False)
        ### Compute backward pass
        if nvprof: nvtx_range_push("task{}({}) BWD(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
        Y_tensors = []
        Y_gradients = [] 
        for name in self.local_model[vt.layers[-1]].Y_names:
            Y = Y_named_tensors[name]
            if isinstance(Y,(torch.Tensor, Variable)) and (Y.requires_grad):
                Y_tensors.append(Y)
                Y_gradients.append(dY_named_tensors[name])
            elif isinstance(Y, list): 
                for i, y in enumerate(Y):
                    if isinstance(y,(torch.Tensor, Variable)) and (y.requires_grad):
                        Y_tensors.append(y)
                        Y_gradients.append(dY_named_tensors[name][i])
        torch.autograd.backward(tuple(Y_tensors), grad_tensors=tuple(Y_gradients))
        if verbose: print("\trank{}: task{}({}) BWD(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
        if nvprof: nvtx_range_pop() 
        ### Clean up {X,Y,dX,dY}
        del X_named_tensors; del Y_named_tensors
        del dY_named_tensors; del Y_tensors; del Y_gradients

    def run_initial_iteration(self, verbose=False, nvprof=False):
        if self.args.no_initial_iter:
            print("rank%d: --- No Initial Iteration ---" % self.rank)
            return

        print("rank%d: initial iteration starts"%(self.rank))
        assert dist.get_rank() == self.rank and torch.cuda.current_device() == self.rank
        # clean memory before start
        torch.cuda.synchronize(self.rank); dist.barrier()
        gc.collect(); torch.cuda.empty_cache() 
        torch.cuda.synchronize(self.rank)
        assert torch.cuda.memory_reserved(self.rank)==0
        dist.barrier()
        # task starts 
        if nvprof:
            probe_cuda_mem = ProbeCudaMem(self.rank)
            probe_cuda_mem.start()  
            cuda_profiler.start()
            nvtx_mark("cudaProfilerStart") 
            print("rank%d: cuda profiler starts" % self.rank)    
        time_start = pc()        
        for j, vt in enumerate(self.rTASKS[self.rank]): # { rank0: [task0,task2,task5,...] }
            if verbose: print("\trank{}: executing {}".format(self.rank, vt))
            if vt.type == 'FWD' and vt.is_gpu:
                # -----------------------------------------------      
                with torch.no_grad():
                    ### Swap-in model {W,B}
                    if nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                    if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                    cur_vt_idx = self.prefetch_model_handler.get(vt, None)
                    assert cur_vt_idx == vt.idx
                    if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                    if nvprof: nvtx_range_pop() 
                    ### Run through each microbatch in a data batch
                    for i, u in enumerate(vt.ubatchszs):
                        self._initial_a_pack_forward_an_ubatch(vt, i, u, requires_grad=False, verbose=verbose, nvprof=nvprof)
                        gc.collect()
                        if verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                    ### Delete model {W,B}
                    self.default_stream.synchronize() # CPU wait Compute
                    if nvprof: nvtx_range_push("task{}({}) Del(W,B)".format(vt.idx, vt.show_layers())) 
                    for l in vt.layers:
                        if not (l in vt.Out['W']) and not (l in vt.Out['B']):
                            self.local_model[l].del_param_grad_buf()
                        elif vt.Out['W'][l].medium=='PIN' and vt.Out['B'][l].medium=='PIN':
                            pass
                        else: # P2P
                            raise ValueError("Underdevelopment")
                    gc.collect()
                    if verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                    if nvprof: nvtx_range_pop() 
                # -----------------------------------------------
            elif vt.type == 'BWD' and vt.is_gpu:
                # -----------------------------------------------
                ### Swap-in model {W,B}
                if nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                cur_vt_idx = self.prefetch_model_handler.get(vt, None)
                assert cur_vt_idx == vt.idx 
                if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                if nvprof: nvtx_range_pop() 
                ### Run through each microbatch in a data batch. 
                for i, u in enumerate(vt.ubatchszs):
                    ### Recompute to create pytorch graph
                    X_named_tensors, Y_named_tensors = \
                        self._initial_a_pack_forward_an_ubatch(vt, i, u, requires_grad=True, verbose=verbose, nvprof=nvprof) 
                    ### Backward pass on recomputed graph
                    self._initial_a_pack_backward_an_ubatch(vt, i, u, X_named_tensors, Y_named_tensors, verbose=verbose, nvprof=nvprof)
                    ### Clean up
                    del X_named_tensors; del Y_named_tensors # very important!
                    gc.collect()
                    if verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                ### Swap-out model {W,dW,B}
                if self.CONFIGS["opt_offld"]:                
                    ### Delete model {W,dW,B}
                    self.default_stream.synchronize() # CPU wait for SwapOut
                    if nvprof: nvtx_range_push("task{}({}) Del(W,dW,B)".format(vt.idx, vt.show_layers())) 
                    for l in vt.layers: 
                        if vt.Out['dW'][l].medium == "LOC" and vt.Out['B'][l].medium == "SHM":
                            self.local_model[l].del_param_grad_buf()
                        else: # 'B' == PIN
                            raise ValueError("Underdevelopment")
                    gc.collect()
                    if verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                    if nvprof: nvtx_range_pop() 
                else:
                    raise ValueError("GPU Optimizer Underdevelopment.")
                # -----------------------------------------------
            elif vt.type == 'UPD' and not vt.is_gpu:
                # -----------------------------------------------
                pass
                # -----------------------------------------------
            else:
                raise ValueError("Unknown vTask.type {} with .device {} !".format(vt.type,vt.device))
        # tasks ends
        torch.cuda.synchronize(self.rank); dist.barrier()
        time_end = pc() 
        if nvprof:
            nvtx_mark("cudaProfilerStop") 
            cuda_profiler.stop()
            probe_cuda_mem.stop()
            print("rank%d: cuda profiler stops" % self.rank) 
        print("rank%d: initial iteration ends. time %.3f s"%(self.rank, time_end-time_start))
        # clean memory
        gc.collect(); torch.cuda.empty_cache() 
        torch.cuda.synchronize(self.rank)
        assert torch.cuda.memory_reserved(self.rank)==0
        dist.barrier()
        
        if self.args.initial_iter_only:
            print("rank%d: --- Initial Iteration Only ---" % self.rank)
            exit(0) 

    ################### Regular Training Loop ###################
    def _a_pack_forward_an_ubatch(self, vt, ubatch_idx, ubatch_size,
                                data_ubatches, target_ubatches, 
                                requires_grad=False, 
                                prefetch_model_handler=None,
                                swapin_stashx_handler=None,
                                swapin_localx_handler=None,
                                swapin_msgx_handler=None,
                                swapout_stashx_handler=None,
                                swapout_localx_handler=None,
                                swapout_msgx_handler=None,
                                sucinfo=None):
        """ requires_grad == False: FWD (non-criterion)
            requires_grad == True: Recompute (for all) """
        is_last_ubatch = ubatch_idx == len(vt.ubatchszs)-1
        if not vt.has_criterion: # not last pack yet
            ### In {X}
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names
            if m.medium == "DAT": # Get one microbatch data
                # Data as X
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}Data)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                X_named_tensors = swp_x.swapin(data_ubatches[ubatch_idx])
                # print("\trank{}: task{}({}) SwapIn(#{}Data)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "P2P":
                if self.nvprof: nvtx_range_push("task{}({}) P2PIn(#{}X)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_p2p_prerecv:
                    X_named_tensors = self.p2px_handler.recv(self.XMETA.get(ubatch_size,l), src=m.rank)
                else:
                    X_named_tensors = self.p2px_handler.prerecv(self.XMETA.get(ubatch_size,l), src=m.rank, is_end=is_last_ubatch) 
                # print("\trank{}: task{}({}) P2PIn(#{}X)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "MSG": # message pass stashed input
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}StashX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_stashx:
                    X_named_tensors = swapin_stashx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_stashx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: task{}({}) SwapIn(#{}StashX)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}LocalX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_localx:
                    X_named_tensors = swapin_localx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_localx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: swp_recv'ed L{}-X".format(self.rank, l))
            else:
                raise NotImplementedError
            if self.nvprof: nvtx_range_pop() 
            ### Prefetch point @ FWD/Recompute (non-criterion)'s ULast 
            if is_last_ubatch:
                if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
                if self.p2px_handler is not None and \
                    not self.args.no_p2p_prerecv and not requires_grad:
                    self.p2px_handler.prerecv_suc(sucinfo.p2pin())
                # if swapin_msgx_handler is not None and \
                #     not self.args.no_prefetch_msgx and not requires_grad:
                #     swapin_msgx_handler.prefetch_suc(sucinfo.msgx())
                # if prefetch_model_handler is not None and \
                #     not self.args.no_prefetch_model and not requires_grad:
                #     prefetch_model_handler.iput(sucinfo.model()) 
                if swapin_stashx_handler is not None and \
                    not self.args.no_prefetch_stashx:
                    swapin_stashx_handler.prefetch_suc(sucinfo.stashx())
                # if swapin_stashx_handler is not None and \
                #     not self.args.no_prefetch_stashx and requires_grad:
                #     swapin_stashx_handler.prefetch_suc(sucinfo.stashx())
                if swapin_localx_handler is not None and \
                    not self.args.no_prefetch_localx and not requires_grad:
                    swapin_localx_handler.prefetch_suc(sucinfo.localx())
                if self.nvprof: nvtx_range_pop() 
            ### Compute forward pass on GPU
            if requires_grad:
                turn_on_X_grad(X_named_tensors) 
            if self.nvprof: nvtx_range_push("task{}({}) {}(#{})".format(vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx)) 
            Y_tensors = [X_named_tensors[name] for name in X_names]
            for l in vt.layers:
                if not requires_grad and l in vt.Out['X']: ### Out {stashX}
                    if self.nvprof: nvtx_range_push("task{}(L{}) SwapOut(#{}StashX)".format(vt.idx, l, ubatch_idx)) 
                    if vt.Out['X'][l].medium == "MSG": # message pass stashed X
                        swapout_stashx_handler.offload(l, 
                                make_tensors_named(self.local_model[l].X_names, Y_tensors))
                        # print("\trank{}: task{}(L{}) SwapOut(#{}StashX)".format(self.rank, vt.idx, l, ubatch_idx))
                    else:
                        raise NotImplementedError
                    if self.nvprof: nvtx_range_pop() 
                # print("\trank{}: task{}(L{}) {}".format(self.rank, vt.idx, l, "FWD" if not requires_grad else "Recompute"))
                Y_tensors = self.local_model[l](*Y_tensors)
                if not isinstance(Y_tensors, tuple):
                    Y_tensors = (Y_tensors,)
                Y_tensors = list(Y_tensors)
            if self.verbose: print("\trank{}: task{}({}) {} (#{})".format(self.rank, vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx))
            if self.nvprof: nvtx_range_pop() 
            ### Save Y
            l = vt.layers[-1]
            Y_names = self.local_model[l].Y_names
            Y_named_tensors = make_tensors_named(Y_names, Y_tensors)
            if not requires_grad:
                ### Out {Y}
                m = vt.Out['Y'][l]
                if m.medium == "P2P":
                    if self.nvprof: nvtx_range_push("task{}({}) P2POut(#{}Y)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                    self.p2px_handler.isend(Y_named_tensors, dst=m.rank)
                    # print("\trank{}: task{}({}) P2POut(#{}Y)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
                elif m.medium == "MSG": # last FWD convert to first BWD
                    if self.nvprof: nvtx_range_push("task{}({}) MSGOut(#{}Y)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                    swapout_msgx_handler.offload(l+1, Y_named_tensors)
                    # print("\trank{}: task{}({}) MSGOut(#{}Y)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
                elif m.medium == "SWP": # swap locally for vDP
                    if self.nvprof: nvtx_range_push("task{}({}) SwapOut(#{}Y)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                    if self.is_convert_ubs:
                        flag_is_convert = True if vt.is_last_fwd else False
                        swapout_localx_handler.offload(l+1, Y_named_tensors, flag_is_convert)
                    else:
                        swapout_localx_handler.offload(l+1, Y_named_tensors)
                    # print("\trank{}: swp_send'ed L{}-Y".format(self.rank, l))
                else:
                    raise NotImplementedError
                if self.nvprof: nvtx_range_pop() 
            ### Clean up
            if self.nvprof: nvtx_range_push("task{}({}) FWDClean(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            # print("\trank{}: task{}({}) FWDClean(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            del Y_tensors
            if not requires_grad:
                del X_named_tensors; del Y_named_tensors
            else: # for backward pass
                return X_named_tensors, Y_named_tensors
        else: # criterion pack
            assert requires_grad # fused forward and backward
            ### In {X}
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names
            if m.medium == "DAT": # a single BWD task
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}Data)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                X_named_tensors = swp_x.swapin(data_ubatches[ubatch_idx])
                # print("\trank{}: task{}({}) SwapIn(#{}Data)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "P2P": # the same above
                if self.nvprof: nvtx_range_push("task{}({}) P2PIn(#{}X)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_p2p_prerecv:
                    X_named_tensors = self.p2px_handler.recv(self.XMETA.get(ubatch_size,l), src=m.rank)
                else:
                    X_named_tensors = self.p2px_handler.prerecv(self.XMETA.get(ubatch_size,l), src=m.rank, is_end=is_last_ubatch) 
                # print("\trank{}: task{}({}) P2PIn(#{}X)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "MSG": # last FWD convert to first BWD
                if self.nvprof: nvtx_range_push("task{}({}) MSGIn(#{}X)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_msgx:
                    X_named_tensors = swapin_msgx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_msgx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: task{}({}) MSGIn(#{}X)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}LocalX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_localx:
                    X_named_tensors = swapin_localx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_localx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: swp_recv'ed L{}-X".format(self.rank, l))
            else:
                raise NotImplementedError
            if self.nvprof: nvtx_range_pop() 
            ### Prefetch point @ Recompute(criterion) ULast
            if is_last_ubatch:
                if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
                if self.p2px_handler is not None and \
                    not self.args.no_p2p_prerecv:
                    self.p2px_handler.prerecv_suc(sucinfo.p2pin())
                if swapin_stashx_handler is not None and \
                    not self.args.no_prefetch_stashx:
                    swapin_stashx_handler.prefetch_suc(sucinfo.stashx())
                if self.nvprof: nvtx_range_pop() 
            ### Recompute on GPU
            turn_on_X_grad(X_named_tensors)
            if self.nvprof: nvtx_range_push("task{}({}) Recompute(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            if len(vt.layers) > 1: # packed
                Y_tensors = [X_named_tensors[name] for name in X_names]
                for l in vt.layers[:-1]:
                    Y_tensors = self.local_model[l](*Y_tensors)
                    if not isinstance(Y_tensors, tuple):
                        Y_tensors = (Y_tensors,)
                    Y_tensors = list(Y_tensors)
                Y_names = self.local_model[vt.layers[-2]].Y_names
                Y_named_tensors = make_tensors_named(Y_names, Y_tensors)
            else: # only last vlayer
                Y_names = X_names
                Y_named_tensors = X_named_tensors
            ### In {T}
            if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}T)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            T_named_tensors = swp_x.swapin(target_ubatches[ubatch_idx])
            if self.nvprof: nvtx_range_pop() 
            ### Compute loss on GPU
            assert vt.layers[-1] == self.CONFIGS['R']-1
            last_vlayer = self.local_model[self.CONFIGS['R']-1]        
            if self.compute_loss is not None: # "bert_thomwolf", "gpt2_2bw", "gpt2_huggingface"
                Y_tensors = self.compute_loss(last_vlayer, Y_named_tensors, Y_names, T_named_tensors)
            else:
                Y_tensors = [last_vlayer(Y_named_tensors[name],T_named_tensors["target"]) for name in Y_names]
                Y_tensors = [sum(Y_tensors)]
            if self.verbose: print("\trank{}: task{}({}) Recompute(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            if self.nvprof: nvtx_range_pop() 
            ### Save Y
            Y_named_tensors = make_tensors_named(['loss'], Y_tensors)
            ### Clean up
            if self.nvprof: nvtx_range_push("task{}({}) FWDClean(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            # print("\trank{}: task{}({}) FWDClean(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            del T_named_tensors; del Y_tensors; 
            return X_named_tensors, Y_named_tensors

    def _a_pack_backward_an_ubatch(self, vt, ubatch_idx, ubatch_size,
                                X_named_tensors, Y_named_tensors,
                                swapin_localx_handler=None,
                                swapout_localx_handler=None,
                                sucinfo=None):
        is_last_ubatch = ubatch_idx == len(vt.ubatchszs) - 1
        ### In {dY}
        if vt.has_criterion:
            dY_named_tensors = ODict({ 'loss': None })
            assert Y_named_tensors['loss'].requires_grad
        else:
            l, m = vt.layers[-1], vt.In['dY'][vt.layers[-1]]
            dY_named_metas = make_dY_named_metas(self.XMETA, ubatch_size, l)
            if m.medium == "P2P":
                if self.nvprof: nvtx_range_push("task{}({}) P2PIn(#{}dY)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_p2p_prerecv:
                    dY_named_tensors = self.p2px_handler.recv(dY_named_metas, src=m.rank)
                else:
                    dY_named_tensors = self.p2px_handler.prerecv(dY_named_metas, src=m.rank, is_end=is_last_ubatch) 
                # print("\trank{}: task{}({}) P2PIn(#{}dY)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}dY)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_localx:
                    dY_named_tensors = swapin_localx_handler.fetch(l+1, dY_named_metas)
                else:
                    dY_named_tensors = swapin_localx_handler.prefetch(l+1, dY_named_metas, is_last_ubatch)
                # print("\trank{}: swp_recv'ed L{}-dY".format(self.rank, l))
            else:
                raise NotImplementedError
            if self.nvprof: nvtx_range_pop()                
        ### Prefetch point @ BWD's ULast
        if is_last_ubatch:
            if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
            if self.p2px_handler is not None and \
                not self.args.no_p2p_prerecv and not vt.has_criterion:
                self.p2px_handler.prerecv_suc(sucinfo.p2pin())
            if swapin_localx_handler is not None and \
                not self.args.no_prefetch_localx:
                swapin_localx_handler.prefetch_suc(sucinfo.localx())
            if self.nvprof: nvtx_range_pop() 
        ### Compute backward pass
        if self.nvprof: nvtx_range_push("task{}({}) BWD(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
        Y_tensors = []
        Y_gradients = [] 
        for name in self.local_model[vt.layers[-1]].Y_names: # only tensor & required_grad can run autograd
            Y = Y_named_tensors[name]
            if isinstance(Y,(torch.Tensor, Variable)) and (Y.requires_grad):
                Y_tensors.append(Y)
                Y_gradients.append(dY_named_tensors[name])
            elif isinstance(Y, list): # output tuple of bert pretrainheader
                for i, y in enumerate(Y):
                    if isinstance(y,(torch.Tensor, Variable)) and (y.requires_grad):
                        Y_tensors.append(y)
                        Y_gradients.append(dY_named_tensors[name][i])
        torch.autograd.backward(tuple(Y_tensors), grad_tensors=tuple(Y_gradients))
        if self.verbose: print("\trank{}: task{}({}) BWD(#{},{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx, ubatch_size))
        if self.nvprof: nvtx_range_pop() 
        ### Out {dX}
        if vt.Out['dX']:
            ### Save dX
            dX_named_tensors = make_dX_from_X(X_named_tensors) # ref to .grad
            l, m = vt.layers[0], vt.Out['dX'][vt.layers[0]] 
            if m.medium == "P2P":
                if self.nvprof: nvtx_range_push("task{}({}) P2POut(#{}dX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                self.p2px_handler.isend(dX_named_tensors, dst=m.rank)
                # print("\trank{}: task{}({}) P2POut(#{}dX)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapOut(#{}dX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.is_convert_ubs:
                    swapout_localx_handler.offload(l, dX_named_tensors, False)
                else:
                    swapout_localx_handler.offload(l, dX_named_tensors)
                # print("\trank{}: swp_send'ed L{}-dX".format(self.rank,l))
            else:
                raise NotImplementedError
            del dX_named_tensors; 
            if self.nvprof: nvtx_range_pop() 
        ### Clean up {X,Y,dX,dY}
        if self.nvprof: nvtx_range_push("task{}({}) BWDClean(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
        # print("\trank{}: task{}({}) BWDClean(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
        del X_named_tensors; del Y_named_tensors
        del dY_named_tensors; del Y_tensors; del Y_gradients

    def run_training_loop(self):
        local_losses = [] # per-minibatch
        global_losses = [] # per-minibatch
        if self.args.no_update: 
            grad_sums = [] # per-minibatch
        self.update_cnt = 0
        self.time_iters = []
        self.avg_it = int(self.args.num_iters * self.args.num_epochs /2.) # from this iter to average time
        
        ### clean memory before start
        torch.cuda.synchronize(self.rank); dist.barrier()
        gc.collect(); torch.cuda.empty_cache() 
        torch.cuda.synchronize(self.rank)
        assert torch.cuda.memory_reserved(self.rank)==0
        self.cgm = CheckGPUMem(self.rank)
        dist.barrier()
        print("rank%d: --- training starts ---" % self.rank)
        
        ### start
        self.rand_state_train.set()
        for epoch in range(self.args.num_epochs): # traverse epoches
            for it, minibatch in enumerate(self.data_loader): # traverse each minibatch
                if it >= self.args.num_iters:
                    break
                ### clean start
                gc.collect() 
                if self.args.empty_cache: torch.cuda.empty_cache() 
                torch.cuda.synchronize(self.rank)
                assert torch.cuda.memory_allocated(self.rank)==0, "iteration begins w/ alloc = {} B".format(torch.cuda.memory_allocated(self.rank)) 
                dist.barrier()
                if self.nvprof and it == self.args.nvprof_iter["start"]:
                    probe_cuda_mem = ProbeCudaMem(self.rank)
                    probe_cuda_mem.start()  
                    cuda_profiler.start()
                    nvtx_mark("cudaProfilerStart") 
                    print("rank%d: cuda profiler starts"%self.rank)
                else:
                    torch.cuda.reset_peak_memory_stats(self.rank) 
                time_start = pc() 
                ### data minibatch
                if self.args.synthetic_data:
                    data_ubatches, target_ubatches = self.data_ubatches, self.target_ubatches
                else:
                    if self.is_copy_minibatch: # "gpt2_huggingface"
                        minibatch = (minibatch, deepcopy(minibatch))
                    if self.is_skip_minibatch(minibatch, self.CONFIGS['D'], self.fdim, verbose=self.verbose): # skip fractional minibatch
                        assert (not self.nvprof) or (self.nvprof and it != self.args.nvprof_iter["end"]), "Unstoped Profiling"
                        continue
                    minibatch = self.preprocess_minibatch(minibatch) # preprocess as if single GPU
                    data_ubatches, target_ubatches = decompose_minibatch(minibatch, self.bnames, self.ubatchszs_fwd_local, self.ubatchszs_bwd_local, self.XMETA, self.TMETA, self.CONFIGS, self.rank, pin_memory=not self.args.no_pin_data) # make microbatches
                ### task starts    
                for j, vt in enumerate(self.rTASKS[self.rank]): # { rank0: [task0,task2,...] }
                    if self.verbose: print("\trank{}: executing {}".format(self.rank, vt))
                    self.sucinfo.set(vt, j)
                    if vt.type == 'FWD' and vt.is_gpu:
                        # -----------------------------------------------      
                        with torch.no_grad():
                            ### Swap-in model {W,B}
                            if self.nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                            if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                            suc_vt = None if self.args.no_prefetch_model else self.sucinfo.model()
                            cur_vt_idx = self.prefetch_model_handler.get(vt, suc_vt)
                            assert cur_vt_idx == vt.idx
                            if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                            if self.nvprof: nvtx_range_pop() 
                            ### Run through each microbatch in a data batch
                            for i, u in enumerate(vt.ubatchszs):
                                self._a_pack_forward_an_ubatch(vt, i, u,
                                        data_ubatches, target_ubatches, 
                                        requires_grad=False, 
                                        prefetch_model_handler=self.prefetch_model_handler,
                                        swapin_stashx_handler=self.swapin_stashx_handler,
                                        swapin_localx_handler=self.swapin_localx_handler,
                                        swapin_msgx_handler=self.swapin_msgx_handler,
                                        swapout_stashx_handler=self.swapout_stashx_handler,
                                        swapout_localx_handler=self.swapout_localx_handler,
                                        swapout_msgx_handler=self.swapout_msgx_handler,
                                        sucinfo=self.sucinfo)
                                if self.verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                                if self.nvprof: nvtx_range_pop() 
                                gc.collect()
                            ### Prefetch point @ FWD Del
                            self.default_stream.synchronize() # CPU wait Compute
                            if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
                            if not self.args.no_prefetch_model:
                                self.prefetch_model_handler.iput(suc_vt)
                            if self.swapin_msgx_handler is not None and not self.args.no_prefetch_msgx:
                                self.swapin_msgx_handler.prefetch_suc(self.sucinfo.msgx())
                            # if self.swapin_stashx_handler is not None and not self.args.no_prefetch_stashx:
                            #     self.swapin_stashx_handler.prefetch_suc(self.sucinfo.stashx())
                            if self.nvprof: nvtx_range_pop() 
                            ### Delete model {W,B}
                            if self.nvprof: nvtx_range_push("task{}({}) Del(W,B)".format(vt.idx, vt.show_layers())) 
                            for l in vt.layers:
                                if not (l in vt.Out['W']) and not (l in vt.Out['B']):
                                    self.local_model[l].del_param_grad_buf()
                                elif vt.Out['W'][l].medium=='PIN' and vt.Out['B'][l].medium=='PIN':
                                    pass
                                else: # P2P
                                    raise NotImplementedError
                            gc.collect()
                            if self.verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                            if self.nvprof: nvtx_range_pop() 
                        # -----------------------------------------------
                    elif vt.type == 'BWD' and vt.is_gpu:
                        # -----------------------------------------------
                        ### Swap-in model {W,B}
                        if self.nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                        if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                        suc_vt = None if self.args.no_prefetch_model else self.sucinfo.model()
                        cur_vt_idx = self.prefetch_model_handler.get(vt, suc_vt)
                        assert cur_vt_idx == vt.idx
                        if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                        if self.nvprof: nvtx_range_pop() 
                        ### Run through each microbatch in a data batch. 
                        m_loss = 0. # loss averaged across examples in this minibatch
                        for i, u in enumerate(vt.ubatchszs):
                            ### Recompute to create pytorch graph
                            X_named_tensors, Y_named_tensors = \
                                self._a_pack_forward_an_ubatch(vt, i, u,
                                                            data_ubatches, target_ubatches, 
                                                            requires_grad=True,
                                                            swapin_stashx_handler=self.swapin_stashx_handler,
                                                            swapin_localx_handler=self.swapin_localx_handler,
                                                            swapin_msgx_handler=self.swapin_msgx_handler,
                                                            sucinfo=self.sucinfo)
                            if self.nvprof: nvtx_range_pop() 
                            if 'loss' in Y_named_tensors:
                                Y_named_tensors['loss'] /= len(vt.ubatchszs) # NOTE: ubatches need to be equal
                                m_loss += Y_named_tensors['loss'].item()
                            ### Backward pass on recomputed graph
                            self._a_pack_backward_an_ubatch(vt, i, u,
                                                        X_named_tensors, Y_named_tensors,
                                                        swapin_localx_handler=self.swapin_localx_handler,
                                                        swapout_localx_handler=self.swapout_localx_handler,
                                                        sucinfo=self.sucinfo)
                            if self.verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                            if self.nvprof: nvtx_range_pop()
                            ### Clean up
                            del X_named_tensors; del Y_named_tensors # very important!
                            gc.collect()
                        ### Prefetch point @ AllReduce
                        if not self.args.no_prefetch_model:
                            if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
                            self.prefetch_model_handler.iput(suc_vt) 
                            if self.nvprof: nvtx_range_pop() 
                        ### Optional dW aggregation (and B sync)
                        if self.CONFIGS["mode"] == 'vDP' and self.CONFIGS['N'] > 1:
                            self.default_stream.synchronize() # TODO: wait Compute by cuda event 
                            if self.nvprof: nvtx_range_push("task{}({}) AllReduce(dW,B)".format(vt.idx, vt.show_layers())) 
                            for l in vt.layers:
                                self.local_model[l].average_grad(self.p2pm_handler)
                                if self.args.average_buffer:
                                    self.local_model[l].average_buf(self.p2pm_handler) # optional: all rank average buffers (can comment out to only use rank0's buf)
                            # TODO: wait AllReduce finish by cuda event
                            if self.nvprof: nvtx_range_pop() 
                        if m_loss != 0.:
                            local_losses.append(m_loss)
                            global_losses.append(m_loss)
                        ### Swap-out model {W,dW,B}
                        if self.CONFIGS["opt_offld"]:                   
                            # ### Clip dW for "gpt2_huggingface"
                            #     self.default_stream.synchronize()
                            #     for l in vt.layers:
                            #         torch.nn.utils.clip_grad_norm_(self.local_model[l].model.parameters(), self.args.max_grad_norm) 
                            ### Out {W,dW,B}
                            self.default_stream.synchronize() # CPU wait
                            if self.nvprof: nvtx_range_push("task{}({}) SwapOut(dW,B)".format(vt.idx, vt.show_layers())) 
                            for l in vt.layers:
                                if vt.Out['dW'][l].medium == "LOC" and vt.Out['B'][l].medium == "SHM":
                                    if self.CONFIGS["mode"]=='vPP' or (self.CONFIGS["mode"]=='vDP' and self.rank==0):
                                        self.local_model[l].swapout_grad() # Swap-out dW (accumulated)
                                        self.local_model[l].swapout_buf() # Swap-out B (updated)
                                else:
                                    raise NotImplementedError
                            if self.verbose: print_gpu_mem(self.rank, vt, "SwapOut'ed(dW,B)")    
                            if self.nvprof: nvtx_range_pop() 
                            ### Delete model {W,dW,B}
                            self.default_stream.synchronize() # CPU wait for SwapOut
                            if self.nvprof: nvtx_range_push("task{}({}) Del(W,dW,B)".format(vt.idx, vt.show_layers())) 
                            for l in vt.layers:
                                if vt.Out['dW'][l].medium == "LOC" and vt.Out['B'][l].medium == "SHM":
                                    self.local_model[l].del_param_grad_buf() # also del gradient
                                else: # 'B' == PIN
                                    raise NotImplementedError
                            gc.collect()
                            if self.verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                            if self.nvprof: nvtx_range_pop() 
                        else:
                            raise ValueError("GPU Optimizer Underdevelopment.")
                        # -----------------------------------------------
                    elif vt.type == 'UPD' and not vt.is_gpu:
                        # -----------------------------------------------
                        ### In {dW,W,K} Out {W,K}
                        if self.nvprof: nvtx_range_push("task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
                        if self.CONFIGS["mode"]=='vPP' or (self.CONFIGS["mode"]=='vDP' and self.rank==0):
                            if not self.args.no_update:
                                self.update_handler.iput(vt)
                        if self.nvprof: nvtx_range_pop() 
                        # -----------------------------------------------
                    else:
                        raise ValueError("Unknown vTask.type {} with .device {} !".format(vt.type,vt.device))
                ### tasks iteration ends
                if not self.args.no_update:
                    self.update_handler.synchronize()
                    self.update_cnt += 1
                torch.cuda.synchronize(self.rank)
                dist.barrier()
                ### statistics
                self.time_iters.append(pc()-time_start) 
                if self.nvprof and it == self.args.nvprof_iter["end"]:
                    nvtx_mark("cudaProfilerStop") 
                    cuda_profiler.stop()
                    probe_cuda_mem.stop()
                    print("rank%d: cuda profiler stops"%self.rank)
                ## if it % self.args.display_period == 0:
                ps = "rank%d: Epoch%d/%d Iter%d/%d %.3f sec, %.3f/%.3f GB" % ( 
                    self.rank, epoch, self.args.num_epochs, it, self.args.num_iters, 
                    self.time_iters[-1],
                    float(torch.cuda.memory_allocated()) / 1024**3,
                    float(torch.cuda.memory_reserved()) / 1024**3)
                if local_losses != []:
                    np.save(os.path.join(self.args.output_dir, "local_losses_rank%d.npy"%self.rank), local_losses)
                if self.CONFIGS["mode"] == 'vDP' and self.CONFIGS['N'] > 1:
                    global_losses[-1] = allreduce_cpu_loss(global_losses[-1], averaging=True)
                if self.rank == self.CONFIGS['loss_rank']:
                    ps += ", Loss %.3f"% global_losses[-1]
                    np.save(os.path.join(self.args.output_dir, "train_losses.npy"), global_losses)
                print(ps)
                if self.args.no_update:
                    assert self.CONFIGS["mode"] !='vPP'
                    gs = checker.check_grad_sum_harmony(self.shared_model)
                    grad_sums.append(gs)
                    np.save(os.path.join(self.args.output_dir, "grad_sums_rank%d.npy"%self.rank), grad_sums)
                # check GPU OoM & cudaFree & cudaMalloc
                self.cgm.check(it, is_check_malloc=not self.args.empty_cache and len(self.time_iters)-1 >= self.avg_it)
        ### end training
        torch.cuda.synchronize(self.rank)
        dist.barrier()
        print("rank%d: --- done ---" % self.rank)

    def finish(self): 
        ### statistics
        if self.verbose:
            print_p2p_bytes(self.rank, self.p2px_handler, self.p2pm_handler, self.update_cnt)
        #
        avg_iter_time = np.mean(self.time_iters[self.avg_it:]) # sec
        avg_throughput = self.CONFIGS['D'] / avg_iter_time # samples/sec
        gpu_reserved = gather_integer(torch.cuda.memory_reserved(), self.rank) # bytes
        if self.rank == 0:
            gpu_reserved = " ".join("%.1f"%(float(byte)/1024**3) for byte in gpu_reserved) # GB
            cpu_occupied = self.pcm.system_cpu_memory(["occupied"])
            print("[Global] Iter[%d,%d) Avg Iter Time: %.3f sec, Avg Throughput: %.3f sample/s, GPU: (%s) GB, CPU: %s, Num Updates: %d\n" % (self.avg_it, len(self.time_iters), avg_iter_time, avg_throughput, gpu_reserved, cpu_occupied, self.update_cnt))
        ### save model
        if self.args.save_final_model and self.rank == 0 and self.save_model is not None:
            self.save_model(self.args, self.shared_model, self.update_cnt)
