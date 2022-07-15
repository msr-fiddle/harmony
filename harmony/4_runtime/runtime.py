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
mp = torch.multiprocessing.get_context('spawn') # for GPU usage

def _assert_assumption(CONFIGS):
    # GPU only
    assert torch.cuda.is_available()
    # FP32 only
    torch.set_default_tensor_type('torch.FloatTensor')
    # Optimizer offload
    assert CONFIGS["opt_offld"]
    # double buffer need equal size
    if CONFIGS['mode'] == 'vPP':
        assert len(set(CONFIGS['ubatchszs_fwd'])) == 1
        assert len(set(CONFIGS['ubatchszs_bwd'])) == 1
    elif CONFIGS['mode'] == 'vDP': 
        for ubatchszs_fwd, ubatchszs_bwd in zip(CONFIGS['ubatchszs_fwd'], CONFIGS['ubatchszs_bwd']):
            assert len(set(ubatchszs_fwd)) == 1
            assert len(set(ubatchszs_bwd)) == 1
    else:
        raise ValueError
    # a single BWD task should have equal ubatchszs_fwd and ubatchszs_bwd per GPU
    if CONFIGS["pack_fwd"] == []: 
        assert CONFIGS["u_fwd"] == CONFIGS["u_bwd"]
        if CONFIGS['mode'] == 'vPP':
            assert CONFIGS['ubatchszs_fwd'] == CONFIGS['ubatchszs_bwd']
        elif CONFIGS['mode'] == 'vDP': 
            for ubatchszs_fwd, ubatchszs_bwd in zip(CONFIGS['ubatchszs_fwd'], CONFIGS['ubatchszs_bwd']):
                assert ubatchszs_fwd == ubatchszs_bwd
        else:
            raise ValueError

def worker_func(*pargs, **kwargs): # per process
    from worker import Worker
    w = Worker(*pargs, **kwargs)
    w.run_initial_iteration()
    w.run_training_loop()
    w.finish()
        
def run(args, real_dataset, create_model, create_optimizer, get_train_steps=None, get_lr_sched=None, compute_loss=None, save_model=None): # main process
    
    import seeding
    seeding.seed(args.seed, args.seed_cudnn)
    
    """ Initialize Harmony. """
    module_path = os.path.join(args.module_dir, args.module_name)
    assert os.path.exists(module_path)
    assert os.path.basename(module_path) not in ["prof", "sched"], "no base_dir in module_path"
    
    # read profiles
    from prof_data_struct import ConstMeta, TensorMeta, XMeta, TMeta, load_prof_data_struct
    prof = ODict()
    for name in args.profile_fnames:
        key = name.split("prof_")[-1]
        prof[key] = load_prof_data_struct(module_path, name + args.suffix, verbose=True)
    
    # read schedule
    from task_data_struct import Medium, vTask, unserialize_scheduled
    if args.schedule_dir == "":
        args.schedule_dir = module_path
    rTASKS, CONFIGS = unserialize_scheduled(args.schedule_dir, args.schedule_fname + args.suffix, verbose=False)
    _assert_assumption(CONFIGS)
    
    """ Initialize data. """
    if args.synthetic_data:
        args.num_epochs = 1
        assert args.num_iters is not None
        args.num_train_steps = args.num_iters
        print('----- Training Info -----')
        print("  num epoches = %d" % args.num_epochs)
        print("  num iterations per epoch = %d" % (args.num_iters))
        print("  num optimization steps = %d" % (args.num_train_steps))
    else:
        data_loader, examples, _, _, _, _, _ = real_dataset(args, CONFIGS["D"], data_workers=0)
        if get_train_steps is not None: # "bert_thomwolf"
            args.num_train_steps = get_train_steps(args, examples, CONFIGS["D"])
        else:
            args.num_train_steps = len(data_loader) * args.num_epochs
        if args.num_iters is None:
            args.num_iters = len(data_loader) # num_minibatch
        else:
            args.num_iters = min(args.num_iters, len(data_loader))
        print('----- Training Info -----')
        print("  num epoches = %d" % args.num_epochs)
        print("  num minibatches per epoch = %d" % len(data_loader))
        print("  num iterations per epoch = %d" % (args.num_iters))
        print("  num optimization steps = %d" % (args.num_train_steps))
        del data_loader
    
    if args.nvprof:
        assert args.num_epochs == 1, "num_epochs must be 1 during nvprof"
        if args.nvprof_iter == "first":
            args.nvprof_iter = { "start" : 0, "end" : 0 }
        elif args.nvprof_iter == "last":
            args.nvprof_iter = { "start" : args.num_iters - 1, "end" : args.num_iters - 1 }
        elif args.nvprof_iter == "all":
            args.nvprof_iter = { "start" : 0, "end" : args.num_iters - 1 } 
        else:
            raise ValueError

    """ Initialize model. """
    from utils import PrintCPUMem
    pcm = PrintCPUMem()
    pcm.print("before creating model")
    model = create_model(args)
    pcm.print("model created")
    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) != 0:
            for param in vlayer.parameters():
                assert not param.is_cuda
    
    # initialize empty model on CPU
    from local_model_gpu import delete_param_grad_buf
    empty_model = []
    for vlayer, _, _ in model:
        with torch.no_grad():
            vlayer_copy = deepcopy(vlayer)
        delete_param_grad_buf(vlayer_copy)
        empty_model.append(vlayer_copy)
    
    # initialize shared model on CPU  
    for vlayer, _, _ in model:
        vlayer.share_memory() # move parameter into shared memory    
    pcm.print("shared model created")

    """ Initialize optimizer. """
    optimizer = create_optimizer(args, model)
    pcm.print("optimizer created")
    
    # initialize shared optimizer on CPU
    from shared_optim_cpu import SharedOptimCPU
    shared_model = model # model is already shared
    shared_optimizer = [] # wrapper object for optimizer
    for id, ((vlayer, _, _), optim) in enumerate(zip(shared_model, optimizer)): 
        shared_optimizer.append(SharedOptimCPU(vlayer, optim, id))
    pcm.print("shared optimizer created")

    """ Initialize distributed training. """ 
    gc.collect(); torch.cuda.empty_cache() 
    assert torch.cuda.memory_reserved() == 0, "fork process begins w/ alloc = {} B".format(torch.cuda.memory_reserved()) 
    
    processes = []
    if args.numa_bind:
        from utils import NumaBinder
        numa_binder = NumaBinder(args.numa_bind_config)
    for rank in range(CONFIGS["N"]):
        p = mp.Process(target=worker_func, 
                        args=(args, real_dataset, shared_model, shared_optimizer, empty_model, get_lr_sched, compute_loss, save_model, prof['XMETA'], prof['TMETA'], rTASKS, CONFIGS, rank),
                        name="rank%d"%rank)
        # NOTE: this moves parameter from pinned memory to shared memory
        p.start()
        processes.append(p)
        if args.numa_bind:
            numa_binder.bind(p, rank)
        
    if args.nvprof:
        from viewer.probe_cpu import ProbeCPU
        probe_cpu = ProbeCPU(pids=[p.pid for p in processes], 
                            ranks=[rank for rank in range(CONFIGS["N"])])
        probe_cpu.run(processes[0])
        print("--- rank -1: Done ---")
        print("--- all pids = (%s) ---"% " ".join("%d"%pid for pid in list([os.getpid()]+[p.pid for p in processes])) )

    for p in processes:
        p.join()
    print("--- all workers joined successfully. ---")
