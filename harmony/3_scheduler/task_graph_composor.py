# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import OrderedDict as ODict
from copy import deepcopy

from task_data_struct import *

def compose_task_configs(args, u_fwd, pack_fwd, u_bwd, pack_bwd, verbose=False):
    CONFIGS = ODict()
    CONFIGS["R"] = args.num_layers
    CONFIGS["mode"] = args.mode
    CONFIGS["N"] = args.num_gpus
    CONFIGS["D"] = args.minibatchsize
    CONFIGS["u_fwd"] = u_fwd
    CONFIGS["ubatchszs_fwd"] = None # add from TASKS
    CONFIGS["pack_fwd"] = pack_fwd
    CONFIGS["u_bwd"] = u_bwd
    CONFIGS["ubatchszs_bwd"] = None # add from TASKS
    CONFIGS["pack_bwd"] = pack_bwd
    CONFIGS["reverse_bwd"] = args.reverse_bwd
    CONFIGS["opt_offld"] = args.offload_optim if hasattr(args, 'offload_optim') else not args.no_offload_optim
    CONFIGS["last_fwd_msg"] = args.last_fwd_msg
    CONFIGS['loss_rank'] = None # add from TASKS
    
    return CONFIGS

def split_minibatchsize(D, U):
    assert isinstance(D, int) and isinstance(U, int)
    assert D >= U
    if D % U == 0:
        ubatchszs = [U] * int(D/U)
    else:
        ubatchszs = [U] * int(D/U) + [ D%U ]
    assert sum(ubatchszs) == D
    
    return ubatchszs

def compose_task_graph(CONFIGS, verbose=False):
    """ use (u_fwd, pack_fwd, u_bwd, pack_bwd) to compose a task graph """
    R = CONFIGS["R"]
    mode = CONFIGS["mode"]
    N = CONFIGS["N"]
    D = CONFIGS["D"]
    u_fwd = CONFIGS["u_fwd"]
    pack_fwd = CONFIGS["pack_fwd"]
    u_bwd = CONFIGS["u_bwd"]
    pack_bwd = CONFIGS["pack_bwd"] 
    reverse_bwd = CONFIGS["reverse_bwd"]
    opt_offld = CONFIGS["opt_offld"]
    last_fwd_msg = CONFIGS["last_fwd_msg"]
    
    TASKS = []
    if mode == 'vPP':
        # ----- find microbatch sizes -----
        ubatchszs_fwd = split_minibatchsize(D, u_fwd)
        ubatchszs_bwd = split_minibatchsize(D, u_bwd)
        if verbose: print("ubatchszs_fwd={}, ubatchszs_bwd={}".format(ubatchszs_fwd, ubatchszs_bwd))
        # ----- create tasks from sized data and packed layers -----
        for a_pack in pack_fwd:
            vt = vTask( layers = a_pack, 
                        type = 'FWD', 
                        ubatchszs = ubatchszs_fwd )
            TASKS.append(vt)
        for a_pack in pack_bwd:
            vt = vTask( layers=a_pack, 
                        type='BWD',
                        ubatchszs = ubatchszs_bwd )
            TASKS.append(vt)
        for a_pack in pack_bwd:
            vt = vTask( layers=a_pack, 
                        type='UPD' )
            TASKS.append(vt)
        if verbose: print_tasks(TASKS, "created tasks (sized data & packed layers)")
        # ----- order tasks with jit update -----
        fwd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='FWD')
        fwd_tasks = sort_tasks_by_attr(fwd_tasks, attr='layers')
        bwd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='BWD')
        bwd_tasks = sort_tasks_by_attr(bwd_tasks, attr='layers', reverse=True)
        upd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='UPD')
        upd_tasks = sort_tasks_by_attr(upd_tasks, attr='layers', reverse=True)
        TASKS = []
        for vt in fwd_tasks:
            TASKS.append(vt)
        for vt1, vt2 in zip(bwd_tasks, upd_tasks):
            TASKS.append(vt1) 
            TASKS.append(vt2)
        for i, vt in enumerate(TASKS):
            vt.idx = i
        if verbose: print_tasks(TASKS, "ordered tasks (jit update)")
        # ----- place/bind tasks in round-robin -----
        # ascending round-robin for both fwd and bwd
        nfwd = len(fwd_tasks)
        for vt in TASKS: # ordered by idx
            if vt.type == 'FWD':
                vt.device = "GPU:%d" % ( vt.idx % N )
            elif vt.type == 'BWD':
                vt.device = "GPU:%d" % ( int(nfwd + (vt.idx-nfwd)/2) % N )
            elif vt.type == 'UPD':
                if opt_offld:
                    vt.device = "CPU:%d" % ( int(nfwd + (vt.idx-1-nfwd)/2) % N )
                else:
                    vt.device = "GPU:%d" % ( int(nfwd + (vt.idx-1-nfwd)/2) % N )
        if verbose: print_tasks(TASKS, "placed/bind tasks (round-robin fwd and bwd)")
        if reverse_bwd: # fwd: round-robin + bwd: reverse round-robin for jit bwd 
            for vt in TASKS: # ordered by idx
                if vt.type == 'BWD':
                    vt.set_new_rank( int(nfwd - (vt.idx-nfwd)/2) % N )
                    assert 0 <= vt.rank and vt.rank < N
                elif vt.type == 'UPD':
                    vt.set_new_rank( int(nfwd - (vt.idx-1-nfwd)/2) % N )
                    assert 0 <= vt.rank and vt.rank < N
            if verbose: print_tasks(TASKS, "placed/bind tasks (with reverse_bwd)")
        # ----- setup tasks' data dependency with p2p -----
        fwd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='FWD')
        bwd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='BWD')
        bwd_tasks_1st_layer = leave_first_layer_in_each_task(bwd_tasks)
        upd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='UPD')
        set_has_data_criterion(TASKS, R)
        set_the_last_fwd(TASKS)
        for vt in TASKS: # ordered by idx
            vt.layers = sorted(vt.layers)
            if vt.type == 'FWD':
                # In { 'X': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() } }
                if vt.has_data: # vt.idx == 0:
                    vt.In['X'] = ODict({vt.layers[0]:Medium('DAT')})
                    # # for DAT's dst T tasks
                    # T_tasks = filter_tasks_by_attr_val(bwd_tasks, attr='has_criterion', value=True)
                    # vt.In['X'][vt.layers[0]].set_for_T(T_tasks)
                elif TASKS[vt.idx-1].rank != vt.rank:
                    vt.In['X'] = ODict({vt.layers[0]:Medium('P2P',vt.idx-1,TASKS)})
                else: # swap locally
                    vt.In['X'] = ODict({vt.layers[0]:Medium('SWP',vt.idx-1)})
                vt.In['W'] = ODict()
                vt.In['B'] = ODict()
                for l in vt.layers:
                    vt.In['W'][l] = Medium("SHM")
                    vt.In['B'][l] = Medium("SHM")
                # Out { 'Y': { L1:Medium() }, 'X': { L1:Medium() }, 'W': {}, 'B': {} }
                if TASKS[vt.idx+1].rank != vt.rank:
                    vt.Out['Y'] = ODict({vt.layers[-1]:Medium('P2P',vt.idx+1,TASKS)})
                else: # swap locally
                    vt.Out['Y'] = ODict({vt.layers[-1]:Medium('SWP',vt.idx+1)})
                vt.Out['X'] = ODict()
                found = find_dependent_tasks_for_layers(bwd_tasks_1st_layer, vt.layers)
                for l, dst_idx in found.items():
                    vt.Out['X'][l] = Medium("MSG",dst_idx,TASKS)
                vt.Out['W'] = ODict()
                vt.Out['B'] = ODict()
            elif vt.type == 'BWD':
                if vt.has_criterion:
                    # In { 'dY':{}, 'InputX': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() }, 'T': {LLast:Medium()} }
                    vt.In['dY'] = ODict()
                    if vt.has_data: # a single BWD task
                        vt.In['X'] = ODict({vt.layers[0]:Medium('DAT')})
                        # # for DAT's dst T tasks
                        # vt.In['X'][vt.layers[0]].set_for_T([vt])
                    elif TASKS[vt.idx-1].rank != vt.rank:
                        vt.In['X'] = ODict({vt.layers[0]:Medium('P2P',vt.idx-1,TASKS)})
                    else: # swap locally
                        vt.In['X'] = ODict({vt.layers[0]:Medium('SWP',vt.idx-1)})
                else:
                    # In { 'dY': { L1:Medium() }, 'StashX': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() }, 'T': {} }
                    if TASKS[vt.idx-2].rank != vt.rank:
                        vt.In['dY'] = ODict({vt.layers[-1]:Medium('P2P',vt.idx-2,TASKS)})
                    else: # swap locally
                        vt.In['dY'] = ODict({vt.layers[-1]:Medium('SWP',vt.idx-2)})
                    vt.In['X'] = ODict()
                    found = find_dependent_tasks_for_layers(fwd_tasks, [vt.layers[0]])
                    for l, src_idx in found.items():
                        vt.In['X'][l] = Medium("MSG",src_idx,TASKS)
                vt.In['W'] = ODict()
                vt.In['B'] = ODict()
                for l in vt.layers:
                    vt.In['W'][l] = Medium("SHM")
                    vt.In['B'][l] = Medium("SHM")
                vt.In['T'] = ODict()
                if vt.has_criterion:
                    vt.In['T'][R-1] = Medium('DAT')
                #     D_task = filter_tasks_by_attr_val(fwd_tasks, attr='has_data', value=True)[0]
                #     vt.In['T'][R-1] = Medium('DAT',D_task.idx,TASKS)
                # Out { 'dX': { L0:Medium() }, 'dW': { L0:Medium(), L1:Medium() }, 'W': {}, 'B': { L0:[Medium(),Medium()], L1:Medium() } }
                if vt.has_data:
                    vt.Out['dX'] = ODict()
                elif TASKS[vt.idx+2].rank != vt.rank:
                    vt.Out['dX'] = ODict({vt.layers[0]:Medium('P2P',vt.idx+2,TASKS)})
                else: # swap locally
                    vt.Out['dX'] = ODict({vt.layers[0]:Medium('SWP',vt.idx+2)})
                vt.Out['dW'] = ODict()
                vt.Out['W'] = ODict()
                vt.Out['B'] = ODict()
                for l in vt.layers:
                    if opt_offld:
                        vt.Out['dW'][l] = Medium("LOC")
                    else:
                        vt.Out['dW'][l] = Medium("PIN",vt.idx+1)
                        vt.Out['W'][l] = Medium("PIN",vt.idx+1)
                    vt.Out['B'][l] = Medium("SHM")
            elif vt.type == 'UPD':
                # In { 'dW': { L0:Medium(), L1:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'K': { L0:Medium(), L1:Medium() } }
                vt.In['dW'] = ODict()
                vt.In['W'] = ODict()
                vt.In['K'] = ODict()
                for l in vt.layers:
                    if opt_offld:
                        vt.In['dW'][l] = Medium("LOC")
                        vt.In['W'][l] = Medium("SHM")
                    else:
                        vt.In['dW'][l] = Medium("PIN",vt.idx-1)
                        vt.In['W'][l] = Medium("PIN",vt.idx-1)
                    vt.In['K'][l] = Medium("SHM")
                # Out { 'W': { L0:[Medium(),Medium()], L1:Medium() }, 'K': { L0:Medium(), L1:Medium() } }
                vt.Out['W'] = ODict()
                vt.Out['K'] = ODict()
                for l in vt.layers:
                    vt.Out['W'][l] = Medium("SHM")
                    vt.Out['K'][l] = Medium("SHM")
            else:
                raise ValueError
        if verbose: print_tasks(TASKS, "setup data dependency") 
        # patch FWD-BWD JIT (PIN {W,B})
        found = find_jit_fwdbwd_pairs_in_one_iter(TASKS, N) # { 'GPU:0': [fwd_idx,[L1,L2],bwd_idx], ... } 
        global MediumMustMatchIdxRank, MediumMustMatchIdx
        MediumMustMatch = MediumMustMatchIdxRank+MediumMustMatchIdx
        for _, value in found.items():
            fwd_vt, bwd_vt = TASKS[value[0]], TASKS[value[2]] # ordered by idx
            for l in value[1]:
                # if: both fwd_vt and bwd_vt don't care MediumMatch
                # (elif: either fwd_vt or bwd_vt cares MediumMatch, but fwd_vt.medium is paired with bwd_vt MediumMatch
                #   then pin {W,B} from fwd_vt to bwd_vt)
                # else:
                #   raise error
                for key in ['W','B']:
                    if (not (l in fwd_vt.Out[key]) or not (fwd_vt.Out[key][l].medium in MediumMustMatch)) and (not (l in bwd_vt.In[key]) or not (bwd_vt.In[key][l].medium in MediumMustMatch)):
                        fwd_vt.Out[key][l] = Medium("PIN",bwd_vt.idx)
                        bwd_vt.In[key][l] = Medium("PIN",fwd_vt.idx)
                    else:
                        raise ValueError("Underdevelopment") 
        # patch replacing Last FWD's P2P(Y) to MSG
        if last_fwd_msg:
            last_fwd_tasks = filter_tasks_by_attr_val(TASKS, attr='is_last_fwd', value=True)
            if last_fwd_tasks == []: # a single BWD task
                CONFIGS["last_fwd_msg"] = False
            else:
                last_fwd_task = last_fwd_tasks[0]
                last_fwd_task = TASKS[last_fwd_task.idx] # ordered
                first_bwd_task = filter_tasks_by_attr_val(TASKS, attr='has_criterion', value=True)
                first_bwd_task = filter_tasks_by_attr_val(first_bwd_task, attr='type', value='BWD')[0]
                first_bwd_task = TASKS[first_bwd_task.idx] # ordered
                # replace the last fwd
                l = last_fwd_task.layers[-1]
                if last_fwd_task.Out['Y'][l].medium == 'P2P':
                    last_fwd_task.Out['Y'][l] = Medium("MSG", first_bwd_task.idx, TASKS)
                # replace the first bwd
                l = first_bwd_task.layers[0]
                if first_bwd_task.In['X'][l].medium == 'P2P':
                    first_bwd_task.In['X'][l] = Medium("MSG", last_fwd_task.idx, TASKS)
    elif mode == 'vDP':
        ubatchszs_fwd = []
        ubatchszs_bwd = []
        for n in range(N):
            per_gpu_tasks = [] # == per_rank_tasks
            # ----- find per-GPU microbatch sizes -----
            DD = int(float(D)/N)
            if D%N != 0: # uneven batch size across GPUs
                if n < D%N:
                    DD += 1
            ubszs_fwd = split_minibatchsize(DD, u_fwd)
            ubszs_bwd = split_minibatchsize(DD, u_bwd)
            ubatchszs_fwd.append(ubszs_fwd)
            ubatchszs_bwd.append(ubszs_bwd)
            if verbose: print("[GPU#{}] ubszs_fwd={}, ubszs_bwd={}".format(n, ubszs_fwd, ubszs_bwd))
            # ----- create tasks from sized data and packed layers -----
            for a_pack in pack_fwd:
                vt = vTask( layers = a_pack, 
                            type = 'FWD', 
                            ubatchszs = ubszs_fwd )
                per_gpu_tasks.append(vt)
            for a_pack in pack_bwd:
                vt = vTask( layers=a_pack, 
                            type='BWD',
                            ubatchszs = ubszs_bwd )
                per_gpu_tasks.append(vt)
            for a_pack in pack_bwd:
                vt = vTask( layers=a_pack, 
                            type='UPD' )
                per_gpu_tasks.append(vt)
            if verbose: print_tasks(per_gpu_tasks, "created tasks (sized data & packed layers) on GPU:%d"%n)
            # ----- order tasks with jit update -----
            fwd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='FWD')
            fwd_tasks = sort_tasks_by_attr(fwd_tasks, attr='layers')
            bwd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='BWD')
            bwd_tasks = sort_tasks_by_attr(bwd_tasks, attr='layers', reverse=True)
            upd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='UPD')
            upd_tasks = sort_tasks_by_attr(upd_tasks, attr='layers', reverse=True)
            per_gpu_tasks = []
            for vt in fwd_tasks:
                per_gpu_tasks.append(vt)
            for vt1, vt2 in zip(bwd_tasks, upd_tasks):
                per_gpu_tasks.append(vt1) 
                per_gpu_tasks.append(vt2)
            for i, vt in enumerate(per_gpu_tasks):
                vt.idx = len(TASKS) + i # n*len(per_gpu_tasks) + i
            if verbose: print_tasks(per_gpu_tasks, "ordered tasks (jit update) on GPU:%d"%n)
            # ----- place/bind tasks -----
            for vt in per_gpu_tasks:
                vt.device = "GPU:%d" % n
                if vt.type == 'UPD' and opt_offld:
                    vt.device = "CPU:%d" % n
            if verbose: print_tasks(per_gpu_tasks, "placed/bind tasks on GPU:%d"%n)
            # ----- setup tasks' data dependency -----
            fwd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='FWD')
            bwd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='BWD')
            bwd_tasks_1st_layer = leave_first_layer_in_each_task(bwd_tasks)
            upd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='UPD')
            set_has_data_criterion(per_gpu_tasks, R)
            set_the_last_fwd(per_gpu_tasks)
            for vt in per_gpu_tasks: # ordered by idx
                vt.layers = sorted(vt.layers)
                if vt.type == 'FWD':
                    # In { 'X': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() } }
                    if vt.has_data: # vt.idx == 0:
                        vt.In['X'] = ODict({vt.layers[0]:Medium('DAT')})
                        # # for DAT's dst T tasks
                        # T_tasks = filter_tasks_by_attr_val(bwd_tasks, attr='has_criterion', value=True)
                        # vt.In['X'][vt.layers[0]].set_for_T(T_tasks)
                    else: # swap locally
                        vt.In['X'] = ODict({vt.layers[0]:Medium('SWP',vt.idx-1)})
                    vt.In['W'] = ODict()
                    vt.In['B'] = ODict()
                    for l in vt.layers:
                        vt.In['W'][l] = Medium("SHM")
                        vt.In['B'][l] = Medium("SHM")
                    # Out { 'Y': { L1:Medium() }, 'X': { L1:Medium() }, 'W': {}, 'B': {} }
                    # swap locally
                    vt.Out['Y'] = ODict({vt.layers[-1]:Medium('SWP',vt.idx+1)})
                    vt.Out['X'] = ODict()
                    found = find_dependent_tasks_for_layers(bwd_tasks_1st_layer, vt.layers)
                    for l, dst_idx in found.items():
                        vt.Out['X'][l] = Medium("MSG",dst_idx,per_gpu_tasks)
                    vt.Out['W'] = ODict()
                    vt.Out['B'] = ODict()
                elif vt.type == 'BWD':
                    if vt.has_criterion:
                        # In { 'dY':{}, 'InputX': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() }, 'T': {LLast:Medium()} }
                        vt.In['dY'] = ODict()
                        if vt.has_data: # a single BWD task
                            vt.In['X'] = ODict({vt.layers[0]:Medium('DAT')})
                            # # for DAT's dst T tasks
                            # vt.In['X'][vt.layers[0]].set_for_T([vt])
                        else: # regular case
                            # swap locally
                            vt.In['X'] = ODict({vt.layers[0]:Medium('SWP',vt.idx-1)})
                    else:
                        # In { 'dY': { L1:Medium() }, 'StashX': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() }, 'T': {} }
                        # swap locally
                        vt.In['dY'] = ODict({vt.layers[-1]:Medium('SWP',vt.idx-2)})
                        vt.In['X'] = ODict()
                        found = find_dependent_tasks_for_layers(fwd_tasks, [vt.layers[0]])
                        for l, src_idx in found.items():
                            vt.In['X'][l] = Medium("MSG",src_idx,per_gpu_tasks)
                    vt.In['W'] = ODict()
                    vt.In['B'] = ODict()
                    for l in vt.layers:
                        vt.In['W'][l] = Medium("SHM")
                        vt.In['B'][l] = Medium("SHM")
                    vt.In['T'] = ODict()
                    if vt.has_criterion:
                        vt.In['T'][R-1] = Medium('DAT')
                        # if vt.has_data: # a single BWD task
                        #     vt.In['T'][R-1] = Medium('DAT',vt.idx,per_gpu_tasks)
                        # else: # regular case
                        #     D_task = filter_tasks_by_attr_val(fwd_tasks, attr='has_data', value=True)[0]
                        #     vt.In['T'][R-1] = Medium('DAT',D_task.idx,per_gpu_tasks)
                    # Out { 'dX': { L0:Medium() }, 'dW': { L0:Medium(), L1:Medium() }, 'W': {}, 'B': { L0:[Medium(),Medium()], L1:Medium() } }
                    if vt.has_data:
                        vt.Out['dX'] = ODict()
                    else: # swap locally
                        vt.Out['dX'] = ODict({vt.layers[0]:Medium('SWP',vt.idx+2)})
                    vt.Out['dW'] = ODict()
                    vt.Out['W'] = ODict()
                    vt.Out['B'] = ODict()
                    for l in vt.layers:
                        if opt_offld:
                            vt.Out['dW'][l] = Medium("LOC")
                        else:
                            vt.Out['dW'][l] = Medium("PIN",vt.idx+1)
                            vt.Out['W'][l] = Medium("PIN",vt.idx+1)
                        vt.Out['B'][l] = Medium("SHM")
                elif vt.type == 'UPD':
                    # In { 'dW': { L0:Medium(), L1:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'K': { L0:Medium(), L1:Medium() } }
                    vt.In['dW'] = ODict()
                    vt.In['W'] = ODict()
                    vt.In['K'] = ODict()
                    for l in vt.layers:
                        if opt_offld:
                            vt.In['dW'][l] = Medium("LOC")
                            vt.In['W'][l] = Medium("SHM")
                        else:
                            vt.In['dW'][l] = Medium("PIN",vt.idx-1)
                            vt.In['W'][l] = Medium("PIN",vt.idx-1)
                        vt.In['K'][l] = Medium("SHM")
                    # Out { 'W': { L0:[Medium(),Medium()], L1:Medium() }, 'K': { L0:Medium(), L1:Medium() } }
                    vt.Out['W'] = ODict()
                    vt.Out['K'] = ODict()
                    for l in vt.layers:
                        vt.Out['W'][l] = Medium("SHM")
                        vt.Out['K'][l] = Medium("SHM")
                else:
                    raise ValueError
            if verbose: print_tasks(per_gpu_tasks, "set data dependency on GPU:%d"%n)
            # ----- save per GPU tasks -----
            TASKS += per_gpu_tasks     
    else:
        raise ValueError("Underdevelopment")
    
    # check results
    for i, vt in enumerate(TASKS): # must be ordered by idx globally
        assert i == vt.idx
    if verbose: print_tasks(TASKS, "final tasks")
    # add extra configs
    CONFIGS['ubatchszs_fwd'] = ubatchszs_fwd
    CONFIGS['ubatchszs_bwd'] = ubatchszs_bwd
    loss_tasks = filter_tasks_by_attr_val(TASKS, attr='has_criterion', value=True)
    loss_tasks = filter_tasks_by_attr_val(loss_tasks, attr='type', value='BWD')
    CONFIGS['loss_rank'] = loss_tasks[0].rank
    if verbose: print("added CONFIGS[loss_rank]={}".format(CONFIGS['loss_rank']))
    
    return TASKS

def verify_basics(tasks, configs, verbose=False):
    assert isinstance(tasks, list)
    # Layers are in ascending order
    for vt in tasks:
        assert vt.layers == sorted(vt.layers, reverse=False)
    if verbose: print("Verified: layer ascending order.")
    # All packed layers == model (no missing, no double count, no extra layers)
    layers_correct = list(range(configs['R']))
    tasks_verify = filter_tasks_by_attr_val(tasks, attr='type', value='FWD')
    layers_verify = []
    for vt in tasks_verify:
        layers_verify += vt.layers
    bwd_tasks = filter_tasks_by_attr_val(tasks, attr='type', value='BWD')
    last_bwd_task = filter_tasks_by_attr_val(bwd_tasks, attr='has_criterion', value=True)[0]
    layers_verify += last_bwd_task.layers
    layers_verify.sort(reverse=False)
    assert layers_verify == layers_correct
    for t in ['BWD','UPD']:
        tasks_verify = filter_tasks_by_attr_val(tasks, attr='type', value=t)
        layers_verify = []
        for vt in tasks_verify:
            layers_verify += vt.layers
        layers_verify.sort(reverse=False)
        assert layers_verify == layers_correct
    if verbose: print("Verified: layers matching model.")
    # Within a task, In layers == Compute layers == Out layers
    for vt in tasks:
        if vt.type == 'FWD':
            assert vt.layers[0] in vt.In['X'] and len(vt.In['X'])==1 
            assert list(vt.In['W'].keys()) == vt.layers
            assert list(vt.In['B'].keys()) == vt.layers
            assert vt.layers[-1] in vt.Out['Y'] and len(vt.Out['Y'])==1
            for l in vt.Out['X'].keys(): # Empty layer skip
                assert l in vt.layers
            for l in vt.Out['W'].keys(): # Empty layer skip
                assert l in vt.layers
            for l in vt.Out['B'].keys(): # Empty layer skip
                assert l in vt.layers
        elif vt.type == 'BWD':
            # In
            if vt.has_criterion:
                assert not vt.In['dY']
                assert vt.layers[0] in vt.In['X'] and len(vt.In['X'])==1 
                assert list(vt.In['W'].keys()) == vt.layers
                assert list(vt.In['B'].keys()) == vt.layers
                assert vt.layers[-1]==configs['R']-1 and vt.layers[-1] in vt.In['T'] and len(vt.In['T'])==1
            else:
                assert vt.layers[-1] in vt.In['dY'] and len(vt.In['dY'])==1 
                assert vt.layers[0] in vt.In['X'] and len(vt.In['X'])==1 
                assert list(vt.In['W'].keys()) == vt.layers
                assert list(vt.In['B'].keys()) == vt.layers
                assert not vt.In['T']
            # Out
            if 0 in vt.layers:
                assert not vt.Out['dX'] # Empty Dict/OrderedDict is False
            else:
                assert vt.layers[0] in vt.Out['dX'] and len(vt.Out['dX'])==1 
            assert list(vt.Out['dW'].keys()) == vt.layers
            if configs['opt_offld']:
                assert not vt.Out['W'] # Empty Dict/OrderedDict is False
            else:
                assert list(vt.Out['W'].keys()) == vt.layers
            assert list(vt.Out['B'].keys()) == vt.layers
        elif vt.type == 'UPD':
            assert list(vt.In['dW'].keys()) == vt.layers
            assert list(vt.In['W'].keys()) == vt.layers
            assert list(vt.In['K'].keys()) == vt.layers
            assert list(vt.Out['W'].keys()) == vt.layers
            assert list(vt.Out['K'].keys()) == vt.layers
    if verbose: print("Verified: In layers == Compute layers == Out layers.")
    # In: each layer has single medium()
    # Out: each layer has single medium(), except {B,W} can have a list of mediums
    list_medium_allowed_in_Out = ["B","W"]
    for vt in tasks:
        for key, lm in vt.In.items():
            for l, m in lm.items():
                assert isinstance(m, Medium)
        for key, lm in vt.Out.items():
            for l, m in lm.items():
                if key in list_medium_allowed_in_Out:
                    if isinstance(m, list):
                        for mm in m:
                            assert isinstance(mm, Medium)    
                    else:
                        assert isinstance(m, Medium)
                else:
                    assert isinstance(m, Medium)
    if verbose: print("Verified: single and list Medium().")
    # Among tasks, Out medium() must be paired with In medium(), expect [DAT,SHM,LOC]
    # 1) selected_medium = ["P2P","MSG","SWP","PIN"]
    # 2) create a copy of tasks
    # 3) traverse original dtask
    #       traverse vt.Out
    #           traverse each layer
    #               if medium in selected_medium:
    #                   match medium to In
    #                   del the medium in both In and Out in copy tasks
    # 4) confirm no selected_medium left in copy tasks
    global MediumMustMatchIdxRank, MediumMustMatchIdx
    dtasks_copy = make_dtasks(deepcopy(tasks))
    for Out_vt in tasks:
        for Out_key, lm in Out_vt.Out.items():
            for Out_l, m in lm.items():
                Out_ms = [m] if isinstance(m,Medium) else list(m)
                for i, Out_m in enumerate(Out_ms):
                    if Out_m() in MediumMustMatchIdxRank+MediumMustMatchIdx:
                        In_vt = filter_tasks_by_attr_val(tasks, attr='idx', value=Out_m.idx)[0] # In_vt = dtasks[Out_m.idx]                        
                        if Out_key == 'Y':
                            if Out_l != configs['R']-1:
                                In_key = 'X'
                                In_l = Out_l+1
                            else:
                                In_key = 'dY'
                                In_l = Out_l    
                        elif Out_key == 'X': # stashX
                            In_key = 'X'
                            In_l = Out_l
                        elif Out_key == 'dX':
                            assert Out_l != 0
                            In_key = 'dY'
                            In_l = Out_l-1
                        elif Out_key in ['dW','W','B','K']: # W/B of fwd,bwd,upd
                            In_key = Out_key
                            In_l = Out_l
                        else:
                            raise ValueError("Unknown Out[{}]".format(Out_key))
                        # match Out medium to In medium
                        In_m = In_vt.In[In_key][In_l]
                        Out_m() == In_m()
                        Out_m.idx == In_vt.idx
                        In_m.idx == Out_vt.idx
                        if Out_m() in MediumMustMatchIdxRank:
                            Out_m.rank == In_vt.rank
                            In_m.rank == Out_vt.rank
                        # del the medium in both In and Out in copy tasks
                        dtasks_copy[In_vt.idx].In[In_key][In_l] = Medium()
                        if isinstance(m,Medium):
                            dtasks_copy[Out_vt.idx].Out[Out_key][Out_l] = Medium()
                        elif isinstance(m,list):
                            dtasks_copy[Out_vt.idx].Out[Out_key][Out_l][i] = Medium()
    # if verbose: print_tasks(dtasks_copy, name="dtasks_copy") 
    for match in MediumMustMatchIdxRank+MediumMustMatchIdx:
        filtered_tasks = filter_tasks_by_attr_val(list(dtasks_copy.values()), attr="medium", value=match)
        assert filtered_tasks == [], "'{}':remaining_tasks={}".format(match, filtered_tasks) 
    if verbose: print("Verified: Out Medium() pairing with In Medium().")
    # # T sharing matches (first layer X.DAT -> last layers's BWD) 
    # DAT_tasks = filter_tasks_by_attr_val(tasks, attr="medium", value='DAT')
    # D_tasks = filter_tasks_by_attr_val(DAT_tasks, attr="has_data", value=True)
    # T_tasks = filter_tasks_by_attr_val(DAT_tasks, attr="has_criterion", value=True)
    # for src in D_tasks:
    #     for dst_idx, dst_rank in zip(src.In['X'][0].dst_idx_T, src.In['X'][0].dst_rank_T):
    #         dst_task = filter_tasks_by_attr_val(T_tasks, attr="idx", value=dst_idx)[0]
    #         assert dst_idx == dst_task.idx
    #         assert dst_rank == dst_task.rank
    #         assert dst_task.In['T'][configs['R']-1].idx == src.idx
    #         assert dst_task.In['T'][configs['R']-1].rank == src.rank
    #         # del the matched dst_task
    #         del_i = None
    #         for i, vt in enumerate(T_tasks):
    #             if vt.idx == dst_task.idx:
    #                 del_i = i
    #                 break
    #         T_tasks.pop(del_i)
    # assert len(T_tasks) == 0
    # if verbose: print("Verified: T sharing matches.")

def verify_scheduled(rtasks, configs, verbose=False):
    """ Verify correctness of tasks before serving Runtime. """
    if verbose: print("\n----- Verification -----")
    if configs['mode'] == 'vPP':
        tasks = unmake_rank_based_task_queue(rtasks)
        verify_basics(tasks, configs)
    elif configs['mode'] == 'vDP':
        for r in range(configs['N']):
            # verfiy per gpu tasks
            verify_basics(rtasks[r], configs)
        # verify D == summed ubatchsizes of all GPU 
        Dfwd, Dbwd = 0, 0
        for r in range(configs['N']):
            fwd_tasks = filter_tasks_by_attr_val(rtasks[r], attr='type', value='FWD')
            bwd_tasks = filter_tasks_by_attr_val(rtasks[r], attr='type', value='BWD')
            Dfwd += sum(fwd_tasks[0].ubatchszs) if fwd_tasks != [] \
                    else sum(bwd_tasks[0].ubatchszs) # a single BWD task
            Dbwd += sum(bwd_tasks[0].ubatchszs)
        assert Dfwd == configs['D']
        assert Dbwd == configs['D']
    else:
        pass
    if verbose: print("Verification Succeeded.\n")

def verify_layer_packs(pack_fwd, pack_bwd, num_layers):
    assert isinstance(pack_fwd, list) and isinstance(pack_bwd, list)
    # All packed layers = model (no missing, no double count, no extra layers, ascend)
    layers_correct = list(range(num_layers))
    # check BWD first
    layers_bwd = []
    for p in pack_bwd:
        layers_bwd += p
    # layers_bwd.sort(reverse=False)
    assert layers_bwd == layers_correct
    # check FWD then
    layers_fwd = []
    for p in pack_fwd:
        layers_fwd += p
    layers_fwd += pack_bwd[-1]
    # layers_fwd.sort(reverse=False)
    assert layers_fwd == layers_correct    

def compose(args, u_fwd, pack_fwd, u_bwd, pack_bwd, verify=True, verbose=False):
    """ top-level function """
    """ generate a task graph from given four-tuple configuration """

    if verify: verify_layer_packs(pack_fwd, pack_bwd, args.num_layers)
    CONFIGS = compose_task_configs(args, u_fwd, pack_fwd, u_bwd, pack_bwd, verbose)
    TASKS = compose_task_graph(CONFIGS, verbose)
    rTASKS = convert_to_per_rank_task_queue(TASKS, args.num_gpus, verbose)
    if verify: verify_scheduled(deepcopy(rTASKS), deepcopy(CONFIGS), verbose)
    
    return CONFIGS, TASKS, rTASKS
