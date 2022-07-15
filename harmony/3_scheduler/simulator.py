# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict as ODict
import time

from task_data_struct import Medium, vTask, unserialize_scheduled, unmake_rank_based_task_queue
from sim_data_struct import *
from sim_chrome_trace import save_to_chrome_trace

def _assert_assumption(CONFIGS):
    assert CONFIGS['opt_offld'] 
    """ - Ignore T """

def create_a_task_events(mode, num_gpus, vt, events, stream_events, left_vt, left2_vt, ubscvt, TASKS, prefetch_offload):
    if mode == 'vPP':
        if vt.type == "FWD" and not vt.has_criterion:
            ev_w = In_WB(vt, events, stream_events, left_vt, False, prefetch_offload)
            for i, u in enumerate(vt.ubatchszs):
                ev_x = In_X(vt, i, events, stream_events, left_vt, left2_vt, TASKS, None, None, prefetch_offload, ev_w if i == 0 else ev_fwd)
                ev_fwd = Compute_FWD(vt, i, events, stream_events, ev_w, ev_x, prefetch_offload)
                ev_msg = Out_sX(vt, i, u, events, stream_events, left_vt, ubscvt, ev_x, prefetch_offload)
                ev_y = Out_Y(vt, i, u, events, stream_events, ubscvt, TASKS, ev_fwd)
            Compute_DEL(vt, events, stream_events, None, prefetch_offload, None)
        elif vt.type == "BWD" and vt.has_criterion:
            ev_w = In_WB(vt, events, stream_events, left_vt, False, prefetch_offload)
            for i, _ in enumerate(vt.ubatchszs):
                ev_x = In_X(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ubscvt, None, prefetch_offload, ev_w if i == 0 else ev_bwd)
                ev_rec, ev_bwd = Compute_BWD(vt, i, events, stream_events, ev_w, ev_x, None)
                ev_dx = Out_dX(vt, i, events, stream_events, TASKS, ev_bwd)
            ev_dw = Out_dWB(vt, events, stream_events, ev_bwd, None, prefetch_offload, None)
            Compute_DEL(vt, events, stream_events, ev_dw, prefetch_offload, None)
        elif vt.type == "BWD" and not vt.has_criterion:
            ev_w = In_WB(vt, events, stream_events, left_vt, True, prefetch_offload)
            src_vt, src_spi = find_stash_subpack_idx(vt, TASKS)
            for i, _ in enumerate(vt.ubatchszs):
                ev_sx = In_sX(vt, i, events, stream_events, left_vt, left2_vt, src_vt, src_spi, prefetch_offload, ev_w if i == 0 else ev_bwd)
                ev_dy = In_dY(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ev_w, False, prefetch_offload)
                ev_rec, ev_bwd = Compute_BWD(vt, i, events, stream_events, ev_w, ev_sx, ev_dy)
                ev_dx = Out_dX(vt, i, events, stream_events, TASKS, ev_bwd)
            ev_dw = Out_dWB(vt, events, stream_events, ev_bwd, None, prefetch_offload, None)
            Compute_DEL(vt, events, stream_events, ev_dw, prefetch_offload, None)
        elif vt.type == "UPD":
            CPU_Update(vt, events, stream_events, left_vt)
        else:
            raise ValueError
    elif mode == 'vDP':
        if vt.type == "FWD" and not vt.has_criterion:
            ev_w = In_WB(vt, events, stream_events, left_vt, True, prefetch_offload)
            for i, u in enumerate(vt.ubatchszs):
                ev_x = In_X(vt, i, events, stream_events, left_vt, left2_vt, TASKS, None, ev_w, prefetch_offload, ev_w if i == 0 else ev_y)
                ev_fwd = Compute_FWD(vt, i, events, stream_events, ev_w, ev_x, prefetch_offload)
                ev_msg = Out_sX(vt, i, u, events, stream_events, left_vt, ubscvt, ev_x, prefetch_offload)
                ev_y = Out_Y(vt, i, u, events, stream_events, ubscvt, TASKS, ev_fwd)
            Compute_DEL(vt, events, stream_events, None, prefetch_offload, ev_y)
        elif vt.type == "BWD" and vt.has_criterion:
            ev_w = In_WB(vt, events, stream_events, left_vt, True, prefetch_offload)
            for i, _ in enumerate(vt.ubatchszs):
                ev_x = In_X(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ubscvt, ev_w, prefetch_offload, ev_w if i == 0 else ev_dx)
                ev_rec, ev_bwd = Compute_BWD(vt, i, events, stream_events, ev_w, ev_x, None)
                ev_dx = Out_dX(vt, i, events, stream_events, TASKS, ev_bwd)
            ev_ard = Compute_ARD(vt, events, stream_events, num_gpus, prefetch_offload, ev_dx)
            ev_dw = Out_dWB(vt, events, stream_events, ev_bwd, ev_ard, prefetch_offload, ev_dx)
            Compute_DEL(vt, events, stream_events, ev_dw, prefetch_offload, None)
        elif vt.type == "BWD" and not vt.has_criterion:
            ev_w = In_WB(vt, events, stream_events, left_vt, True, prefetch_offload)
            src_vt, src_spi = find_stash_subpack_idx(vt, TASKS)
            for i, _ in enumerate(vt.ubatchszs):
                ev_sx = In_sX(vt, i, events, stream_events, left_vt, left2_vt, src_vt, src_spi, prefetch_offload, ev_w if i == 0 else ev_dx)
                ev_dy = In_dY(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ev_w, True, prefetch_offload)
                ev_rec, ev_bwd = Compute_BWD(vt, i, events, stream_events, ev_w, ev_sx, ev_dy)
                ev_dx = Out_dX(vt, i, events, stream_events, TASKS, ev_bwd)    
            ev_ard = Compute_ARD(vt, events, stream_events, num_gpus, prefetch_offload, ev_dx)
            ev_dw = Out_dWB(vt, events, stream_events, ev_bwd, ev_ard, prefetch_offload, ev_dx)
            Compute_DEL(vt, events, stream_events, ev_dw, prefetch_offload, None)
        elif vt.type == "UPD":
            CPU_Update(vt, events, stream_events, left_vt)
        else:
            raise ValueError
    else:
        raise ValueError

def simulate(args, rTASKS, CONFIGS, TASKS=None, prefetch_offload=True, verbose=True, view=True):
    """ top-level function """
    """ estimate runtime of given task graph by an event-driven simulation """

    _assert_assumption(CONFIGS)
    TASKS = unmake_rank_based_task_queue(rTASKS) if TASKS is None else TASKS
    if CONFIGS['mode'] == 'vPP':
        ubatchszs_fwd = CONFIGS['ubatchszs_fwd']
        ubatchszs_bwd = CONFIGS['ubatchszs_bwd']
    elif CONFIGS['mode'] == 'vDP':
        ubatchszs_fwd = CONFIGS['ubatchszs_fwd'][0]
        ubatchszs_bwd = CONFIGS['ubatchszs_bwd'][0]
    else:
        raise ValueError
    if CONFIGS["u_fwd"] != CONFIGS["u_bwd"]:
        ubscvt = UBSConverter(ubatchszs_fwd, ubatchszs_bwd, CONFIGS['u_bwd'], verbose)
    else:
        ubscvt = None
        assert ubatchszs_fwd == ubatchszs_bwd
    
    if CONFIGS['mode']=='vPP' and CONFIGS['N']>1:
        sim_mode = 'vPP'
    elif (CONFIGS['mode']=='vPP'and CONFIGS['N'] == 1) or CONFIGS['mode']=='vDP':
        sim_mode = 'vDP' # """ treat as a single GPU """
    else:
        raise NotImplementedError
    
    non_empty_gpus = sum([tasks != [] for tasks in rTASKS.values()])
    res = ODict()

    # ----------------------- Time --------------------------------
    if verbose: t_start = time.time() 
    ### make events with dependency
    events = ODict() # { id: Event() } # vPP for all ranks, vDP for rank0
    rank_stream_events = ODict() # { rank: { stream: [Event()] } or {} } 
    for rank, tasks in rTASKS.items(): # { rank: [Task()] or [] }
        if sim_mode == 'vDP' and rank != 0:
            break
        rank_stream_events[rank] = ODict() # { stream: [Event()] } or {}
        left_vt, left2_vt = None, None    
        for vt in tasks:
            create_a_task_events(sim_mode, non_empty_gpus, vt, events, rank_stream_events[rank], left_vt, left2_vt, ubscvt, TASKS, prefetch_offload)
            if vt.type in ['FWD', 'BWD']:
                left2_vt = left_vt
                left_vt = vt
    for ev in events.values(): # convert remaining input ids to events
        ev.inputs = [inev if isinstance(inev, Event) else events[inev] 
                        for inev in ev.inputs]
    for ev in events.values(): # add p2p dependency
        ev.solve_peer(events, rank_stream_events)
    if verbose: 
        print_events(events) 
        print_rank_stream_events(rank_stream_events)
    # if sim_mode == 'vPP': debug(non_empty_gpus, events, rank_stream_events)
    ### dispatch events for execution
    dispatcher = Dispatcher(rank_stream_events)
    executor = Executor(args, non_empty_gpus, CONFIGS, TASKS, rank_stream_events)
    if verbose:
        print("=== Dispatching: %d Streams, %d Events ==="%(
                dispatcher.num_streams, len(events.keys()) ))
    while True:
        ev = dispatcher.dispatch()
        if isinstance(ev, Event):
            executor.execute(ev)
            if verbose: print("Executed: %s"%(ev))
        elif isinstance(ev, list):
            res['global_endtime'] = float('inf')
            res['per_rank_endtime'] = [float('inf')] if sim_mode == 'vDP' else \
                                      [float('inf')] * len(rTASKS.keys())
            res['max_endidle'] = float('inf')
            res['avg_endidle'] = float('inf')
            res['avg_compute_to_globaltime'] = float('inf')
            res['avg_compute_to_ranktime'] = float('inf')
            print("=== Deadlock! after executed %d events. stop here: ==="%(executor.cnt))
            for e in ev: 
                print("\t%s" % (e))
            break
        elif ev == "done":
            executor.end()
            res['global_endtime'] = executor.global_endtime
            res['per_rank_endtime'] = executor.per_rank_endtime
            res['max_endidle'] = executor.max_endidle
            res['avg_endidle'] = executor.avg_endidle
            res['avg_compute_to_globaltime'] = executor.avg_compute_to_globaltime
            res['avg_compute_to_ranktime'] = executor.avg_compute_to_ranktime
            if verbose:
                t_end = time.time() 
                print("=== Simulation Done ===")
                print("Global End Time: %.2f sec (%s)"%(
                        res['global_endtime'], 
                        ", ".join("%.2f"%et for et in res['per_rank_endtime']) ))
                print("Max/Avg End Idle: %.0f%% / %.0f%%"%(
                        res['max_endidle']*100., 
                        res['avg_endidle']*100.))
                print("Compute2Global/Compute2Rank: %.0f%% / %.0f%%"%(
                        res['avg_compute_to_globaltime']*100., 
                        res['avg_compute_to_ranktime']*100.))
                print("Time Cost: %.3f sec (%d Streams, %d Events)"%   
                        (t_end-t_start, dispatcher.num_streams, executor.cnt))
            if view: 
                save_to_chrome_trace(args, CONFIGS, events)
            break
        else:
            raise ValueError
    
    # ----------------------- Memory --------------------------------
    if verbose:
        print("=== Estimating Memory ===")
        t_start = time.time() 
    C_of_task = CofTask(args.prof, sim_mode, CONFIGS['R'])
    per_rank_memory = []
    for rank, tasks in rTASKS.items(): # { rank: [Task()] or [] }
        if sim_mode == 'vDP' and rank != 0:
            break
        if tasks == []:
            max_mem = 0.
        else:
            max_mem = max([C_of_task(vt) for vt in tasks])
        per_rank_memory.append(max_mem) # bytes
    res['global_memory'] = max(per_rank_memory)
    res['per_rank_memory'] = per_rank_memory
    if verbose:
        t_end = time.time() 
        print("Memory: %.0f MB (%s)"% (
                res['global_memory']/1024./1024., 
                ", ".join("%.0f"%(m/1024./1024.) for m in res['per_rank_memory']) ) )
        print("Time Cost: %.3f sec"%(t_end-t_start))
    # ---------------------------------------------------------------
    
    return res

def sim_res_str(res, title="simulated"):
    return "=== %s : %.3f sec (idle max/avg: %.0f%%/%.0f%%) (compute2global/compute2rank: %.0f%%/%.0f%%), %.0f MB (%s) ==="% (
        title, 
        res['global_endtime'], 
        res['max_endidle']*100., 
        res['avg_endidle']*100., 
        res['avg_compute_to_globaltime']*100.,  
        res['avg_compute_to_ranktime']*100.,
        res['global_memory']/1024./1024.,
        ", ".join("%.0f"%(m/1024./1024.) for m in res['per_rank_memory']) )
