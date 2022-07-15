# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import json
import numpy as np
from time import perf_counter as pc
from collections import OrderedDict as ODict
import sys

from layer_packing import greedy_memory_packing, balanced_memory_packing, reuse_memory_packing, balanced_time_packing, manual_pack
from task_graph_composor import compose
from simulator import simulate, sim_res_str

class TopK(object):
    def __init__(self, k=1, hash_fn=None, print_fn=None):
        """ hash_fn: if None, use counter as id (can have duplicate configs); 
                     else, use hash value as id (de-duplicated configs) """
        self.k = k if k > 0 else float('inf')
        self.roster = ODict() # { id: { "time" : 1., "config" : ODict } }
        self.cnt = 0
        self.hash_fn = hash_fn
        self.print_fn = print_fn 
        if self.print_fn is None: 
            self.print_fn = lambda r, t, c: print("Top{}: time {}, config {}".format(r + 1, t, c))

    def add(self, time, config):
        """ rank topK based on this argument time """
        id = self.cnt if self.hash_fn is None else self.hash_fn(config)
        if self.cnt < self.k: # topK not full yet
            if id not in self.roster:
                self.roster[id] = ODict({"time":time, "config":config})
                self.cnt += 1      
        else: # topK is full
            # find the worst one
            topk_id = [key for key in self.roster.keys()]
            topk_time = [v["time"] for v in self.roster.values()]
            worst_id = topk_id[ topk_time.index(max(topk_time)) ]
            # if in topK and unique, replace the worst one
            if time < self.roster[worst_id]['time'] and (id not in self.roster):
                del self.roster[worst_id]
                self.roster[id] = ODict({"time":time, "config":config})
                self.cnt += 1
    
    def summary(self, title=""):
        print("%s" % title)
        
        topk_time = [v["time"] for v in self.roster.values()]
        topk_config = [v["config"] for v in self.roster.values()]
        indices = np.argsort(topk_time) # returning index of a sorted list (ascending)
        sorted_config, sorted_print = [], []
        for r, i in enumerate(indices):
            time = topk_time[i]
            config = topk_config[i]
            s = self.print_fn(r, time, config)
            sorted_config.append(config)
            sorted_print.append(s)
        
        return sorted_config, sorted_print

def hash_fn(config):
    u_fwd = int(config['CONFIGS']['u_fwd'])
    pack_fwd = config['CONFIGS']['pack_fwd']
    u_bwd = int(config['CONFIGS']['u_bwd'])
    pack_bwd = config['CONFIGS']['pack_bwd']
    # e.g., pack_fwd = []
    # e.g., pack_fwd = [[0, 1, 2, 3, 4...12, 13, 14, 15, 16, 17, 18]]
    # e.g., pack_fwd = [[0, 1, 2, 3], ..., [16, 17, 18, 19, 20]]
    pack_fwd = tuple( tuple(p) for p in pack_fwd )
    pack_bwd = tuple( tuple(p) for p in pack_bwd )
    # hash the immutables
    
    return hash((u_fwd,pack_fwd,u_bwd,pack_bwd))

def print_fn(r, time, config):
    # print_str1 = print_global(config["end2end_times"], config["end2end_memories"], title="Top%d"%(r+1))
    print_str1 = sim_res_str(config["res"], title="Top%d"%(r+1))
    print(print_str1)
    
    u_fwd = config['CONFIGS']['u_fwd']
    pack_fwd = config['CONFIGS']['pack_fwd']
    u_bwd = config['CONFIGS']['u_bwd']
    pack_bwd = config['CONFIGS']['pack_bwd']
    print_str2  = "\tu_fwd   : {}\n".format(u_fwd)
    print_str2 += "\tpack_fwd: {} =\t{}\n".format(len(pack_fwd), pack_fwd)
    print_str2 += "\t{}\n".format(config["packing_method_fwd"])
    print_str2 += "\tu_bwd   : {}\n".format(u_bwd)
    print_str2 += "\tpack_bwd: {} =\t{}\n".format(len(pack_bwd), pack_bwd)
    print_str2 += "\t{}".format(config["packing_method_bwd"])
    print(print_str2)
    
    return print_str1 + "\n" + print_str2
    # print("\tu_fwd   : {}".format(u_fwd))
    # print("\tpack_fwd: {}x =\t{}".format(len(pack_fwd), pack_fwd))
    # print("\t{}".format(config["packing_method_fwd"]))
    # print("\tu_bwd   : {}".format(u_bwd))
    # print("\tpack_bwd: {}x =\t{}".format(len(pack_bwd), pack_bwd))
    # print("\t{}".format(config["packing_method_bwd"]))  

def is_equal_ubatchsize(args, ubatchsize):
    """ for both Ufwd and Ubwd """
    D, N = args.minibatchsize, args.num_gpus
    if args.mode == 'vDP':
        is_equal = True
        for n in range(N):
            # ----- find per-GPU microbatch sizes -----
            DD = int(float(D)/N)
            if D%N != 0: # uneven batch size across GPUs
                if n < D%N:
                    DD += 1
            assert DD >= ubatchsize
            if DD % ubatchsize != 0:
                is_equal = False
                break
        return is_equal
    elif args.mode == 'vPP':
        assert D >= ubatchsize
        return D % ubatchsize == 0
    else:
        raise ValueError

def find_ubatchsizes(args):
    """ find valid ubatch sizes to search """
    # ubatchsize = 1 ~ min(Umax,min(DD)) for vDP both fwd and bwd
    # ubatchsize = 1 ~ min(Umax,D) for vPP both fwd and bwd
    Umin_fwd, Umax_fwd = min(args.fwd_ubatchsizes), max(args.fwd_ubatchsizes)
    Umin_bwd, Umax_bwd = min(args.bwd_ubatchsizes), max(args.bwd_ubatchsizes)
    D, N = args.minibatchsize, args.num_gpus    
    if args.mode == 'vDP':
        D = int(float(D)/N) # namely, min(DD)
    elif args.mode == 'vPP':
        D = D
    else:
        raise ValueError
    ubatchsizes_fwd = list(range(Umin_fwd, min(Umax_fwd, D)+1, args.ubatchsize_step))
    ubatchsizes_bwd = list(range(Umin_bwd, min(Umax_bwd, D)+1, args.ubatchsize_step))
    
    # then select equal ubatchsize if needed (DD/D % ubatchsize == 0)
    if not args.inequal_ubatchsize:
        ubatchsizes_fwd = [u for u in ubatchsizes_fwd if is_equal_ubatchsize(args, u)]
        ubatchsizes_bwd = [u for u in ubatchsizes_bwd if is_equal_ubatchsize(args, u)]
    else:
        print("[WARNING] allow inequal microbatchsize will disable double buffering. Although we can still search, runtime is not supported currently.")
    
    return sorted(ubatchsizes_fwd), sorted(ubatchsizes_bwd)

def find_layer_packs(args, ubatchsize, num_layers, type, reuse_packs=None, verbose=True):
    """ for an ubatchsize, find valid layer packs to search """
    assert type in ["FWD","BWD"]
    
    # build per_layer_memories list
    memory_list = []; x_list = []
    for l in range(num_layers):
        mem  = args.prof['MEMORY_FWDBWD'].get(type, ubatchsize, l, interp=True) # int
        xmem = args.prof["XMETA"].get_bytes(ubatchsize, l, interp=True)# int
        memory_list.append(mem - xmem) # bytes 
        x_list.append(xmem) # bytes
    
    # different packing methods
    layer_packs = ODict() # { "greedy" : [[0,1,2], [3,4,5]] }
    packing_method = args.packing_method_fwd if type == 'FWD' else \
                     args.packing_method_bwd
    tab = "\t\t\t" if type == 'FWD' else "\t"
    for method in packing_method:
        if method == "greedy":
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, verbose=verbose, tab=tab)
        elif method == "greedy_addx":
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, per_layer_x=x_list, verbose=verbose, title="greedy_memory_packing (addx)", tab=tab)
        elif method == "greedy_reverse":
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, reverse=True, verbose=verbose,title="greedy_memory_packing (reverse)", tab=tab)
        elif method == "greedy_reverse_addx":
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, reverse=True, per_layer_x=x_list, verbose=verbose,title="greedy_memory_packing (reverse,addx)", tab=tab)
        elif method == "balanced":
            layer_packs[method] = balanced_memory_packing(memory_list, args.memory_cap, verbose=verbose, tab=tab)
        elif method == "balanced_addx":
            layer_packs[method] = balanced_memory_packing(memory_list, args.memory_cap, per_layer_x=x_list, verbose=verbose, title="balanced_memory_packing (addx)", tab=tab)
        elif method == "reuse":
            layer_packs[method] = reuse_memory_packing(reuse_packs, memory_list, x_list, args.memory_cap, verbose=verbose, tab=tab)
        elif method == "balanced_time":
            time_list = [ args.prof['TIME_FWDBWD'].get(type, ubatchsize, l, "GPU",interp=True) for l in range(num_layers) ] # sec (float)
            layer_packs[method] = balanced_time_packing(time_list, memory_list, x_list, args.memory_cap, verbose=verbose, tab=tab)
        # skip invalid packing
        if layer_packs[method] is None:
            del layer_packs[method]
    
    return layer_packs

def search(args):
    """ top-level function """
    """ search for the best configuration (Ufwd, Pfwd, Ubwd, Pbwd) for min estimated runtime under memory capacity constraints. """
    
    ### find microbatch sizes to search
    ubatchsizes_fwd, ubatchsizes_bwd = find_ubatchsizes(args)
    if args.verbose: 
        print("searchable ubatchsizes_fwd: {}".format(ubatchsizes_fwd)) 
        print("searchable ubatchsizes_bwd: {}".format(ubatchsizes_bwd)) 
    ubatchsizes_fwd = np.array(ubatchsizes_fwd, dtype=np.uint64)
    
    ### find valid ubatch size and layer packs
    valid_size_pack = [] # [(u_fwd, pack_fwd, u_bwd, pack_bwd, method_fwd, method_bwd)]
    t_start = pc()
    ## search BWD first
    for u_bwd in ubatchsizes_bwd: 
        if args.verbose: print("\nFor u_bwd: %d, find pack_bwd..."%u_bwd)
        method_packs_bwd = find_layer_packs(args, u_bwd, args.num_layers, "BWD", verbose=args.verbose)
        # under each BWD packing, search FWD
        for method_bwd, pack_bwd in method_packs_bwd.items():
            if args.verbose: print("\tFor %s, find FWD..."%method_bwd)
            assert len(pack_bwd) > 0
            if len(pack_bwd) == 1: # Empty (single BWD pack)
                if args.verbose: print("\t\tFWD is empty")
                u_fwd = u_bwd
                pack_fwd = []
                valid_size_pack.append((u_fwd, pack_fwd, u_bwd, pack_bwd, "", method_bwd))
            else: # search FWD
                num_layers_fwd = pack_bwd[:-1][-1][-1] + 1
                # assert num_layers_fwd == sum(len(p) for p in pack_bwd[:-1])
                if args.smaller_ufwd: # allow u_fwd < u_bwd
                    idx = 0
                    print("[WARNING] allow microbatch size of forward be smaller than backward is still bleeding.")
                else: # make u_fwd starts from u_bwd
                    idx = np.searchsorted(ubatchsizes_fwd,u_bwd,side='left') # works for non-valid u_fwd, valid u_fwd, even out of range u_fwd
                for u_fwd in ubatchsizes_fwd[idx:]: 
                    u_fwd = int(u_fwd)
                    if args.verbose: print("\t\tFor u_fwd: %d, find pack_fwd..."%u_fwd)
                    method_packs_fwd = find_layer_packs(args, u_fwd, num_layers_fwd, "FWD", reuse_packs=pack_bwd[:-1], verbose=args.verbose)
                    for method_fwd, pack_fwd in method_packs_fwd.items():
                        valid_size_pack.append((u_fwd, pack_fwd, u_bwd, pack_bwd, method_fwd, method_bwd))
    
    ### evaluate each valid one
    print("\nEvalute valid size and packs (%d points) ..."%len(valid_size_pack))
    if args.verbose: 
        print("< u_fwd, num_pack_fwd, u_bwd, num_pack_bwd, packing_method_fwd, packing_method_bwd >")
    top_k = TopK(args.topk, hash_fn if args.dedup else None, print_fn)
    for u_fwd, pack_fwd, u_bwd, pack_bwd, method_fwd, method_bwd in valid_size_pack:
        ## compose task graph
        if args.mode == 'vPP' and args.num_gpus > 1 and u_fwd != u_bwd:
            args.last_fwd_msg = True
        else:
            args.last_fwd_msg = False
        CONFIGS, TASKS, rTASKS = compose(args, u_fwd, pack_fwd, u_bwd, pack_bwd, verbose=False)
        ## estimate runtime 
        res = simulate(args, rTASKS, CONFIGS, TASKS=TASKS, prefetch_offload=args.prefetch_offload, verbose=False, view=False)
        if args.verbose: 
            print(sim_res_str(res, 
                              title="< %d, %d, %d, %d, %s, %s >"%
                              (u_fwd, len(pack_fwd), u_bwd, len(pack_bwd), 
                               method_fwd[:6].ljust(6,' '), 
                               method_bwd[:6].ljust(6,' ') ) ) )
        ## compare for the best
        global_time = res['global_endtime']
        if len(pack_bwd) == 1:
            if not args.rank_fit_normally:
                print("[WARNING] The entire model can fit onto a single GPU. Not recommend to use Harmony. The fit config is now put as Top1.")
                global_time /= 1000.
            else:
                print("[WARNING] The entire model can fit onto a single GPU. Not recommend to use Harmony. The fit config is still ranked normally.")
        ## add to top k best
        top_k.add(  global_time, 
                    ODict({ 'res': res, 
                            'CONFIGS': CONFIGS,
                            'rTASKS': rTASKS,
                            'packing_method_fwd': method_fwd,
                            'packing_method_bwd': method_bwd })) 
    
    t_end = pc()
    print("\n--- Search done: %.3f sec ---"%(t_end-t_start))
    
    return top_k


    
