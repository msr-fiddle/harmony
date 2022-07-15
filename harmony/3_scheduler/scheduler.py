# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
from collections import OrderedDict as ODict
from copy import deepcopy
import pickle

import sys; sys.path.append("../2_profiler")
from prof_data_struct import load_prof_data_struct

from sched_args import initialize_args
from layer_packing import manual_pack
from task_graph_composor import compose
from searcher import search
from simulator import simulate, sim_res_str
from task_data_struct import serialize_scheduled

def manual_schedule(args):
    ### add num of layers    
    xmeta = load_prof_data_struct(args.module_path, "prof_XMETA" + args.suffix)
    args.num_layers = len(xmeta.get_vlayer_ids())
    ### manual size data and pack layers
    if args.manual_packsize != -1:
        args.manual_pack_fwd, args.manual_pack_bwd = manual_pack(args.num_layers, args.manual_packsize)
    else: # DEBUG
        if args.module_name in ("bert_large", "bert_thomwolf", "bert_seq", "bert_2bw"):
            assert args.num_layers == 28 # 0~26 regular Bert + 27th criterion
            if args.manual_numpacks == 4:
                args.manual_pack_fwd = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
                args.manual_pack_bwd = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27]]
            elif args.manual_numpacks == 5:
                args.manual_pack_fwd = [list(range(0,14))]
                args.manual_pack_bwd = [list(range(0,4)), list(range(4,8)), list(range(8,14)), list(range(14,28))]
            elif args.manual_numpacks == 9:
                args.manual_pack_fwd = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]] 
                args.manual_pack_bwd = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27]]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    # ----- generate task graph -----
    if args.verbose:
        print("\n< Ufwd {}, Pfwd {}, Ubwd {}, Pbwd {} >\n".format(
            args.manual_ufwd, args.manual_pack_fwd, args.manual_ubwd, args.manual_pack_bwd))  
    if args.mode == 'vPP' and args.num_gpus > 1 and args.manual_ufwd != args.manual_ubwd:
        args.last_fwd_msg = True
    else:
        args.last_fwd_msg = False
    CONFIGS, TASKS, rTASKS = compose(args, args.manual_ufwd, args.manual_pack_fwd, args.manual_ubwd, args.manual_pack_bwd, verbose=args.verbose)
    #------ estimate runtime --------
    if args.simulation:
        ### read all profiles
        prof = ODict()
        for name in args.profile_fnames:
            key = name.split("prof_")[-1]
            prof[key] = load_prof_data_struct(args.module_path, name + args.suffix)
        args.prof = prof
        ### simulation
        res = simulate(args, rTASKS, CONFIGS, TASKS=TASKS, prefetch_offload=args.prefetch_offload, verbose=args.verbose, view=args.view)
        print(sim_res_str(res, 
                title="< %d, %d, %d, %d >"%
                (args.manual_ufwd, len(args.manual_pack_fwd), args.manual_ubwd, len(args.manual_pack_bwd)) ) )
    # ----- serialize into pickle -----
    fname = "D%d_%s_N%d_Ufwd%d_Ubwd%d"% (CONFIGS["D"], CONFIGS["mode"],CONFIGS["N"],CONFIGS["u_fwd"], CONFIGS["u_bwd"])
    if args.manual_packsize != -1:
        fname += "_P%d" % args.manual_packsize
    elif args.manual_numpacks != -1:
        fname += "_numP%d" % args.manual_numpacks
    print()
    serialize_scheduled(rTASKS, CONFIGS, args.output_dir, fname + args.suffix, base_dir="sched")

def search_schedule(args):
    ### read profiles
    prof = ODict()
    for name in args.profile_fnames:
        key = name.split("prof_")[-1]
        prof[key] = load_prof_data_struct(args.module_path, name + args.suffix)
    args.prof = prof
    args.num_layers = len(prof['XMETA'].get_vlayer_ids())
    args.fwd_ubatchsizes = prof['TIME_FWDBWD'].get_ubatchsizes('FWD')
    args.bwd_ubatchsizes = prof['TIME_FWDBWD'].get_ubatchsizes('BWD')
    if args.verbose: 
        print("number of layers: %d" % args.num_layers)
        print("fwd ubatchsizes: {}".format(args.fwd_ubatchsizes))
        print("bwd_ubatchsizes: {}".format(args.bwd_ubatchsizes))
    ### start search
    top_k = search(args)
    ### save TopK schedules
    sorted_ods, sorted_print = top_k.summary()
    ## schedule.pickle
    print()
    fname = "D%d_%s_N%d"%(args.minibatchsize, args.mode, args.num_gpus)
    for r, od in enumerate(sorted_ods):
        serialize_scheduled(od["rTASKS"], od["CONFIGS"], args.output_dir, fname + "_Top%d" % (r + 1) + args.suffix, base_dir="sched")
    ## summary.txt
    summary_path = os.path.join(args.output_dir, "sched", fname + "_Summary" + args.suffix + ".txt")
    with open(summary_path, "wt") as f:
        f.write("\n".join(sorted_print))
    print("searched summary saved: %s" % summary_path)
    ## global time list
    sorted_global_time = [ od['res']['global_endtime'] for od in sorted_ods ]
    gt_path = os.path.join(args.output_dir, "sched", fname + "_GlobalTime" + args.suffix + ".pickle")
    with open(gt_path,'wb') as f:
        pickle.dump(sorted_global_time, f)
    print("global time list saved: %s" % gt_path)

if __name__ == "__main__":
    
    args = initialize_args()
    
    if args.manual:
        manual_schedule(args)
    else:
        search_schedule(args)
    
