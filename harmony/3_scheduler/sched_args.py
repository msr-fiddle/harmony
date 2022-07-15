# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os

def initialize_args():
    parser = argparse.ArgumentParser(description="Harmony Scheduler")

    """ Manual Schedule """
    parser.add_argument("--manual", default=False, action="store_true",
                        help="use manual schedule, instead of search schedule") 
    parser.add_argument("--manual_ufwd", type=int, default=-1, 
                        help="microbatch size of forward")
    parser.add_argument("--manual_ubwd", type=int, default=-1,
                        help="microbatch size of backward")
    parser.add_argument("--manual_packsize", type=int, default=-1, 
                        help="constant layer pack size; Use either --manual_packsize or --manual_numpacks")
    parser.add_argument("--manual_numpacks", type=int, default=-1,
                        help="[DEBUG] varible layer pack size")
    
    """ Search Schedule """
    parser.add_argument("--ubatchsize_step", type=int, default=1, 
                        help="the interval to search microbatch sizes")
    parser.add_argument("--inequal_ubatchsize", default=False, action='store_true', 
                        help="allow inequal microbatchsize (disable double buffering)")
    parser.add_argument("--smaller_ufwd", default=False, action='store_true',
                        help="allow microbatch size of forward be smaller than backward")
    parser.add_argument("--packing_method_fwd", type=str, nargs='+', 
                        default=['greedy_reverse_addx', 'balanced_addx', 'reuse', 'balanced_time'], 
                        help="choose from ['greedy','greedy_addx','greedy_reverse','greedy_reverse_addx','balanced','balanced_addx', 'reuse', 'balanced_time']")
    parser.add_argument("--packing_method_bwd", type=str, nargs='+', 
                        default=['greedy_reverse_addx', 'balanced_addx', 'balanced_time'], 
                        help="choose from ['greedy','greedy_addx','greedy_reverse','greedy_reverse_addx','balanced','balanced_addx','balanced_time']")
    parser.add_argument("--memory_cap", type=float, default=10.0*(1024.**3),
                        help="per-GPU memory capacity in bytes")
    parser.add_argument("--memory_cap_scale", type=float, default=1.0,
                        help="scale factor for Per-GPU memory capacity. Consider CUDA context, NCCL context, and anything non-ideal for memory estimation.")
    parser.add_argument("--topk", type=int, default=1, 
                        help="top k configuration to search. set to -1 for keeping all configs.")
    parser.add_argument("--dedup", default=False, action='store_true', 
                        help="dedup configurations in top k")
    parser.add_argument("--rank_fit_normally", default=False, action='store_true', 
                        help="by default, when the entire model fits onto a single GPU, searcher automatically put the fit config as Top1. If it turns on, rank any fit configs normally.")

    """ Common Args """
    ### Model
    parser.add_argument("--module_dir", type=str, default="../results", 
                        help="directory for input profiles")
    parser.add_argument("--module_name", type=str, default="bert_large")
    parser.add_argument("--suffix", type=str, nargs='?', const='', default='', 
                        help="suffix to input profiles and output schedules."
                        "If --suffix xx, use xx. Elif --suffix, use const. Else, use default.") 
    ### Data
    parser.add_argument("--minibatchsize", type=int, default=50, 
                        help="The global minibatch size")
    ### Config
    parser.add_argument("--mode", type=str, default='vPP', 
                        help="Harmony DP (vDP) or Harmony PP (vPP)")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--reverse_bwd", default=False, action="store_true", 
                        help="[Deprecated] whether to reverse round-robin order of backward tasks. If True, jit backward tasks right after their forward tasks (e.g. the backward task of the last layer is executed on the same GPU as its forward task, saving swaps at the cost of extra bubble)")
    parser.add_argument("--no_offload_optim", default=False, action='store_true')
    parser.add_argument("--last_fwd_msg", action="store_true", 
                        help="[Automated] Replace last FWD's P2P(Y) to first BWD with MSG; vPP only")
    ### Simulation
    parser.add_argument("--simulation", default=False, action='store_true', 
                        help="whether run simulation during --manual")
    parser.add_argument("--profile_fnames", type=str, nargs='+', 
                        default=["prof_TIME_FWDBWD","prof_MEMORY_FWDBWD","prof_XMETA","prof_TMETA","prof_TIME_UPD","prof_WMETA","prof_BMETA","prof_KMETA"])
    parser.add_argument("--prefetch_offload", default=False, action='store_true', 
                        help="whether simulation uses all_prefetch_offload")
    parser.add_argument("--bw_swap", type=float, default=11.5*1024**3,        
                        help="actual CPU swap bandwidth (per-direction) (bytes/sec)")
    parser.add_argument("--bw_p2p", type=float, default=8.2*1024**3,        
                        help="actual P2P bandwidth (per-direction) (bytes/sec)")
    parser.add_argument("--bw_msg", type=float, default=128/8*1024**3,        
                        help="actual MSG bandwidth (per-direction) (bytes/sec)")
    parser.add_argument("--time_del", type=float, default=40./1000,        
                        help="empirical deletion time per task (sec)")
    parser.add_argument("--use_random", default=False, action='store_true', 
                        help="[DEBUG] use random time")
    parser.add_argument("--seed", type=int, default=0, 
                        help="[DEBUG] random seed")
    ### Simulation Viewer
    parser.add_argument("--view", default=False, action='store_true', 
                        help="[DEBUG] whether visualize simulation by chrome")
    parser.add_argument("--separate_swap", default=False, action='store_true', 
                        help="[DEBUG] for view")
    parser.add_argument("--dir_jsons", type=str, default="tmps", 
                        help="[DEBUG] for view, the output directory for json")
    parser.add_argument("--json", type=str, default="tmp.json.gz", 
                        help="[DEBUG] for view, the json filename")
    ### Output
    parser.add_argument("--output_dir", type=str, default='', 
                        help="By default empty, save schedule.pickle to '--module_dir/--module_name'; otherwise, save them to this output_dir")
    parser.add_argument("--verbose", default=False, action='store_true')

    """ Parse """
    args = parser.parse_args()
    
    args.module_path = os.path.join(args.module_dir, args.module_name)
    assert os.path.exists(args.module_path)
    assert os.path.basename(args.module_path) != "prof", "no base_dir in module_path"

    if args.output_dir == "":
        args.output_dir = args.module_path
    assert os.path.exists(args.output_dir)
    assert os.path.basename(args.output_dir) != "sched", "no base_dir in output_dir"

    if args.manual:
        assert args.manual_ufwd != -1 and args.manual_ubwd != -1
        assert (args.manual_packsize != -1 and args.manual_numpacks == -1) or \
               (args.manual_packsize == -1 and args.manual_numpacks != -1)
    else:
        args.memory_cap *= args.memory_cap_scale # scale memory capacity
        print("calibrated memory_cap = %.0f MB" % ( args.memory_cap/1024./1024. ) )

    print("----------- arguments ------------") 
    for key, val in vars(args).items():
        print("{:24} {}".format(key, val))
    print("----------------------------------")  

    return args

