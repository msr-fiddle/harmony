# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os

def initialize_args(custom_args_provider=None):
    parser = argparse.ArgumentParser(description="Harmony Runtime")

    """ Custom """
    if custom_args_provider is not None:
        parser = custom_args_provider(parser)

    """ Harmony """
    ### Data
    parser.add_argument("--synthetic_data", default=False, action='store_true')
    parser.add_argument('--data_workers', type=int, default=0, 
                        help='number of data loading workers for all models')
    ### Model
    parser.add_argument("--module_dir", type=str, default="../../results")
    parser.add_argument("--module_name", type=str, default="bert_large")
    parser.add_argument("--suffix", type=str, nargs='?', const='', default='', 
                        help="suffix to input profiles and schedules."
                        "If --suffix xx, use xx. Elif --suffix, use const. Else, use default."
                        "e.g. _seqlen128") 
    parser.add_argument("--profile_fnames", type=str, nargs='+', 
                        default=["prof_XMETA","prof_TMETA"], 
                        help="under --module_dir/--module_name")
    parser.add_argument("--schedule_dir", type=str, default="",
                        help="By default empty, use --module_dir/--module_name; otherwise, use this directory.")
    parser.add_argument("--schedule_fname", type=str, default="xxx.pickle", 
                        help="under --schedule_dir")
    ### Training Args
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="The number of epochs to train.")
    parser.add_argument('--num_iters', type=int, default=None, 
                        help="Within each epoch, the number of iterations (updates) to train.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="random seed for initialization")
    parser.add_argument("--seed_cudnn", default=False, action='store_true',
                        help="Whether to seed cuDNN")
    parser.add_argument("--average_buffer", default=False, action='store_true',
                        help="Whether to average buffers during vDP")
    ### Distributed environment
    parser.add_argument("--master_addr", default="localhost", 
                        help="IP address of master")
    parser.add_argument("--master_port", default=12345, 
                        help="Port used to communicate tensors")
    ### Runtime optimizations
    parser.add_argument("--no_all_prefetch_offload", default=False, action="store_true", 
                        help="Disable all prefetch/prerecv/offload below")
    parser.add_argument("--no_prefetch_model", default=False, action="store_true", 
                        help="Disable prefetch model")
    parser.add_argument("--no_prefetch_stashx", default=False, action="store_true", 
                        help="Disable prefetch stashX")
    parser.add_argument("--no_prefetch_localx", default=False, action="store_true", 
                        help="Disable prefetch local X and dY in vDP")
    parser.add_argument("--no_prefetch_msgx", default=False, action="store_true", 
                        help="Disable prefetch msg X and dY in vPP")
    parser.add_argument("--no_p2p_prerecv", default=False, action="store_true", 
                        help="Disable P2P prerecv X and dY in vPP")
    parser.add_argument("--no_offload_stashx", default=False, action="store_true", 
                        help="Disable nonblocking offload stashX")
    parser.add_argument("--no_offload_localx", default=False, action="store_true", 
                        help="Disable nonblocking offload local Y and dX in vDP")
    parser.add_argument("--no_offload_msgx", default=False, action="store_true", 
                        help="Disable nonblocking offload msg Y and dX in vPP")
    parser.add_argument("--empty_cache", default=False, action="store_true", 
                        help="Empty cache every iteration")
    parser.add_argument("--numa_bind", default=False, action="store_true", 
                        help="bind each rank to numa cpus")
    parser.add_argument("--numa_bind_config", default="../.numa_bind_configs/N8_1080Ti.json", 
                        help="{ rank0: [cpu#0, cpu#1] } ")
    parser.add_argument("--no_pin_model", default=False, action="store_true", 
                        help="Disable local pinned model (for swapin W and B)")
    parser.add_argument("--no_pin_grad_buf", default=False, action="store_true", 
                        help="Disable local pinned grad and buf (for swapout dW and B') [critial-path]")
    parser.add_argument("--no_pin_x", default=False, action="store_true", 
                        help="Disable pinned memory for StashX/LocalX/MSGX")
    parser.add_argument("--no_pin_data", default=False, action="store_true", 
                        help="Disable pinned memory for Data/Target [critial-path]")
    ### Outputs
    parser.add_argument("--output_dir", default="./logs",
                        help="The output directory to save everything.")
    parser.add_argument("--save_final_model", default=False, action="store_true",
                        help="Whether to save model at last")
    # parser.add_argument('--display_period', type=int, default=1)
    ### Debugs
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--nvprof', default=False, action='store_true', 
                        help="Whether use nvprof for 'viewing' runtime")
    parser.add_argument("--nvprof_iter", type=str, default="none", 
                        help="Which iteration to nvprof: first, last, all, none")
    parser.add_argument("--initial_iter_only", default=False, action="store_true", 
                        help="Run initial iteration only for debug.")
    parser.add_argument("--no_initial_iter", default=False, action="store_true",
                        help="Skip initial iteration, risking perf drop and hang.")
    parser.add_argument('--no_update', default=False, action='store_true', 
                        help="Disable update for debug")
    """ Parse """
    args = parser.parse_args()

    if args.no_all_prefetch_offload:
        args.no_prefetch_model = True
        args.no_prefetch_stashx = True
        args.no_prefetch_localx = True
        args.no_prefetch_msgx = True
        args.no_p2p_prerecv = True
        args.no_offload_stashx = True
        args.no_offload_localx = True
        args.no_offload_msgx = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("----------- arguments ------------") 
    for key, val in vars(args).items():
        print("{:24} {}".format(key, val))
    print("----------------------------------")  

    return args


