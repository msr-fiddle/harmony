# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os

def initialize_args(custom_args_provider=None):
    parser = argparse.ArgumentParser(description="Harmony Profiler")

    """ Custom """
    if custom_args_provider is not None:
        parser = custom_args_provider(parser)

    """ Harmony """
    ### Model
    parser.add_argument("--module_dir", type=str, default="../../results",
                        help="directory for model code")
    parser.add_argument("--module_name", type=str, default="bert_large")
    parser.add_argument("--mode", type=str, default='normal', 
                        help=" 'probe' or 'normal' mode ")
    ### "probe" mode for probing the max microbatch size
    parser.add_argument("--probe_what", type=str, default='FWD', 
                        help="probing either forward 'FWD' or backward 'BWD'")
    ### "normal" mode for actual profiling after probe mode
    parser.add_argument("--fwd_umax", type=int, default=-1, 
                        help="By default -1, use probed result in --output_dir; Otherwise, force using this number.")
    parser.add_argument("--bwd_umax", type=int, default=-1, 
                        help="By default -1, use probed result in --output_dir; Otherwise, force using this number.")
    parser.add_argument("--ubatchsize_step", type=float, default=1.0,
                        help="profiling stride across microbatch sizes: when >= 1.0, use rounded int as the raw step size; when < 1.0, use it as the percentage of max microbatch size for the step size")
    parser.add_argument("--num_trials", type=int, default=1,
                        help="how many repeated runs for profiling")
    parser.add_argument("--what", type=str, nargs='+', default=['FWDBWD', 'UPD'],
                        help="profiling either forward plus backward ['FWDBWD'] or update ['UPD'] or all ['FWDBWD', 'UPD']")
    parser.add_argument("--no_offload_optim", default=False, action="store_true")
    ### Output
    parser.add_argument("--output_dir", type=str, default='', 
                        help="By default empty, save probe* and prof* to '--module_dir/--module_name'; otherwise, save them to this output_dir")
    parser.add_argument("--outname_suffix", type=str, default='', 
                        help='suffix to probe* and prof*. E.g., _seqlen128')
    parser.add_argument("--verbose", default=False, action="store_true")
    
    """ Parse """
    args = parser.parse_args()
    
    if args.output_dir == "":
        args.output_dir = os.path.join(args.module_dir, args.module_name)
    assert os.path.exists(args.output_dir)
    assert os.path.basename(args.output_dir) != "prof", "no base_dir in output_dir"
    
    print("----------- arguments ------------") 
    for key, val in vars(args).items():
        print("{:24} {}".format(key, val))
    print("----------------------------------")  

    return args

