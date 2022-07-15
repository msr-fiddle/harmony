# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division, print_function
import os
import argparse
import json
import numpy as np
from collections import OrderedDict as ODict
from matplotlib import pyplot as plt

import sys
from prof_data_struct import *

""" Analyze Profiles of a Pack """
def time_of_pack(prof, type, ubatchsize, start_id, end_id, offload_optim=True, interp_ubatchsize=True):
    assert start_id <= end_id, "a pack is [start_id, end_id]"
    if type in ["FWD", "BWD"]:
        if interp_ubatchsize in [False, True]: # linear interp (per vlayer, then sum to be packed time)
            return sum([prof["TIME_FWDBWD"].get(type, ubatchsize, id, "GPU", interp=interp_ubatchsize) for id in range(start_id, end_id+1)])
        elif interp_ubatchsize == 'fit': # poly fit (per vlayer, then sum to be packed time)
            return sum([prof["TIME_FWDBWD"].get_fit(type, ubatchsize, id, "GPU") for id in range(start_id, end_id+1)])
        else:
            assert False
    elif type in ["UPD"]:
        if offload_optim:
            return sum([prof["TIME_UPD"].get(type, None, id, "CPU") for id in range(start_id, end_id+1)]) 
        else:
            return sum([prof["TIME_UPD"].get(type, None, id, "GPU") for id in range(start_id, end_id+1)]) 
    # Return Sec

def memory_of_pack(prof, type, ubatchsize, start_id, end_id, offload_optim=True, interp_ubatchsize=True):
    assert start_id <= end_id, "a pack is [start_id, end_id]"
    if type in ["FWD", "BWD"]:
        memory_sum = sum([prof["MEMORY_FWDBWD"].get(type, ubatchsize, id, interp=interp_ubatchsize) for id in range(start_id, end_id+1)]) 
        xmeta_sum = sum([prof["XMETA"].get_bytes(ubatchsize, id, interp=interp_ubatchsize) for id in range(start_id+1, end_id+1)])            
        return int(memory_sum - xmeta_sum) # linear interp (per vlayer, then sub to be packed memory)
    elif type in ["UPD"]:
        if offload_optim:
            return 0
        else:
            return sum([prof["MEMORY_UPD"].get(type, None, id) for id in range(start_id, end_id+1)]) 
    # Return byte

def model_size_of_pack(prof, which, start_id, end_id):
    assert start_id <= end_id, "a pack is [start_id, end_id]"
    assert which in ['W','B','K','WMETA','BMETA','KMETA']
    if not "META" in which:
        which += "META"
    return sum([prof[which].get_bytes(id) for id in range(start_id, end_id+1)]) # byte

""" Visualize Profiles of a Pack """
def pack2show(start_id, end_id):
    assert start_id <= end_id, "a pack is [start_id, end_id]"
    if start_id == end_id:
        return "%d" % start_id
    else:
        return "%d-%d" % (start_id, end_id)

def time2show(t): # sec
    assert isinstance(t,float)
    return "-" if t == 0.0 else "%.3f" % (t*1000) # milli-sec    

def memory2show(m): # byte
    assert m is None or isinstance(m, int)
    return "-" if m is None else "%d"%( int(float(m)/1024/1024) ) # MB

def size2show(s):
    return memory2show(s)

def show_packs_given_type_u(module_name, packs, prof, type, ubatchsize, offload_optim=True):
    print("\n=================== {}'s {} ===================".format(module_name, 'BWD with Recompute' if type == 'BWD' else type ))
    if type in ['FWD','BWD']:
        print("----------------- ubatchsize={} ------------------".format(ubatchsize))
        print("vlayer: time(ms) mem(MB) X(MB) W(MB) B(MB)")
        for p in packs:
            start_id, end_id = p[0], p[-1]
            print("{}:\t{}\t{}\t{}\t{}\t{}".format(
                pack2show(start_id, end_id),
                time2show(time_of_pack(prof, type, ubatchsize, start_id, end_id)),
                memory2show(memory_of_pack(prof, type, ubatchsize, start_id, end_id)),
                size2show(prof["XMETA"].get_bytes(ubatchsize, start_id, interp=True)),
                size2show(model_size_of_pack(prof, 'W', start_id, end_id)),
                size2show(model_size_of_pack(prof, 'B', start_id, end_id))
            ))
    elif type in ['UPD'] and offload_optim:
        print("vlayer: time(ms) W(MB) K(MB)")
        for p in packs:
            start_id, end_id = p[0], p[-1]
            print("{}:\t{}\t{}\t{}".format(
                pack2show(start_id, end_id),
                time2show(time_of_pack(prof, 'UPD', None, start_id, end_id, offload_optim=True)),
                size2show(model_size_of_pack(prof, 'W', start_id, end_id)),
                size2show(model_size_of_pack(prof, 'K', start_id, end_id))
            ))
    else:
        raise ValueError
    print("========================================================")

def time2plot(t): # sec
    assert isinstance(t,float)
    return t*1000 # milli-sec    

def memory2plot(m): # byte
    assert m is None or isinstance(m, int)
    return 0 if m is None else int(float(m)/1024/1024) # MB

def size2plot(s):
    return memory2plot(s)

def plot_time_memory_size_uscale( xy_TIME, xy_interp_TIME, xy_fit_TIME, 
                                  xy_MEMORY, xy_X, xy_W, xy_B, 
                                  plotdir="plots", plotname="FWD_vLayer0.pdf", 
                                  figsize=(6.4,3.1), 
                                  xlabel="MicroBatchSize", 
                                  y1label="Memory/Size (MB)", y2label="Time (ms)", 
                                  y1colors = ["tab:red","tab:orange","tab:purple","tab:pink"],
                                  y2colors = ["tab:blue", "tab:cyan", "limegreen"] ):
    print("plotting: {} ...".format(plotname))
    assert len(xy_TIME[0]) == len(xy_TIME[1]) and len(xy_interp_TIME[0]) == len(xy_interp_TIME[1]) and len(xy_fit_TIME[0]) == len(xy_fit_TIME[1])
    assert len(xy_MEMORY[0]) == len(xy_MEMORY[1])
    assert len(xy_X[0]) == len(xy_X[1])
    assert len(xy_W[0]) == len(xy_W[1])
    assert len(xy_B[0]) == len(xy_B[1])
    # -----------------------
    # Ref: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
    fig, ax1 = plt.subplots(figsize=figsize)
    # ------ Y1 Axis ---------
    ax1.plot(xy_MEMORY[0], xy_MEMORY[1], "-D", label="Memory", color=y1colors[0], markevery=1, markersize=5)
    ax1.plot(xy_X[0], xy_X[1], "-s", label="X", color=y1colors[1], markevery=1, markersize=4)
    ax1.plot(xy_W[0], xy_W[1], "-^", label="W", color=y1colors[2], markevery=1, markersize=2)
    ax1.plot(xy_B[0], xy_B[1], "-v", label="B", color=y1colors[3], markevery=1, markersize=2)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label) #, color=color1)
    ax1.tick_params(axis='y') #, labelcolor=color1)
    ax1.grid()
    ax1.set_title(plotname.replace('.pdf',''))
    ax1.legend(loc='upper left')
    # ------ Y2 Axis ---------
    ax2 = ax1.twinx()

    ax2.plot(xy_TIME[0], xy_TIME[1], "-x", label="Time", color=y2colors[0], markevery=1, markersize=5)
    ax2.plot(xy_interp_TIME[0], xy_interp_TIME[1], "-o", label="Time(Interp)", color=y2colors[1], markevery=1, markersize=2)
    ax2.plot(xy_fit_TIME[0], xy_fit_TIME[1], "--", label="Time(Fit)", color=y2colors[2], markevery=1, markersize=1)

    # we already handled the x-label with ax1
    ax2.set_ylabel(y2label) #,color=color2)  
    ax2.tick_params(axis='y') #, labelcolor=color2)
    ax2.legend(loc='lower right') # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
    # ----- Optional: Align scale of two y-axis ----
    # ref: https://stackoverflow.com/questions/45037386/trouble-aligning-ticks-for-matplotlib-twinx-axes
    # l1 = ax1.get_ylim()
    # l2 = ax2.get_ylim()
    # f = lambda x : l2[0]+(x-l1[0])/(l1[1]-l1[0])*(l2[1]-l2[0])
    # ticks = f(ax1.get_yticks())
    # ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    # ----- Save figure -----
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    plot_path = os.path.join(plotdir, plotname)
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print("plot written: {}".format(plot_path))

def show_uscale_given_type_pack(args, prof, type, pack):
    assert type in ['FWD','BWD']
    start_id, end_id = pack[0], pack[-1]
    # ---- Time ----
    # actual points
    x = prof["TIME_FWDBWD"].get_ubatchsizes(type)
    y = [ time_of_pack(prof, type, ubatchsize, start_id, end_id, interp_ubatchsize=False) for ubatchsize in x ] #sec
    y = [time2plot(t) for t in y] # ms
    xy_TIME = [x,y]
    # linear interp
    x_interp = range(x[0],x[-1]+1,1)
    y_interp = [ time_of_pack(prof, type, ubatchsize, start_id, end_id, interp_ubatchsize=True) for ubatchsize in x_interp ] #sec
    y_interp = [time2plot(t) for t in y_interp] # ms
    xy_interp_TIME = [x_interp,y_interp]
    # poly fit
    x_fit = prof["TIME_FWDBWD"].get_ubatchsizes_fit(type)
    y_fit = [ time_of_pack(prof, type, ubatchsize, start_id, end_id, interp_ubatchsize='fit') for ubatchsize in x_fit ] #sec
    y_fit = [time2plot(t) for t in y_fit] # ms
    xy_fit_TIME = [x_fit,y_fit]
    # ---- Memory and Size ----
    # MEMORY
    x = prof["MEMORY_FWDBWD"].get_ubatchsizes(type)
    y = [memory_of_pack(prof, type, ubatchsize, start_id, end_id, interp_ubatchsize=False) for ubatchsize in x] # Bytes
    y = [memory2plot(t) for t in y] # MB
    xy_MEMORY = [x,y]
    # X
    x = xy_MEMORY[0]
    y = [prof["XMETA"].get_bytes(ubatchsize, start_id, interp=False) for ubatchsize in x] # Bytes
    y = [size2plot(t) for t in y] # MB
    xy_X = [x,y]
    # W
    x = xy_MEMORY[0]
    y = [model_size_of_pack(prof, 'W', start_id, end_id) for _ in x] # Bytes
    y = [size2plot(t) for t in y] # MB
    xy_W = [x,y]
    # B
    x = xy_MEMORY[0]
    y = [model_size_of_pack(prof, 'B', start_id, end_id) for _ in x] # Bytes
    y = [size2plot(t) for t in y] # MB
    xy_B = [x,y]
    # ---- Plot ----
    plot_time_memory_size_uscale( xy_TIME, xy_interp_TIME, xy_fit_TIME, 
                                  xy_MEMORY, xy_X, xy_W, xy_B, 
                                  plotdir=args.analysis_dir, 
                                  plotname="{}_vLayer{}.pdf".format(
                                      type, pack2show(start_id, end_id)) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1) Analyze time/memory/size of a given layer pack and given microbatch size, and 2) Visualize Harmony profiles")
    ### Input profiles
    parser.add_argument("--module_dir", type=str, default="../results",
                        help="directory for input profiles")
    parser.add_argument("--module_name", type=str, default="bert_large")
    parser.add_argument("--profiled_fnames", type=str, nargs='+', 
                        default=["prof_TIME_FWDBWD","prof_MEMORY_FWDBWD","prof_XMETA","prof_TMETA","prof_TIME_UPD","prof_WMETA","prof_BMETA","prof_KMETA"])
    parser.add_argument("--suffix", type=str, default='', help='suffix to prof*. E.g., _seqlen128')
    parser.add_argument("--no_offload_optim", default=False, action="store_true")
    ### Visual option
    parser.add_argument("--mode", type=str, required=True,
                        help =
                        "model: show model time/memory/size for given ubatchsize;"
                        "data:  plot ubatchsize scale for given model")
    parser.add_argument("--analysis_dir", type=str, required=True,
                        help = "The directory to save plots and configs. E.g., analysis/bert_large/Ufwd30_Ubwd30_P2")
    parser.add_argument("--size_pack", type=str, default="size_pack.json",
                        help = "config of data size and layer pack, under --analysis_dir")
    """                 { 'u_fwd':10, 'pack_fwd':[ [0,1,2], [3,4,5], [6,7,8] ],  
                          'u_bwd':10, 'pack_bwd':[ [0,1,2], [3,4,5], [6,7,8] ], 
                          'num_layers': 9 } 
                        If pack_fwd/bwd not exist, then set it to one layer pack. """
    args = parser.parse_args()
    
    ### read size_pack.json
    assert os.path.exists(args.analysis_dir)
    with open(os.path.join(args.analysis_dir, args.size_pack)) as f:
        size_pack = json.load(f)
        if not ('pack_fwd' in size_pack):
            size_pack['pack_fwd'] = [ [l] for l in range(size_pack['num_layers']) ]
        if not ('pack_bwd' in size_pack):
            size_pack['pack_bwd'] = [ [l] for l in range(size_pack['num_layers']) ]
        # Assert format of layer packs:
        # 0) layer is continous within a pack
        # 1) pack is inclusive
        # 2) packed layer id always increasing
        # 3) last bwd pack element is R-1    
        # (the same as vt.layers)
        def assert_pack_format(packs, num_layers):
            unpack = []
            for p in packs:
                unpack += p
            assert unpack == list(range(num_layers))
        assert_pack_format(size_pack['pack_bwd'], size_pack['num_layers'])
        if size_pack['pack_fwd'][-1][-1] == size_pack['num_layers']-1:
            assert_pack_format(size_pack['pack_fwd'], size_pack['num_layers'])
        else:
            assert_pack_format(size_pack['pack_fwd']+[size_pack['pack_bwd'][-1]], size_pack['num_layers'])
        args.size_pack = size_pack
    
    print("------ input arguments ------") 
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))
    print("-----------------------------") 
    
    ### read prof_data_struct
    module_path = os.path.join(args.module_dir, args.module_name)
    assert os.path.exists(module_path)
    assert os.path.basename(module_path) != "prof", "no base_dir in module_path"
    prof = ODict()
    for name in args.profiled_fnames:
        key = name.split("prof_")[-1]
        prof[key] = load_prof_data_struct(module_path, name+args.suffix)
    
    ### visualize
    if args.mode == 'model':
        show_packs_given_type_u(args.module_name, args.size_pack['pack_fwd'], prof, 'FWD', args.size_pack['u_fwd'])
        show_packs_given_type_u(args.module_name, args.size_pack['pack_bwd'], prof, 'BWD', args.size_pack['u_bwd'])
        show_packs_given_type_u(args.module_name, args.size_pack['pack_bwd'], prof, 'UPD', None, not args.no_offload_optim)
    elif args.mode == 'data':
        for p in args.size_pack['pack_fwd']:
            show_uscale_given_type_pack(args, prof, 'FWD', p)
        for p in args.size_pack['pack_bwd']:
            show_uscale_given_type_pack(args, prof, 'BWD', p)
    else:
        raise ValueError
