# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import importlib
from collections import OrderedDict as ODict
from copy import deepcopy
import os
from shutil import copyfile
import pickle
import json
import psutil
from psutil._common import bytes2human
import gc
import time

import torch
import torch.distributed as dist

class NumaBinder(object):
    def __init__(self, config_path):
        os.path.exists(config_path)
        with open(config_path) as f:
            self.config = json.load(f)
            # { rank0: [cpu#0, cpu#1] } 
    def bind(self, process, rank):
        proc = psutil.Process(int(process.pid))
        assert str(rank) in self.config
        assert isinstance(self.config[str(rank)], list)
        proc.cpu_affinity(self.config[str(rank)])
        # time.sleep(3)
        result = sorted( list( set( proc.cpu_affinity() ) ) )
        assert result == self.config[str(rank)], \
              "numa binding failed: rank{} to cpu{} (vs config {})".format(rank, result, self.config[str(rank)])
        print("[Numa Binding] succeeded: rank{} to cpu{} ".format(rank, self.config[str(rank)]))
       

@torch.no_grad()
def allreduce_cpu_loss(loss, averaging=True):
    # NOTE: per-GPU batch size need to be equal
    torch.cuda.synchronize()
    dist.barrier()
    assert isinstance(loss, float)
    t = torch.tensor(loss, dtype=torch.float32, device='cpu')
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if averaging:
        t /= float(dist.get_world_size())
    return t.item()

def gather_integer(integer, rank, dst=0):
    assert isinstance(integer, int)
    t = torch.tensor(integer, dtype=torch.int64, device='cpu')
    if rank == dst:
        gather_list = [ torch.tensor(0, dtype=torch.int64, device='cpu') 
                                    for _ in range(dist.get_world_size()) ] 
    else:
        gather_list = None
    dist.gather(t, gather_list=gather_list, dst=dst)
    if rank == dst:
        return [t.item() for t in gather_list]
    else:
        return None
    

def load_model(src_state_dict, dst_model, verbose=False):
    """ Load state dict of baseline model into Harmony model
    (ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html)

    e.g. load_model(torch.load(os.path.join(path, "pytorch_model.bin")), model, verbose=True)  
    """
    assert isinstance(src_state_dict, ODict)
    assert isinstance(dst_model, list)
    src_sdict_iter = iter(src_state_dict.items())
    for i, (vlayer, _, _) in enumerate(dst_model):
        new_state_dict = ODict()
        for key, val in vlayer.state_dict().items():
            src_key, src_val = next(src_sdict_iter)
            assert val.shape == src_val.shape
            assert val.dtype == src_val.dtype
            new_state_dict[key] = src_val
        vlayer.load_state_dict(new_state_dict)
    if verbose: print("model loaded")

def copy_model(src_model, dst_state_dict, verbose=False):
    """ in-place copy state dict of Harmony model to a baseline model.

    e.g. state_dict = copy_model(model, torch.load(os.path.join(path, "pytorch_model.bin")), verbose=True)  
    """
    assert isinstance(src_model, list)
    assert isinstance(dst_state_dict, ODict)
    dst_sdict_iter = iter(dst_state_dict.items())
    for i, (vlayer, _, _) in enumerate(src_model):
        for key, val in vlayer.state_dict().items():
            dst_key, dst_val = next(dst_sdict_iter)
            assert val.shape == dst_val.shape
            assert val.dtype == dst_val.dtype
            dst_state_dict[dst_key] = val
    if verbose: print("model copied")
    return dst_state_dict
        
def save_model(state_dict, dir_orig, dir_save, white_list=['json','txt'], verbose=False):
    """ save state dict of baseline model to disk """
    os.makedirs(dir_save, exist_ok=True)
    torch.save(state_dict, os.path.join(dir_save, "pytorch_model.bin"))
    for fname in os.listdir(dir_orig):
        if fname.split('.')[-1] in white_list:
            copyfile(src=os.path.join(dir_orig, fname), 
                     dst=os.path.join(dir_save, fname))
    if verbose: print("saved model to {}".format(dir_save))
    

class CheckGPUMem(object):
    def __init__(self, rank):
        self.rank = rank
        self.mem_stats_prev = torch.cuda.memory_stats(self.rank) 

    def check(self, it, is_check_retry=True, is_check_malloc=True):
        # check OoM & cudaFree
        mem_stats = torch.cuda.memory_stats(self.rank) 
        if is_check_retry and \
            mem_stats["num_alloc_retries"] != self.mem_stats_prev["num_alloc_retries"]:
            inc_retry = mem_stats["num_alloc_retries"] - \
                        self.mem_stats_prev["num_alloc_retries"] # num of failed cudaMalloc calls that result in a cache flush and retry.
            print("rank%d: Warning!! @ it %d, memory allocation retried (%d times)"%(self.rank, it, inc_retry)) 
        # check cudaMalloc
        if is_check_malloc and \
            mem_stats["segment.all.allocated"] != self.mem_stats_prev["segment.all.allocated"]:
            inc_malloc = mem_stats["segment.all.allocated"] - \
                        self.mem_stats_prev["segment.all.allocated"]
            inc_malloc_bytes = mem_stats["reserved_bytes.all.allocated"] - \
                            self.mem_stats_prev["reserved_bytes.all.allocated"]
            print("rank%d: Warning! @ it %d, cudaMalloc called (%d times, %.3f MB)"%(self.rank, it, inc_malloc, int(inc_malloc_bytes/1024./1024.)))
        self.mem_stats_prev = mem_stats


def print_gpu_mem(rank, vt, description):
    print("\trank{}: task{}({}) {} with {}/{} MB".format(
            rank, vt.idx, vt.show_layers(), description,
            int(torch.cuda.memory_allocated(rank)/1024.0/1024.0),
            int(torch.cuda.memory_reserved(rank)/1024.0/1024.0) ))  

def print_p2p_bytes(rank, p2px_handler, p2pm_handler, update_cnt):
    if p2px_handler is not None:
        p2pout_bytes = p2px_handler.send_byte_cnt/float(update_cnt)
        p2pin_bytes  = p2px_handler.recv_byte_cnt/float(update_cnt)
        print("rank%d: P2POut %.3f MB, P2PIn %.3f MB (per iter)"%
                (rank, p2pout_bytes/1024./1024., p2pin_bytes/1024./1024.) )
    if p2pm_handler is not None:
        allreduce_bytes = p2pm_handler.allreduce_byte_cnt/float(update_cnt)
        print("rank%d: AllReduce %.3f MB (per iter)"% 
                (rank, allreduce_bytes/1024./1024.) )

class PrintCPUMem(object):
    """ NOTE: use with caution. High overhead. """
    def __init__(self, 
                train_loop_choices=[], # ["process","system"], 
                the_rest_choices=["system"]): # ["tensor","process","system"]
        self.p = psutil.Process(None) # None means self pid
        self.train_loop_choices = train_loop_choices
        self.the_rest_choices = the_rest_choices
        
    def tensor_cpu_memory(self):
        """ return current cpu tensors' total size, in current process """
        total_bytes = sum([obj.numel()*obj.element_size() for obj in gc.get_objects() 
                            if isinstance(obj, torch.Tensor) and not obj.is_cuda])
        return "cpu tensor %s"%bytes2human(total_bytes) # bytes int, return string

    def process_cpu_memory(self, metrics=["uss", "pss", "vms", "shared"]):
        """ 
        return current process-level cpu memory usage as a string:
        e.g. 'Uss 174.8M, Pss 175.8M, Rss 176.5M, Vms 3.4G, Shared 104.6M, Swap 0.0B'
        
        memory_full_info():
        - rss: aka “Resident Set Size”, this is the non-swapped physical memory a process has used. On UNIX it matches “top“‘s RES column). 
        - vms: aka “Virtual Memory Size”, this is the total amount of virtual memory used by the process. On UNIX it matches “top“‘s VIRT column.
        - shared: memory that could be potentially shared with other processes. This matches “top“‘s SHR column).
        - text: aka TRS (text resident set) the amount of memory devoted to executable code. This matches “top“‘s CODE column).
        - data: aka DRS (data resident set) the amount of physical memory devoted to other than executable code. It matches “top“‘s DATA column).
        - lib: the memory used by shared libraries.
        - dirty: the number of dirty pages.
        - uss: aka “Unique Set Size”, this is the memory which is unique to a process and which would be freed if the process was terminated right now.
        - pss: aka “Proportional Set Size”, is the amount of memory shared with other processes, accounted in a way that the amount is divided evenly between the processes that share it. I.e. if a process has 10 MBs all to itself and 10 MBs shared with another process its PSS will be 15 MBs.
        - swap: amount of memory that has been swapped out to disk.
        ref (https://psutil.readthedocs.io/en/latest/index.html?highlight=memory_info#psutil.Process.memory_info)
        """
        named_tuple = self.p.memory_full_info() # pfullmem(rss=10199040, vms=52133888, shared=3887104, text=2867200, lib=0, data=5967872, dirty=0, uss=6545408, pss=6872064, swap=0)
        ps = []
        for name in metrics:
            value = getattr(named_tuple, name)
            value = bytes2human(value)
            ps.append('%s %s'%(name.capitalize(),value))
        return ', '.join(ps)

    def system_cpu_memory(self, metrics=["used", "shared", "available", "swap", "occupied"]):
        """ 
        return current system-level cpu memory usage as a string:
        e.g. 'Used 3.3G, Active 23.6G, Shared 2.7M, Available 746.5G, Swap 0.0B'
        
        psutil.virtual_memory():
        - total: total physical memory (exclusive swap).
        - available: the memory that can be given instantly to processes without the system going into swap.
        - used: memory used. total - free does not necessarily match used.
        - free: memory not being used at all (zeroed) that is readily available; note that this doesn’t reflect the actual memory available (use available instead). total - used does not necessarily match free.
        - active: memory currently in use or very recently used, and so it is in RAM.
        - inactive: memory that is marked as not used.
        - buffers: cache for things like file system metadata.
        - cached: cache for various things.
        - shared: memory that may be simultaneously accessed by multiple processes.
        - slab: in-kernel data structures cache.
        psutil.swap_memory():
        - total: total swap memory in bytes
        - used (swap): used swap memory in bytes
        - free: free swap memory in bytes
        ref (https://psutil.readthedocs.io/en/latest/index.html?highlight=virutal_memory#psutil.virtual_memory)
            (https://github.com/giampaolo/psutil/blob/master/psutil/_common.py)
        - occupied: total - available
        """
        named_tuple = psutil.virtual_memory() # svmem(total=810210418688, available=801578504192, percent=1.1, used=3584335872, free=737506328576, active=25313386496, inactive=41244495872, buffers=4782653440, cached=64337100800, shared=2789376, slab=3403677696)
        named_tuple2 = psutil.swap_memory() # sswap(total=1023406080, used=0, free=1023406080, percent=0.0, sin=0, sout=0)
        ps = []
        for name in metrics:
            if name == "swap":
                value = getattr(named_tuple2, 'used')
            elif name == "occupied":
                value = getattr(named_tuple, 'total') - getattr(named_tuple, 'available')
            else:
                value = getattr(named_tuple, name)
            #
            if name != 'percent':
                value = bytes2human(value)
            ps.append('%s %s'%(name.capitalize(),value))
        return ', '.join(ps)

    def print(self, title="", train_loop=False):
        choices = self.train_loop_choices if train_loop else self.the_rest_choices
        if choices == []:
            return
        ps = [title]
        for c in choices:
            if c == "tensor":
                ps.append(self.tensor_cpu_memory())
            elif c == "process":
                ps.append(self.process_cpu_memory())
            elif c == "system":
                ps.append(self.system_cpu_memory())
        print(" | ".join(ps))
        # print(title, "|", tensor_cpu_memory(), "|", process_cpu_memory(metrics=["uss", "pss", "vms", "shared", "swap"]), "|", system_cpu_memory(metrics=["used", "shared", "available", "swap"]))

# def dump(obj, path):
#     if ".pickle" not in path:
#         path += ".pickle"
#     with open(path, 'wb') as f:
#         # pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
#         torch.save(obj, f)
