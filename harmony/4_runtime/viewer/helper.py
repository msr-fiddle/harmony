# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sqlite3
import enum
import json
import subprocess
import os
import sys
import copy
from collections import OrderedDict as ODict
import numpy as np

def vPP_P2P_stream_names(rank, world_size, reverse_bwd=True):
    """ Copied from 4_runtime/p2p.py """
    # build two-process group for NCCL broadcast (r1, r2)  
    groups = ODict() # { "r1->r2": "P2PIn/Out_v/^" } # ordered in time
    # in-order round-robin
    for r1 in range(world_size):
        r2 = (r1+1) % world_size
        if rank in [r1,r2]:
            groups["{}->{}".format(r1,r2)] = "P2POut_v" if rank == r1 else "P2PIn_v"
    # reverse round-robin
    if reverse_bwd:
        for r1 in range(world_size):
            r2 = (r1-1) % world_size
            if rank in [r1,r2]:
                groups["{}->{}".format(r1,r2)] = "P2POut_^" if rank == r1 else "P2PIn_^"
    # initialize NCCL communicator and its cudaStream in mainthread
    # print("[vPP_P2P_stream_names] rank={}, world_size={}:".format(rank, world_size))
    names = []
    for i, (key, name) in enumerate(groups.items()):
        # print("[P2P] __init__: rank={} init'ed NCCL communicator[{}] and its cudaStream".format(rank, key))
        # print("\tP2PStream#{} -- {}:{}".format(i, key, name))
        names.append(name)
    # 
    return names # stream names order in time

def munge_time(t):
    """Take a time from nvprof (nano-sec) and convert it into a chrome://tracing's event time (ts & dur is micro-sec) """
    # For strict correctness, divide by 1000, but this reduces accuracy. 
    # > still accurate after division
    return t / 1000.0
    """Also, 'displayTimeUnit' is a string that specifies in which unit timestamps should be displayed. This supports values of “ms” or “ns”. By default this is value is “ms”. """
    # > ms is good

def demunge_time(t):
    """ Reverse above function """
    return t * 1000.0

def demangle_string(name):
    """Demangle a C++ identifier using c++filt"""
    # TODO: create the process only once. (70% processing time goes here but complex to do)
    try:
        with open(os.devnull, 'w') as devnull:
            return subprocess.check_output(['c++filt', '-n', name], stderr=devnull).rstrip().decode("ascii")
    except subprocess.CalledProcessError:
        return name

# import cxxfilt # https://pypi.org/project/cxxfilt/
# def demangle_string_fast(name):
#     """Demangle a C++ identifier using python cxxfilt"""
#     try:
#         return cxxfilt.demangle(name, external_only=False).rstrip().decode("ascii")
#     except:
#         return name
# > doesn't work for torch function name

def throughput(size_byte, dur_ns):
    tput = float(size_byte)/1024.0/1024.0/1024.0 / (float(dur_ns)/1000000000.0) # GB/s
    return "%.2f GB/s" % tput

def sizeof_fmt(num, suffix='B'):
    """Format size with metric units (like nvvp)"""
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%.2f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.2f %s%s" % (num, 'Y', suffix)

def parse_memory_probe(mark, gpu_memory_cap): # "peak_allocated B /peak_reserved B" # gpu_memory_cap bytes
    peak_allocated, peak_reserved = mark.split("/")
    peak_allocated, peak_reserved = int(peak_allocated), int(peak_reserved)
    peak_allocated_percent = peak_allocated/float(gpu_memory_cap)*100.0 # percent 
    peak_reserved_percent = peak_reserved/float(gpu_memory_cap)*100.0 # percent 
    peak_allocated_mb = int(peak_allocated/1024./1024.) # MB
    peak_reserved_mb = int(peak_reserved/1024./1024.) # MB
    return (peak_allocated_percent, peak_reserved_percent, 
            peak_allocated_mb, peak_reserved_mb)

def parse_cpu_probe(mark): # "total_percent|p0_percent,p1_percent|total_cnt"
    total_percent, rank_percent, total_cnt = mark.split("|") 
    total_percent = float(total_percent)
    rank_percent = [float(percent) for percent in rank_percent.split(',')]
    total_cnt = int(total_cnt)
    return (total_percent, rank_percent, total_cnt)

def sort_events_by_field(events, field, reverse=False):
    # 
    def fn(ev):
        if field == 'ts':
            return ev['ts']
        else:
            raise NotImplementedError
    # sort list out-of-place by key function taking each element
    assert isinstance(events, list)
    return sorted(events, key=fn, reverse=reverse)

def left_shift_counter_value_by_one(events):
    assert isinstance(events, list)
    for i, ev in enumerate(events[:-1]):
        ev["args"] = events[i+1]["args"]

def periodize_counter_value(events, pad_max_value=None):
    """ events = list, pad_max_value = None or {'serie': 100.} """
    if events == []:
        return []
    # sort probed memory counter in time
    sorted_events = sort_events_by_field(events, 'ts')
    # move counter value one event ahead (for peak memory period)
    left_shift_counter_value_by_one(sorted_events)
    if pad_max_value is not None:
        # append dummies to the last counter for 100% height display
        ev_dummy = copy.deepcopy(sorted_events[-1])
        ev_dummy["ts"] += 1 # us
        for k, v in pad_max_value.items():
            ev_dummy["args"][k] = v
        sorted_events.append(ev_dummy)
        # sorted_events[-1]["args"]["peak allocated memory"] = 100.
    return sorted_events
  
class CpuUtilAvg(object):
    def __init__(self):
        """ Works for either raw util or normalized util """
        """ Always use raw number (without clipping) """
        self.rank_util = ODict() # { rank: [util1, util2] }
    
    def add(self, total_percent=None, rank_percent=[]):
        if total_percent is not None:
            if not (-1 in self.rank_util):
                self.rank_util[-1] = []
            self.rank_util[-1].append(total_percent)
        for r, p in enumerate(rank_percent):
            if not (r in self.rank_util):
                self.rank_util[r] = []
            self.rank_util[r].append(p)
            
    def print(self, title="Raw CPU Util (%)"): 
        """ Per-rank averaging across the list == per-rank averaging area """
        # averging
        for r, util_list in self.rank_util.items():
            self.rank_util[r] = np.mean(util_list)
        # print
        print("=== {} ===".format(title))
        for r, avg in self.rank_util.items():
            avg_str = "%.1f" % avg
            avg_str = avg_str.rjust(6, ' ')
            if r == -1:
                print("Total :\t%s"%avg_str)
                print("--------------------")
            else:
                print("Rank%d :\t%s"%(r, avg_str))
        print("====================\n")

def clip_raw_util(percent, total_count):
    cap = total_count*100.
    if percent > cap:
        return cap + 1.
    else:
        return percent

# def marks_to_ranges(events):
#     """ "BKGD: task0(L0-1) UPD(W) BEG"--"BKGD: task0(L0-1) UPD(W) END" ==>
#         "BKGD: task0(L0-1) UPD(W)"
#     """
#     if events == []:
#         return []
#     # sort mark events in time
#     sorted_events = sort_events_by_field(events, 'ts')
#     # pair two event to a range
#     assert len(sorted_events) % 2 == 0, "two marks for one range"
#     range_events = []
#     for i in range(0, len(sorted_events), 2):
#         name_beg = sorted_events[i]['name'] 
#         name_end = sorted_events[i+1]['name']
#         assert name_beg[-3:] == 'BEG' and name_end[-3:] == 'END'
#         assert name_beg[:-4] == name_end[:-4]
#         event = {
#                     "name": name_beg[:-4],
#                     "ph": "X",
#                     "cat": "cpu",
#                     "ts": sorted_events[i]['ts'],
#                     "dur": sorted_events[i+1]['ts'] - sorted_events[i]['ts'],
#                     "tid": sorted_events[i]['tid'],
#                     "pid": sorted_events[i]['pid'],
#                     "cname": sorted_events[i]["cname"]
#                 }
#         range_events.append(event)
#     return range_events
# # # convert background markers to range
# traceEvents["MARKER_BKGD"] = marks_to_ranges(traceEvents["MARKER_BKGD"])
