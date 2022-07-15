# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# TODO: fix print and print_avg with less child than world sizes (e.g. bert_seq, P14, vPP, N4)
"""
====== bert_xxxx (rank xxxx) =============
Actual Time Breakdown (Note)
------------- GPU Kernel -----------------
SwapIn [s25]    :        0.62 s        0.81% (4.12 GB, 6.61 GB/s)
SwapIn [s27]    :        0.62 s        0.81% (4.12 GB, 6.61 GB/s)
SwapIn [s27]    :        0.62 s        0.81% (4.12 GB, 6.61 GB/s)
P2PIn           :        4.64 s        6.06% (includes wait time; contains both ^v)
Compute         :       42.42 s       55.44%
P2POut          :       44.07 s       57.59% (includes wait time; contains both ^v)
P2P             :
SwapOut [s7]    :        0.46 s        0.60% (3.86 GB, 8.46 GB/s)
SwapOut [s29]   :        0.46 s        0.60% (3.86 GB, 8.46 GB/s)
-------------- CPU Marker ----------------
SwapIn(W,B) 
SwapIn(#0StashX)
SwapIn(#0LocalX)
SwapIn(X*)                                    (contains StashX/LocalX/dY/Data/T)
P2PIn(#0X)
P2PIn(X*)                                     (contains X/dY)
FWD
FWDClean
Del(W,B)
Recompute
BWD
BWDClean
Del(W,dW,B)
P2POut(Y*)                                      (contains Y/dX)
AllReduce
SwapOut(#0StashX)
SwapOut(X*)                                     (contains StashX/Y/dX)
SwapOut(dW,B)
UPD
__UPD
P2PInAllociBcast
Alloc(W,B)      
__SyncPin(W,B) 
__CopyIn(W,B)   
Alloc(X*)                                       (contains StashX/LocalX/dY/Data/T)
__SyncCopyIn(X*)                                (contains StashX/LocalX/dY/Data/T)
PrefetchSuc(X*)                                 (contains StashX/LocalX/dY/Data/T)
__ConvertU(X)                                   (contains StashX/LocalX)
------------------------------------------
ProfOvrhd       :        0.73 s        0.96%
End2End         :        76.52 s      100.00%
==========================================
"""
# GPU
def _tid2namenote(tid, unify_swap=False):
    if tid.startswith("SwapIn"):
        if unify_swap:
            return "SwapIn", "may contain multiple streams"
        else:
            return tid, None
    elif tid.startswith("P2PIn"):
        return "P2PIn", "includes wait time; contains both ^v"
    elif tid.startswith("Compute"):
        return "Compute", "may contain other Compute streams"
    elif tid.startswith("P2POut"):
        return "P2POut", "includes wait time; contains both ^v"
    elif tid.startswith("P2P"):
        return "P2P", "may contain other P2P streams"
    elif tid.startswith("SwapOut"):
        if unify_swap:
            return "SwapOut", "may contain multiple streams"
        else:
            return tid, None
    else:
        print("Warning: Unknown GPU tid: {}".format(tid))
        return None, None
        # raise ValueError("Unknown tid: {}".format(tid))

# CPU
def _marker2namenote(marker):
    if marker.startswith("cudaProfiler"):
        return None, None
    elif "SwapIn(W,B)" in marker:
        return "SwapIn(W,B)", None
    elif "SwapIn(#0StashX)" in marker:
        return "SwapIn(#0StashX)", None
    elif "SwapIn(#0LocalX)" in marker:
        return "SwapIn(#0LocalX)", None
    elif "SwapIn(#" in marker:
        return "SwapIn(X*)", "contains StashX/LocalX/dY/Data/T"
    elif "MSGIn(#0X)" in marker:
        return "MSGIn(#0X)", None
    elif "MSGIn(#" in marker:
        return "MSGIn(X*)", "contains X/dY"
    elif "P2PIn(#0X)" in marker:
        return "P2PIn(#0X)", None
    elif "P2PIn(#" in marker:
        return "P2PIn(X*)", "contains X/dY"
    # elif "FWD(#" in marker:
    #     return "FWD", None
    # elif "FWDClean(#" in marker:
    #     return "FWDClean", None
    elif "Del(W,B)" in marker:
        return "Del(W,B)", None
    # elif "Recompute" in marker:
    #     return "Recompute", None
    # elif "BWD(#" in marker:
    #     return "BWD", None
    elif "BWDClean(#" in marker:
        return "BWDClean", None
    elif "Del(W,dW,B)" in marker:
        return "Del(W,dW,B)", None
    # elif "P2POut(#" in marker:
    #     return "P2POut(Y*)", "contains Y/dX"
    elif "MSGOut(#" in marker:
        return "MSGOut(Y*)", "contains Y/dX"
    elif "AllReduce" in marker:
        return "AllReduce", None
    elif "SwapOut(#0StashX)" in marker:
        return "SwapOut(#0StashX)", None
    elif "SwapOut(#" in marker:
        return "SwapOut(X*)", "contains StashX/Y/dX"
    elif "SwapOut(dW,B)" in marker:
        return "SwapOut(dW,B)", None
    # elif "UPD(" in marker and "__" != marker[:2]:
    #     return "UPD", None
    elif "UPD(" in marker and "__" == marker[:2]:
        return "__UPD", None
    # elif "PrefetchPt" in marker:
    #     return "PrefetchPt", None
    elif "P2PIn Alloc & iBcast" == marker:
        return "P2PInAllociBcast", None
    elif "Alloc(W,B)" in marker:
        return "Alloc(W,B)", None
    elif "__" == marker[:2] and "SyncPin(W,B)" in marker:
        return "__SyncPin(W,B)", None
    elif "__" == marker[:2] and "CopyIn(W,B)" in marker:
        return "__CopyIn(W,B)", None
    elif "Wait(X)" in marker:
        return "Wait(X*)", "contains StashX/LocalX/dY/Data/T"
    elif "Alloc(X)" in marker:
        return "Alloc(X*)", "contains StashX/LocalX/dY/Data/T"
    elif "__" == marker[:2] and "SyncCopyIn(X)" in marker:
        return "__SyncCopyIn(X*)", "contains StashX/LocalX/dY/Data/T"
    # elif marker.startswith("PrefetchSuc") and "X" in marker:
    #     return "PrefetchSuc(X*)", "contains StashX/LocalX/dY/Data/T"
    elif "__" == marker[:2] and "WaitCopyOut(X)" in marker:
        return "__WaitCopyOut(X*)", "contains StashX/LocalX/dY"
    elif "__" == marker[:2] and "ConvertU(X)" in marker:
        return "__ConvertU(X)", "contains StashX/LocalX"
    else:
        # print("Warning: Unknown CPU marker: {}".format(marker))
        return None, None
        # raise ValueError("Unknown Marker: {}".format(marker))

# in breakdown order
GPU_NAME_ORDER = [
"SwapIn",
"SwapIn",
"SwapIn",
"SwapIn",
"SwapIn",
"P2PIn",
"Compute",
"P2POut",
"P2P",
"P2P",
"SwapOut",
"SwapOut",
"SwapOut",
"SwapOut"] 
CPU_NAME_ORDER = [
"SwapIn(W,B)", 
"SwapIn(#0StashX)",
"SwapIn(#0LocalX)",
"SwapIn(X*)",
"MSGIn(#0X)",
"MSGIn(X*)",
"P2PIn(#0X)",
"P2PIn(X*)",
"FWD",
"FWDClean",
"Del(W,B)",
"Recompute",
"BWD",
"BWDClean",
"Del(W,dW,B)",
"P2POut(Y*)",
"MSGOut(Y*)",
"AllReduce",
"SwapOut(#0StashX)",
"SwapOut(X*)", 
"SwapOut(dW,B)",
"UPD",
"__UPD",
"PrefetchPt",
"P2PInAllociBcast",
"Alloc(W,B)",     
"__SyncPin(W,B)", 
"__CopyIn(W,B)",
"Wait(X*)",
"Alloc(X*)",
"__SyncCopyIn(X*)",
# "PrefetchSuc(X*)",
"__WaitCopyOut(X*)",
"__ConvertU(X)"] 
MIC_NAME_ORDER = [
'ProfOvrhd',
'End2End'] 

####################################################################
import numpy as np
from collections import OrderedDict as ODict
from helper import demunge_time
DEBUG=False

class Component(object):
    def __init__(self, name="", note=None):
        # assert name in GPU_NAME_ORDER+CPU_NAME_ORDER+MIC_NAME_ORDER
        self.name = name
        self.accum_time = 0.0 # ns
        self.accum_size = 0.0 # bytes
        self.percent_str = None
        self.tput_str = None
        self.note = note
    def add_time(self, dur): # ns
        assert isinstance(dur, (int, float))
        self.accum_time += dur
    def add_size(self, size): # bytes
        assert isinstance(size, (int, float))
        self.accum_size += size
    def cal_percent(self, end2end_time): # ns
        assert isinstance(end2end_time, (int, float))
        self.percent_str = "%2.1f%%" % (self.accum_time/end2end_time*100.0) # %
    def cal_throughput(self):
        self.tput_str = "%.2f GB/s" % (self.accum_size/1024.0/1024.0/1024.0 / (self.accum_time/1000000000.0) ) # GB/s
        return self.tput_str
    def __repr__(self):
        name_str = "%s" % self.name.ljust(16, ' ') # Make string left-justified of specificed length by padding spaces to the right of it
        accum_time_str = "%.2f s" % (self.accum_time/1000000000.0)
        accum_time_str = accum_time_str.rjust(10, ' ') # Make string right-justified of specified length by padding spaces to left
        percent_str = self.percent_str.rjust(7, ' ')
        str = "%s:\t%s\t%s" % (name_str, accum_time_str, percent_str)
        #
        if DEBUG: str += "\t<%.0f ns, %.0f byte>" % (self.accum_time, self.accum_size)
        #
        if self.note is not None:
            str += "\t(%s)" % self.note
        # 
        if self.accum_size != 0.0:
            str += "\t(%.2f GB, %s)" % (
                        self.accum_size/1024.0/1024.0/1024.0,
                        self.cal_throughput() )
        # if self.name in ["P2PIn","P2POut"]:
        #     str += "\t(includes wait time; contains both ^v)"
        # elif self.name == "Others":
        #     str += "\t(e.g. blocking cudaMalloc, hostMalloc)" #, FWD/BWD gaps)"
        return str

class BreakdownV2(object):
    def __init__(self, unify_swap=False):
        self.kind_rank_components = ODict()  
        # { 'gpu': { rank0 : { name0: Component0 } } } 
        self.kind_rank_components['gpu'] = ODict()  
        self.kind_rank_components['cpu'] = ODict()  
        self.kind_rank_components['mic'] = ODict()  
        self.rank_end2end = ODict() 
        # { rank0 : [profilerStartTime, profilerEndTime] --> end2end_time } # ns
        self.unify_swap = unify_swap
    
    def set_start_time(self, rank, timestamp): # ns
        assert isinstance(timestamp, (int, float))
        if not (rank in self.rank_end2end):
            self.rank_end2end[rank] = [0.0, 0.0]
        self.rank_end2end[rank][0] = float(timestamp)
        # print("rank%d: set_start_time"%rank)
        
    def set_end_time(self, rank, timestamp): # ns
        assert isinstance(timestamp, (int, float))
        if not (rank in self.rank_end2end):
            self.rank_end2end[rank] = [0.0, 0.0]
        self.rank_end2end[rank][1] = float(timestamp)
        # print("rank%d: set_end_time"%rank)

    def add_component_time(self, kind, rank, name, dur, size=None): # ns, bytes
        kind = kind.lower()
        if kind == 'gpu':
            name, note = _tid2namenote(name, self.unify_swap)
        elif kind == 'cpu':
            name, note = _marker2namenote(name)
        elif kind == 'mic':
            name, note = name, None
        else:
            assert False
        if name is None: 
            return
        rank_components = self.kind_rank_components[kind]
        if not (rank in rank_components):
            rank_components[rank] = ODict()
        if not (name in rank_components[rank]):
            rank_components[rank][name] = Component(name, note)
        rank_components[rank][name].add_time(dur)
        if size is not None:
            rank_components[rank][name].add_size(size)
    
    def _render_table(self, kind_components, title, rank):
        print("====== {} (rank{}) ======".format(title, rank))
        print("Actual Time Breakdown (Note)")
        print("------------------- GPU Kernel ------------------")
        names_with_stream = list(kind_components['gpu'].keys())
        names_without_stream = [ n.split(' ')[0] for n in names_with_stream ]
        for name in GPU_NAME_ORDER:
            if name in names_without_stream:
                idx = names_without_stream.index(name) # return index of the 1st occur
                names_without_stream.pop(idx) # pop out used component
                the_name = names_with_stream.pop(idx) 
                print(kind_components['gpu'][the_name])
        print("------------------ CPU Marker -------------------")
        for name in CPU_NAME_ORDER:
            if name in kind_components['cpu']:
                print(kind_components['cpu'][name])
        print("-------------------------------------------------")
        for name in MIC_NAME_ORDER:
            if name in kind_components['mic']:
                print(kind_components['mic'][name])
        print("=================================================\n")
    
    def print_component(self, rank, title="title", traceAll=None):
        # calcuate end2end time
        if rank in self.rank_end2end:
            if isinstance(self.rank_end2end[rank], list):
                start, end = self.rank_end2end[rank]
                assert start > 0 and end > 0
                assert end > start
                self.rank_end2end[rank] = float(end - start)
            elif isinstance(self.rank_end2end[rank], float):
                pass
            else: 
                raise ValueError
        else:
            print("rank{}: cudaProfilerStart/Stop not available; Calculating ...".format(rank))
            timestamps = [] # ns
            for event in traceAll:
                timestamps.append( demunge_time(event["ts"]) )
                if "dur" in event:
                    timestamps.append( demunge_time(event["ts"] + event["dur"]) )
            self.rank_end2end[rank] = float(max(timestamps) - min(timestamps))
            print("rank{}: end2end time calculated.".format(rank))
        end2end_time = self.rank_end2end[rank]
        # # add Others component
        # sum_time = sum(com.accum_time for com in self.rank_components[rank].values())
        # self.add_component_time(rank, "Others", end2end_time-sum_time)
        # add End2End component
        self.add_component_time('mic', rank, "End2End", end2end_time)
        # calcuate percent time %
        for _, rank_components in self.kind_rank_components.items():
            for com in rank_components[rank].values():
                com.cal_percent(end2end_time)
        # # calcuate throughput
        # for com in self.rank_components[rank].values():
        #     com.cal_throughput()
        # print each component in order
        kind_components = ODict()
        for kind, rank_components in self.kind_rank_components.items():
            kind_components[kind] = rank_components[rank]
        self._render_table(kind_components, title, rank)

    def print_avg_component(self, ranks, title="title"): 
        """ Averaging each componenet across child ranks """
        # get avg end2end time
        end2end_time = np.mean([float(self.rank_end2end[r]) for r in ranks])
        # create avg components
        kind_name_note = ODict() # { "gpu": {name: note} }
        for kind, rank_components in self.kind_rank_components.items():
            kind_name_note[kind] = ODict()
            for r in ranks:
                for com in rank_components[r].values():
                    kind_name_note[kind][com.name] = com.note
        #
        kind_avg_coms = ODict() # { "gpu": {name0: Component0} } 
        for kind, name_note in kind_name_note.items():
            kind_avg_coms[kind] = ODict()
            for name, note in name_note.items():
                kind_avg_coms[kind][name] = Component(name, note)
        #
        for kind, avg_coms in kind_avg_coms.items():
            rank_components = self.kind_rank_components[kind]
            for name, com in avg_coms.items():
                for r in ranks: 
                    if name in rank_components[r]:
                        com.add_time(rank_components[r][name].accum_time)
                        com.add_size(rank_components[r][name].accum_size)
                com.accum_time /= len(ranks)
                com.accum_size /= len(ranks)        
        # calcuate percent time %
        for avg_coms in kind_avg_coms.values():
            for com in avg_coms.values():
                com.cal_percent(end2end_time)
        # # calcuate throughput
        # for com in components.values():
        #     com.cal_throughput()
        # print each component in order
        self._render_table(kind_avg_coms, title, ranks)
                
