""" NOTE: 
- the timestamp in nvprof's nvvp file is elapsed time (ns) since 1970
- chrome://tracing's event time (ts & dur) is micro-sec (us)
- chrome://tracing viewer use 'displayTimeUnit' (default: ms)
- chrome://tracing viewer display all events offset from time = 0 (default: ms)
"""
""" MARKS format:
- cpu probe: **|**|**
- main thread: regular
- background: __****
- cuda mem probe: **/**
"""
import sqlite3
import argparse
import enum
import json
import subprocess
import os
import sys
import copy
from collections import OrderedDict as ODict
import gzip
import numpy as np
import time
#
from helper import *
from inspect_db import *
from cbids import WhiteCbids, CBIDS_WHITELIST
from colors import tid2cname, COLORS
# from breakdown import Breakdown
from breakdown_v2 import BreakdownV2

##############################################
def find_start_end_timestamp(nvvp_file):
    assert os.path.exists(nvvp_file)
    conn = sqlite3.connect(nvvp_file)
    conn.row_factory = sqlite3.Row
    # print("finding start and end timestamp from {}".format(nvvp_file))
    # print strings dictionary for kernel names
    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        strings[r["id"]] = demangle_string(r["value"])
    # === MARKER and RANGE ===
    ts_start, ts_end = None, None
    for row in conn.execute(" ".join([
            "SELECT",
            ",".join([
                "start.name AS name",
                "start.timestamp AS start_time",
                "end.timestamp AS end_time"
            ]),
            "FROM",
            "(SELECT * FROM CUPTI_ACTIVITY_KIND_MARKER WHERE name != 0) AS start",
            "LEFT JOIN",
            "(SELECT * FROM CUPTI_ACTIVITY_KIND_MARKER WHERE name = 0) AS end",
            "ON start.id = end.id"])):
        ev_name = strings[row["name"]]
        # get end2end time
        if ev_name == 'cudaProfilerStart':
            ts_start = row["start_time"] # ns
        elif ev_name == 'cudaProfilerStop':
            ts_end = row["start_time"] # ns
    #
    assert (ts_start is not None) and (ts_end is not None)
    # print("found: ts_start=%d, ts_end=%d (end2end_time=%.3f sec)"%(
    #              ts_start, ts_end, (ts_end-ts_start)/1000000000.))
    return ts_start, ts_end # ns

def convert_parent(nvvp_file, rank, world_size, ts_start=0, ts_end=float('inf')):
    """ convert parent nvvp (rank) within [ts_start, ts_end] to chrome-trace.json """
    assert os.path.exists(nvvp_file)
    conn = sqlite3.connect(nvvp_file)
    conn.row_factory = sqlite3.Row
    # get pid from filename '*pid%d.nvvp' or '*pid%d_*.nvvp' or '*pid%d-*.nvvp' 
    fname = os.path.basename(nvvp_file)
    assert 'pid' in fname, "converting parent rank requires pid in filename"
    str_has_pid = fname.split('pid')[1]
    str_pid = []
    for char in str_has_pid:
        if '0' <= char and char <= '9':
            str_pid.append(char)
        else:
            break
    pid = int("".join(str_pid))
    print("converting {}: pid {}, rank {}, world_size {}".format(nvvp_file, pid, rank, world_size))
    # inspect nvvp database
    # inspect_db(conn, deep_inspect=False) # DEBUG
    # print strings dictionary for kernel names
    # t1 = time.time()
    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        strings[r["id"]] = demangle_string(r["value"])
    # t2 = time.time()
    # print("[DEBUG] demangle_string costs %.3f sec"% (t2-t1) )
    # # # convert nvvp database to chrome trace events # # #
    traceEvents = ODict()
    counterEvents = ODict()
    # # === OVERHEAD === # empty
    # traceEvents["OVERHEAD"] = []
    # for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_OVERHEAD"):
    #     if  row["start"] < ts_start or row["end"] > ts_end:
    #         continue
    #     # within [ts_start, ts_end]
    #     dur_ns = row["end"] - row["start"]
    #     event = {
    #             "name": "profiling overhead",
    #             "ph": "X", # Complete Event (Begin + End event)
    #             "cat": "profiling",
    #             "ts": munge_time(row["start"]),
    #             "dur": munge_time(dur_ns),
    #             "tid": "Profiling Overhead",
    #             "pid": "Rank{}".format(rank),
    #             }
    #     event["cname"] = tid2cname(event['tid'])
    #     traceEvents["OVERHEAD"].append(event)
    # === MARKER and RANGE ===
    counterEvents["TOTAL_CPU_UTIL"] = []
    for r in range(world_size):
        counterEvents["RANK%d_CPU_UTIL"%r] = []
    traceEvents["MARKER_CPU"] = []
    cpu_util_avg = CpuUtilAvg()
    for row in conn.execute(" ".join([
            "SELECT",
            ",".join([
                "start.name AS name",
                "start.timestamp AS start_time",
                "end.timestamp AS end_time"
            ]),
            "FROM",
            "(SELECT * FROM CUPTI_ACTIVITY_KIND_MARKER WHERE name != 0) AS start",
            "LEFT JOIN",
            "(SELECT * FROM CUPTI_ACTIVITY_KIND_MARKER WHERE name = 0) AS end",
            "ON start.id = end.id"])):
        if row["start_time"] < ts_start or row["start_time"] > ts_end:
            continue
        # within [ts_start, ts_end]
        ev_name = strings[row["name"]]
        if ('|' in ev_name): # marks for probe cpu
            ev_ts = munge_time(row["start_time"])
            (total_percent, rank_percent, total_cnt) = parse_cpu_probe(ev_name)
            #
            cnt_event = {
                "name": "Total CPU Util (%)",
                "ph": "C", # Counter Event
                "cat": "cpu",
                "ts": ev_ts,
                "pid": ": Total CPU Stat",
                "args": { "total cpu util": total_percent },
                }
            cnt_event["cname"] = tid2cname(cnt_event['name'])
            counterEvents['TOTAL_CPU_UTIL'].append(cnt_event)
            #
            assert len(rank_percent) == world_size
            for r in range(world_size):
                cnt_event = {
                "name": "CPU Util (%)",
                "ph": "C", # Counter Event
                "cat": "cpu",
                "ts": ev_ts,
                "pid": ": Rank{} CPU Stat".format(r),
                "args": { "cpu util": clip_raw_util(rank_percent[r], total_cnt) }, # see Note in probe_cpu.py
                }
                cnt_event["cname"] = tid2cname(cnt_event['name'])
                counterEvents["RANK%d_CPU_UTIL"%r].append(cnt_event)
            #
            event = {
                    "name": "%.1f%% | %s | %d" % (
                            total_percent, 
                            ",".join("%.1f%%"%percent for percent in rank_percent), 
                            total_cnt),
                    "ph": "I", # Instant Event
                    "cat": "cpu",
                    "ts": ev_ts,
                    "tid": "CPU Probe",
                    "pid": ": Total CPU Stat",
                    "args": { "processId": "%d"% int(pid) },
                    }
            event["cname"] = tid2cname(event['tid'])
            traceEvents["MARKER_CPU"].append(event)
            # 
            cpu_util_avg.add(total_percent, rank_percent)
    # # # left shift utilization number
    counterEvents['TOTAL_CPU_UTIL'] = periodize_counter_value(counterEvents['TOTAL_CPU_UTIL'], pad_max_value={"total cpu util":total_cnt*100.} ) 
    for r in range(world_size):
        counterEvents["RANK%d_CPU_UTIL"%r] = periodize_counter_value(counterEvents["RANK%d_CPU_UTIL"%r], pad_max_value={"cpu util":total_cnt*100.+1}) 
    # ==============
    # Arrange display order
    traceAll = []
    # traceAll += traceEvents["OVERHEAD"]
    # --- Total CPU Stat ---
    traceAll += counterEvents["TOTAL_CPU_UTIL"] # counter always tops
    traceAll += traceEvents["MARKER_CPU"]      
    # ===============
    # save results for Rank CPU Stat
    rank_cpuUtilEvents = ODict() # { rank: counter events of CPU Util }
    for r in range(world_size):
        rank_cpuUtilEvents[r] = counterEvents["RANK%d_CPU_UTIL"%r] 
    # print("converted: {}".format(nvvp_file))
    #
    cpu_util_avg.print(title="Raw CPU Util (%) Avg")
    return traceAll, rank_cpuUtilEvents
    # change process name > doesn't work
    # metaEvent = {   "name": "Process Rank-1 Total CPU Stat", 
    #                 "ph": "M", 
    #                 "pid": "Rank-1 Total CPU Stat", 
    #                 # "tid": 2347,
    #                 "args": { "name" : "RendererProcess"} }
    # traceAll += metaEvent   

def convert_child(nvvp_file, rank, world_size, mode, reverse_bwd, break_down, per_gpu_memory_cap, cpuUtilEvents=[]):
    """ convert one nvvp file (rank) to chrome-trace.json """
    # get CONFIGS from filename
    fname = os.path.basename(nvvp_file)
    # reverse_bwd = False if "inRR" in fname or  else True
    #
    assert os.path.exists(nvvp_file)
    conn = sqlite3.connect(nvvp_file)
    conn.row_factory = sqlite3.Row
    # get pid
    pid = -1
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        pid = int(row["processId"])
        break
    assert pid != -1
    print("converting {}: pid {}, rank {}, world_size {}, mode {}".format(nvvp_file, pid, rank, world_size, mode))
    # inspect nvvp database
    # inspect_db(conn, deep_inspect=False) # DEBUG
    # print strings dictionary for kernel names
    # t1 = time.time()
    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        strings[r["id"]] = demangle_string(r["value"])
    # t2 = time.time()
    # print("[DEBUG] demangle_string costs %.3f sec"% (t2-t1) )
    # # # convert nvvp database to chrome trace events # # #
    traceEvents = ODict()
    counterEvents = ODict()
    # === OVERHEAD ===
    """
    _id_: 500
    overheadKind: 131072
    objectKind: 1
    objectId: b'\xf2&\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    start: 1624406786345062842
    end: 1624406786345158214
    """
    # https://docs.nvidia.com/cupti/Cupti/annotated.html#structCUpti__ActivityOverhead
    traceEvents["OVERHEAD"] = []
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_OVERHEAD"):
        dur_ns = row["end"] - row["start"]
        event = {
                "name": "profiling overhead",
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "profiling",
                "ts": munge_time(row["start"]),
                "dur": munge_time(dur_ns),
                "tid": "Profiling Overhead",
                "pid": ": Rank{} Profiling".format(rank),
                }
        event["cname"] = tid2cname(event['tid'])
        traceEvents["OVERHEAD"].append(event)
        break_down.add_component_time('mic', rank, 'ProfOvrhd', dur_ns)
    # === RUNTIME API ===
    """
    _id_: 11625
    cbid: 17
    start: 1496933427584362152
    end: 1496933427584362435
    processId: 1317533
    threadId: 1142654784
    correlationId: 13119
    returnValue: 0
    """
    traceEvents['RUNTIME'] = []
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        # be selective
        assert row["processId"] == pid
        cbid = int(row["cbid"])
        if cbid in CBIDS_WHITELIST:
            event = {
                    "name": WhiteCbids(cbid).name,
                    "ph": "X", # Complete Event (Begin + End event)
                    "cat": "cpu",
                    "ts": munge_time(row["start"]),
                    "dur": munge_time(row["end"] - row["start"]),
                    "tid": "CUDA API",
                    "pid": ": Rank{} CPU".format(rank),
                    "args": {
                        "threadId": "%d"% int(row["threadId"]),
                        "processId": "%d"% int(pid),
                        },
                    }            
            event["cname"] = tid2cname(event['tid'])
            if event['name'].startswith("cudaFree") and not event['name'].endswith("Host"):
                event["cname"] = COLORS[8][0]
                print("Warning!! %s" % (event['name']))
            elif event['name'].startswith("cudaMalloc") and not event['name'].endswith("Host"):
                event["cname"] = COLORS[9][0]
                print("Warning! %s" % (event['name']))
            elif event['name'].startswith("cudaStreamCreate"):
                event["cname"] = COLORS[21][0]
                print("Warning: %s" % (event['name']))
            traceEvents['RUNTIME'].append(event)
        # > doesn't work
        # if cbid == 171: # cudaProfilerStart = 171 
        #     break_down.set_start_time(rank, row["start"]) # ns
        # elif cbid == 172: # cudaProfilerStop = 172
        #     break_down.set_end_time(rank, row["start"]) # ns
    # === DRIVER API ===
    # === MARKER and RANGE ===
    """
    _id_: 1
    flags: 2
    timestamp: 1496844806028263989
    id: 1
    objectKind: 2
    objectId: b'\xe5\xc0\x16\x00@\xe7\x10J\x00\x00\x00\x00'
    name: 3
    domain: 0
    """
    traceEvents["MARKER_MAIN"] = [] # main thread
    traceEvents["MARKER_BKGD"] = [] # background
    counterEvents["PEAK_ALLOC"] = [] # probe cuda memory
    counterEvents["PEAK_RESRV"] = [] # probe cuda memory
    traceEvents["MARKER_MEM"] = [] # probe cuda memory
    for row in conn.execute(" ".join([
            "SELECT",
            ",".join([
                "start.name AS name",
                "start.timestamp AS start_time",
                "end.timestamp AS end_time"
            ]),
            "FROM",
            "(SELECT * FROM CUPTI_ACTIVITY_KIND_MARKER WHERE name != 0) AS start",
            "LEFT JOIN",
            "(SELECT * FROM CUPTI_ACTIVITY_KIND_MARKER WHERE name = 0) AS end",
            "ON start.id = end.id"])):
        ev_name = strings[row["name"]]
        if ev_name[:2] == "__":
            marker_type = "BKGD"
        elif '/' in ev_name:
            marker_type = "MEM"
        else:
            marker_type = "MAIN"
        if marker_type == "MAIN": # main markers & ranges
            event = {
                    "name": ev_name,
                    "cat": "cpu",
                    "ts": munge_time(row["start_time"]),
                    # Weirdly, these don't seem to be associated with a
                    # CPU/GPU.  I guess there's no CUDA Context available
                    # when you run these, so it makes sense.  But nvvp
                    # associates these with a GPU strangely enough
                    "tid": "Main",
                    "pid": ": Rank{} CPU".format(rank),
                    }
            event["cname"] = tid2cname(event['tid'])
            if row["end_time"] is None:
                event["ph"] = "I"
                dur_ns = 0
            else:
                event["ph"] = "X"
                dur_ns = row["end_time"] - row["start_time"]
                event["dur"] = munge_time(dur_ns)
            traceEvents["MARKER_MAIN"].append(event)
            # get end2end time
            if event['name'] == 'cudaProfilerStart':
                break_down.set_start_time(rank, row["start_time"]) # ns
            elif event['name'] == 'cudaProfilerStop':
                break_down.set_end_time(rank, row["start_time"]) # ns
            # get component breakdown
            break_down.add_component_time('cpu', rank, ev_name, dur_ns)
            # if 'FWDClean' in event['name']: 
            #     break_down.add_component_time('cpu', rank, 'FWDClean', dur_ns)
            # elif 'Del(W,B)' in event['name']:
            #     break_down.add_component_time('cpu', rank, 'FWDDelW', dur_ns)
            # elif 'BWDClean' in event['name']:
            #     break_down.add_component_time('cpu', rank, 'BWDClean', dur_ns)
            # elif 'Del' in event['name'] and 'dW' in event['name']:
            #     break_down.add_component_time('cpu', rank, 'BWDDelW', dur_ns)
            # elif 'UPD' in event['name']:
            #     pass # break_down.add_component_time('cpu', rank, 'UpdateW', dur_ns)
        elif marker_type == "BKGD": # background marks
            event = {
                    "name": ev_name,
                    "cat": "cpu",
                    "ts": munge_time(row["start_time"]),
                    "tid": "Background",
                    "pid": ": Rank{} CPU".format(rank),
                    }
            if "UPD" in ev_name:
                event["cname"] = COLORS[5][0]
            else:
                event["cname"] = tid2cname(event['tid'])
            if row["end_time"] is None:
                event["ph"] = "I"
                dur_ns = 0
            else:
                event["ph"] = "X"
                dur_ns = row["end_time"] - row["start_time"]
                event["dur"] = munge_time(dur_ns)
            traceEvents["MARKER_BKGD"].append(event)
            # get component breakdown
            break_down.add_component_time('cpu', rank, ev_name, dur_ns)
            # if "UPD" in ev_name:
            #     break_down.add_component_time('cpu', rank, 'UpdateW', dur_ns)
        elif marker_type == "MEM": # marks for probe cuda memory
            ev_ts = munge_time(row["start_time"])
            peak_allocated_percent, peak_reserved_percent, peak_allocated_mb, peak_reserved_mb = parse_memory_probe(ev_name, per_gpu_memory_cap)
            cnt_event = {
                "name": "Peak Allocated (%)",
                "ph": "C", # Counter Event
                "cat": "cuda",
                "ts": ev_ts,
                "pid": ": Rank{} GPU Memory".format(rank),
                "args": { "peak allocated memory": peak_allocated_percent },
                }
            cnt_event["cname"] = tid2cname(cnt_event['name'])
            counterEvents['PEAK_ALLOC'].append(cnt_event)
            #
            cnt_event = {
                "name": "Peak Reserved (%)",
                "ph": "C", # Counter Event
                "cat": "cuda",
                "ts": ev_ts,
                "pid": ": Rank{} GPU Memory".format(rank),
                "args": { "peak reserved memory": peak_reserved_percent },
                }
            cnt_event["cname"] = tid2cname(cnt_event['name'])
            counterEvents['PEAK_RESRV'].append(cnt_event)
            #
            event = {
                    "name": "%d MB / %d MB" % (peak_allocated_mb, peak_reserved_mb),
                    "ph": "I", # Instant Event
                    "cat": "cpu",
                    "ts": ev_ts,
                    "tid": "GPU Mem Probe",
                    "pid": ": Rank{} GPU Memory".format(rank),
                    }
            event["cname"] = tid2cname(event['tid'])
            traceEvents["MARKER_MEM"].append(event)
    # # # probe cuda memory: left shift counter values
    counterEvents['PEAK_ALLOC'] = periodize_counter_value(counterEvents['PEAK_ALLOC'], pad_max_value={'peak allocated memory': 100.})
    counterEvents['PEAK_RESRV'] = periodize_counter_value(counterEvents['PEAK_RESRV'], pad_max_value={'peak reserved memory': 100.})
    
    # === CONCURRENT_KERNEL (compute & nccl) ===
    # kernel name: index into StringTable
    """
    _id_: 1
    cacheConfig: b'\x00'
    sharedMemoryConfig: 1
    registersPerThread: 32
    partitionedGlobalCacheRequested: 2
    partitionedGlobalCacheExecuted: 2
    start: 1496844806032514222
    end: 1496844806032531694
    completed: 1496844806032531694
    deviceId: 0
    contextId: 1
    streamId: 7
    gridX: 57
    gridY: 1
    gridZ: 1
    blockX: 128
    blockY: 1
    blockZ: 1
    staticSharedMemory: 0
    dynamicSharedMemory: 0
    localMemoryPerThread: 0
    localMemoryTotal: 78643200
    correlationId: 487
    gridId: 669
    name: 5
    """
    # separate traceEvents by Stream ID
    compute_stream_ids, nccl_stream_ids= set(), set()
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
        # eprintRow(row) # DEBUG
        assert int(row["contextId"]) == 1
        event = {
                    "name": strings[row["name"]],
                    "ph": "X", # Complete Event (Begin + End event)
                    "cat": "cuda",
                    "ts": munge_time(row["start"]),
                    "dur": row["end"] - row["start"], # delay munge_time
                    # "tid": "{} [s{}]".format(stream2name[id], id),
                    "pid": ": Rank{} GPU".format(rank)
                }
        #
        id = int(row["streamId"])
        if "nccl" in event["name"]:
            nccl_stream_ids.add(id)
        else:
            compute_stream_ids.add(id)
            event["tid"] = "Compute [s{}]".format(id)
            event["cname"] = tid2cname(event['tid'])
            dur_ns = event["dur"]
            event["dur"] = munge_time(dur_ns)
            break_down.add_component_time('gpu', rank, event["tid"], dur_ns)
        #
        k = "CONCUR_KERNEL_STREAM%d"%id
        if not (k in traceEvents):
            traceEvents[k] = []
        traceEvents[k].append(event)
    # identify compute stream
    compute_stream_ids = sorted(list(compute_stream_ids)) # ascend order is time order
    assert compute_stream_ids==[] or min(compute_stream_ids) in [0,7]
    # COMPUTE_STREAM_ID = min(compute_stream_ids)
    # assert COMPUTE_STREAM_ID in [0,7]
    # print("rank{}: found COMPUTE_STREAM_ID = {}".format(rank, COMPUTE_STREAM_ID))
    # for event in traceEvents["CONCUR_KERNEL_STREAM%d"%COMPUTE_STREAM_ID]:
    #     event["tid"] = "Compute [s{}]".format(COMPUTE_STREAM_ID)
    #     event["cname"] = tid2cname(event['tid'])
    #     dur_ns = event["dur"]
    #     event["dur"] = munge_time(dur_ns)
    #     break_down.add_component_time('gpu', rank, event["tid"], dur_ns)
    #     # break_down.add_component_time('gpu', rank, 'Compute', dur_ns)
    # stream_ids.remove(COMPUTE_STREAM_ID)
    # identify nccl streams
    nccl_stream_ids = sorted(list(nccl_stream_ids)) # ascend order is time order
    use_p2pin_p2pout = False
    if mode == 'vPP':
        if (reverse_bwd and len(nccl_stream_ids)==4) or \
           (not reverse_bwd and len(nccl_stream_ids)==2):
            use_p2pin_p2pout = True
    if use_p2pin_p2pout:
        nccl_stream_names = vPP_P2P_stream_names(rank, world_size, reverse_bwd) # ["P2PIn/Out_v/^"] in time order
        assert len(nccl_stream_ids) == len(nccl_stream_names)
        nccl_stream_name2id = ODict()
        for id, name in zip(nccl_stream_ids, nccl_stream_names):
            for event in traceEvents["CONCUR_KERNEL_STREAM%d"%id]:
                event["tid"] = "{} [s{}]".format(name, id)
                event["cname"] = tid2cname(event['tid'])
                dur_ns = event["dur"]
                event["dur"] = munge_time(dur_ns)
                break_down.add_component_time('gpu', rank, event["tid"], dur_ns)
                # break_down.add_component_time('gpu', rank, name.split('_')[0], dur_ns) 
            nccl_stream_name2id[name] = id
    else: # vDP, others
        for id in nccl_stream_ids:
            for event in traceEvents["CONCUR_KERNEL_STREAM%d"%id]:
                event["tid"] = "P2P [s{}]".format(id)
                event["cname"] = tid2cname(event['tid'])
                dur_ns = event["dur"]
                event["dur"] = munge_time(dur_ns)
                break_down.add_component_time('gpu', rank, event["tid"], dur_ns)
                # break_down.add_component_time('gpu', rank, "P2P", dur_ns)
    # === KERNEL ===
    # cnt = 0
    # for _ in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
    #     cnt += 1
    # assert cnt == 0
    # === MEMCPY === 
    """
    _id_: 1
    copyKind: 1
    srcKind: 1
    dstKind: 3
    flags: 0
    bytes: 7436640
    start: 1496933426915778221
    end: 1496933426916558424
    deviceId: 0
    contextId: 1
    streamId: 7
    correlationId: 809
    runtimeCorrelationId: 0
    """
    traceEvents["MEMCPY_H2D"] = []
    traceEvents["MEMCPY_D2H"] = []
    traceEvents["MEMCPY_D2D"] = []
    # traceEvents["MEMCPY_Other"] = []
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        #eprintRow(row)
        assert int(row["contextId"]) == 1
        # copyKind:
        #   1 - Memcpy HtoD
        #   2 - Memcpy DtoH
        #   8 - Memcpy DtoD
        # flags: ???
        #   0 - Sync
        #   1 - Async
        # srcKind/dstKind
        #   1 - Pageable
        #   2 - Page-locked ???
        #   3 - Device
        if row["flags"] == 0:
            flags = "sync"
        elif row["flags"] == 1:
            flags = "async"
        else:
            flags = str(row["flags"])
        #
        dur_ns = row["end"] - row["start"]
        event = {
                # "name": "Memcpy {} [{}]".format(copyKind, flags),
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "cuda",
                "ts": munge_time(row["start"]),
                "dur": munge_time(dur_ns),
                # "tid": "MemCpy ({})".format(copyKind), #
                "pid": ": Rank{} GPU".format(rank),
                "args": {
                    "Size": sizeof_fmt(row["bytes"]),
                    "Throughput": throughput(row["bytes"], dur_ns)
                    },
                }
        #
        if row["copyKind"] == 1:
            copyKind = "HtoD"
            event['name'] = "Memcpy {} [{}]".format(copyKind, flags)
            event["tid"] = "SwapIn [s{}]".format(int(row['streamId']))
            event["cname"] = tid2cname(event['tid'])
            traceEvents['MEMCPY_H2D'].append(event)
            break_down.add_component_time('gpu', rank, event["tid"], dur_ns, row["bytes"])
            # break_down.add_component_time('gpu', rank, "SwapIn", dur_ns, row["bytes"])
        elif row["copyKind"] == 2:
            copyKind = "DtoH"
            event['name'] = "Memcpy {} [{}]".format(copyKind, flags)
            event["tid"] = "SwapOut [s{}]".format(int(row['streamId']))
            event["cname"] = tid2cname(event['tid'])
            traceEvents['MEMCPY_D2H'].append(event)
            break_down.add_component_time('gpu', rank, event["tid"], dur_ns, row["bytes"])
            # break_down.add_component_time('gpu', rank, "SwapOut", dur_ns, row["bytes"])
        elif row["copyKind"] == 8:
            copyKind = "DtoD"
            event['name'] = "Memcpy {} [{}]".format(copyKind, flags)
            # assert int(row['streamId']) == COMPUTE_STREAM_ID # assert stream2name[int(row['streamId'])] == "Compute"
            event["tid"] = "Compute [s{}]".format(int(row['streamId']))
            event["cname"] = tid2cname(event['tid'])
            traceEvents['MEMCPY_D2D'].append(event)
            break_down.add_component_time('gpu', rank, event["tid"], dur_ns)
            # break_down.add_component_time('gpu', rank, "Compute", dur_ns)
        else:
            raise ValueError("Unknow copyKind {} of MEMCOPY".format(row["copyKind"]))
    # === MEMCPY2 ===
    # cnt = 0
    # for _ in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY2"):
    #     cnt += 1
    # assert cnt == 0
    # === MEMSET ===
    """
    _id_: 1
    value: 0
    bytes: 8
    start: 1624059262851283834
    end: 1624059262851285562
    deviceId: 0
    contextId: 1
    streamId: 7
    correlationId: 536
    flags: 0
    memoryKind: 3
    """
    traceEvents["MEMSET"] = []
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET"):
        # assert int(row['streamId']) == COMPUTE_STREAM_ID # assert stream2name[int(row['streamId'])] == "Compute"
        assert int(row["contextId"]) == 1
        dur_ns = row["end"] - row["start"]
        event = {
                "name": "Memset",
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "cuda",
                "ts": munge_time(row["start"]),
                "dur": munge_time(dur_ns),
                "tid": "Compute [s{}]".format(int(row['streamId'])),
                "pid": ": Rank{} GPU".format(rank),
                "args": {
                    "Size": sizeof_fmt(row["bytes"]),
                    "Throughput": throughput(row["bytes"], dur_ns)
                    },
                }
        event["cname"] = tid2cname(event['tid'])
        traceEvents["MEMSET"].append(event)
        break_down.add_component_time('gpu', rank, event["tid"], dur_ns)
        # break_down.add_component_time('gpu', rank, "Compute", dur_ns)
    # ==============
    # Arrange display order
    traceAll = []
    traceAll += traceEvents["OVERHEAD"]
    # --- CPU Stat ---
    traceAll += cpuUtilEvents # can be empty
    # --- CPU ---
    traceAll += traceEvents['RUNTIME']
    traceAll += traceEvents["MARKER_MAIN"]
    traceAll += traceEvents["MARKER_BKGD"]
    # --- GPU ---
    traceAll += traceEvents["MEMCPY_H2D"]
    if use_p2pin_p2pout: # mode == 'vPP' and len(nccl_stream_ids) == 4 or 2:
        if reverse_bwd:
            traceAll += traceEvents["CONCUR_KERNEL_STREAM%d" % nccl_stream_name2id['P2PIn_^']]
        traceAll += traceEvents["CONCUR_KERNEL_STREAM%d"%nccl_stream_name2id['P2PIn_v']]
        for id in compute_stream_ids:
            traceAll += traceEvents["CONCUR_KERNEL_STREAM%d"%id]
        traceAll += traceEvents["MEMCPY_D2D"] + traceEvents["MEMSET"]
        traceAll += traceEvents["CONCUR_KERNEL_STREAM%d"%nccl_stream_name2id['P2POut_v']]
        if reverse_bwd:
            traceAll += traceEvents["CONCUR_KERNEL_STREAM%d" % nccl_stream_name2id['P2POut_^']]
    else: # vDP, others
        for id in compute_stream_ids:
            traceAll += traceEvents["CONCUR_KERNEL_STREAM%d"%id]
        traceAll += traceEvents["MEMCPY_D2D"] + traceEvents["MEMSET"]
        for id in nccl_stream_ids:
            traceAll += traceEvents["CONCUR_KERNEL_STREAM%d"%id]
    traceAll += traceEvents["MEMCPY_D2H"]
    # --- GPU Memory ---
    traceAll += counterEvents["PEAK_ALLOC"] # counter always tops
    traceAll += counterEvents["PEAK_RESRV"]
    traceAll += traceEvents["MARKER_MEM"]
    #
    # print("converted: {}".format(nvvp_file))
    return traceAll
    # # Try metaEvent to order threads (doesnt' work for counter event)
    # metaEvent = {
    #             "name": "thread_order0",
    #             "ph": "M",
    #             "tid": "SwapOut [s{}]".format(7),
    #             "pid": "{}: Rank{} GPU".format(pid, rank),
    #             "args": { "sort_index": 0 }
    #             }
    # traceAll += metaEvent
    # metaEvent = {
    #             "name": "thread_order1",
    #             "ph": "M",
    #             "tid": "PeakAllocated (MB)",
    #             "pid": "{}: Rank{} GPU".format(pid, rank),
    #             "args": { "sort_index": 100 }
    #             }
    # traceAll += metaEvent

##############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View Harmony Runtime in Chrome Trace, by converting nvprof output to Google-Event-Trace compatible JSON.')
    parser.add_argument("--dir_nvvps", type=str, required=True, 
                        help="the source directory containing nvvp files")
    parser.add_argument("--nvvps", type=str, nargs='+', required=True, 
                        help="a list of source filenames." 
                             "case-1 = [child0, child1, ...]"
                             "case-2 = [parent]"
                             "case-3 = [parent, child0, child1, ... ]")
    parser.add_argument("--ranks", type=int, nargs='+', required=True, 
                        help="a list of ranks matching --nvvps"
                             "parant is -1; childs are  0, 1, ...")
    parser.add_argument("--world_size", type=int, required=True, 
                        help="total number of children ranks (without skip)")
    parser.add_argument("--mode", type=str, default="other", 
                        help="vDP: one P2P stream"
                             "vPP: four P2P streams"
                             "other: abitrary P2P streams")
    parser.add_argument("--reverse_bwd", action="store_true", help="[Deprecated]")
    parser.add_argument("--dir_jsons", type=str, required=True, 
                        help="the destination directory for converted json files")
    parser.add_argument("--json", type=str, required=True, 
                        help="the single json filename of all ranks")
    parser.add_argument("--per_gpu_memory_cap", type=float, default=1.172204E10, 
                        help="bytes")
    parser.add_argument("--unify_swap", action="store_true", 
                        help="unify different swap in/out streams for simpler view")
    args = parser.parse_args()
    #
    assert os.path.exists(args.dir_nvvps)
    assert len(args.nvvps) == len(args.ranks)
    assert len(args.nvvps) != 0
    # if args.ranks[0] == -1:
    #     assert len(args.ranks[1:]) <= args.world_size
    # else:
    #     assert len(args.ranks) <= args.world_size
    assert args.mode in ['vDP','vPP','other']
    assert os.path.exists(args.dir_jsons)
    
    # #  # start convertion
    t_start = time.time() #===
    total_traceAll = []    
    #
    if args.ranks[0] != -1:
        print("case-1: only child")
        child_nvvps, child_ranks = args.nvvps, args.ranks
        # convert child
        break_down = BreakdownV2(args.unify_swap)
        for nvvp, rank in zip(child_nvvps,child_ranks):
            traceAll = convert_child(os.path.join(args.dir_nvvps,nvvp), rank, args.world_size, args.mode, args.reverse_bwd, break_down, args.per_gpu_memory_cap)
            total_traceAll += traceAll
            break_down.print_component(rank, nvvp, traceAll)
        break_down.print_avg_component(child_ranks, args.json)
    elif args.ranks == [-1]:
        print("case-2: only parent")
        parent_nvvp, parent_rank = args.nvvps[0], -1
        # convert parent
        traceAll, rank_cpuUtilEvents = convert_parent( os.path.join(args.dir_nvvps, parent_nvvp), parent_rank, args.world_size )
        total_traceAll += traceAll
        for events in rank_cpuUtilEvents.values():
            total_traceAll += events
    else:
        print("case-3: both parent and childs")
        parent_nvvp, parent_rank = args.nvvps[0], -1
        child_nvvps, child_ranks = args.nvvps[1:], args.ranks[1:]
        # find child's timewindow
        ts_start, ts_end = find_start_end_timestamp( os.path.join(args.dir_nvvps, child_nvvps[0]) )
        # convert parent
        traceAll, rank_cpuUtilEvents = convert_parent( os.path.join(args.dir_nvvps, parent_nvvp), parent_rank, args.world_size, ts_start, ts_end )
        total_traceAll += traceAll
        # convert child
        break_down = BreakdownV2(args.unify_swap)
        for nvvp, rank in zip(child_nvvps,child_ranks):
            traceAll = convert_child(os.path.join(args.dir_nvvps,nvvp), rank, args.world_size, args.mode, args.reverse_bwd, break_down, args.per_gpu_memory_cap, rank_cpuUtilEvents[rank])
            total_traceAll += traceAll
            break_down.print_component(rank, nvvp, traceAll)
        break_down.print_avg_component(child_ranks, args.json)    
    
    # # # write to a single json file
    path_json = os.path.join(args.dir_jsons,args.json)
    if '.gz' in args.json:
        with gzip.open(path_json, 'w') as f: # gzipped json
            f.write(json.dumps(total_traceAll).encode('utf-8'))
        # why need to compress?
        # > google trace-viewer has 256MB file limit: https://github.com/google/trace-viewer/issues/627
        # > google trace-viewer has Large File Silent Failure: https://github.com/google/trace-viewer/issues/298
        # If compressed too small (<5KB), trace-viewr doesn't accept it either.
        if os.stat(path_json).st_size < 5000:
            path_json = path_json.replace(".gz", "")
            with open(path_json, 'w') as f: # pure json
                json.dump(total_traceAll, f)
    else:
        with open(path_json, 'w') as f: # pure json
            json.dump(total_traceAll, f)
    t_end = time.time() #===
    print("time cost: %.3f sec"%(t_end - t_start))
    print("chrome-trace.json dumped to %s (%.3f MB)"%(path_json, float(os.stat(path_json).st_size/1024./1024.)))


# # # old code
# if traceEvents["MARKER_MEM"] != []:
#     # sort probed memory counter in time
#     counterEvents['PEAK_ALLOC'] = sort_events_by_field(counterEvents['PEAK_ALLOC'], 'ts')
#     counterEvents['PEAK_RESRV'] = sort_events_by_field(counterEvents['PEAK_RESRV'], 'ts')
#     # move counter value one event ahead (for peak memory period)
#     left_shift_counter_value_by_one(counterEvents['PEAK_ALLOC'])
#     left_shift_counter_value_by_one(counterEvents['PEAK_RESRV'])
#     # append dummies to the last counter for 100% height display
#     ev_dummy = copy.deepcopy(counterEvents['PEAK_ALLOC'][-1])
#     ev_dummy["ts"] += 1 # us
#     ev_dummy["args"]["peak allocated memory"] = 100.
#     counterEvents['PEAK_ALLOC'].append(ev_dummy)
#     ev_dummy = copy.deepcopy(counterEvents['PEAK_RESRV'][-1])
#     ev_dummy["ts"] += 1 # us
#     ev_dummy["args"]["peak reserved memory"] = 100.
#     counterEvents['PEAK_RESRV'].append(ev_dummy)
#     # counterEvents['PEAK_ALLOC'][-1]["args"]["peak allocated memory"] = 100.
#     # counterEvents['PEAK_RESRV'][-1]["args"]["peak reserved memory"] = 100.
