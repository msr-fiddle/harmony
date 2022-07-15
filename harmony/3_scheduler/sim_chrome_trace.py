# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os
from collections import OrderedDict as ODict
import gzip

import sys; sys.path.append("../4_runtime/viewer")
from colors import COLORS

from sim_data_struct import Event

STREAM_ORDER = ["SwapIn", "P2PIn", "Compute", "P2POut", "SwapOut", "CPU"]

def munge_time(t):
    """Take a time from simulator (sec) and convert it into a chrome://tracing's event time (ts & dur is micro-sec) """
    """Also, 'displayTimeUnit' is a string that specifies in which unit timestamps should be displayed. This supports values of “ms” or “ns”. By default this is value is “ms”. """
    return t * 1E6

def ev2cname(ev):
    if ev.stream.startswith("Swap"):
        return COLORS[31][0]
    elif ev.stream.startswith("P2P"):
        return COLORS[0][0]
    elif ev.kind.startswith("FWD"):
        return COLORS[2][0]
    elif ev.kind.startswith("REC"):
        return COLORS[32][0]
    elif ev.kind.startswith("BWD"):
        return COLORS[22][0]
    elif ev.kind.startswith("ARD"):
        return COLORS[0][0]
    elif ev.kind.startswith("DEL"):
        return COLORS[21][0]
    elif ev.kind.startswith("MSG"):
        return COLORS[6][0]
    elif ev.kind.startswith("Update"):
        return COLORS[5][0]
    else:
        raise ValueError

def save_to_chrome_trace(args, configs, events):
    """ top-level function """ 
    # print("Converting events to chrome trace ...")
    ### Convert
    rank_stream_traces = ODict() # { rank: { stream: [trace, trace] } 
    for id, ev in events.items():
        rank = ev.vt.rank
        if ev.stream.startswith("Swap") and args.separate_swap:
            stream = ev.stream + ev.kind
        else:
            stream = ev.stream
        #
        inputs = ODict() # { inputs[0]: event id }
        for i, inev in enumerate(ev.inputs):
            inputs["inputs[%d]"%i] = inev.id
        #
        trace = {
                    "ph": "X", # Complete Event
                    "name": ev.name,
                    "cat": ev.id,
                    "ts": munge_time(ev.begin),
                    "dur": munge_time(ev.dur),
                    "args": inputs,
                    "tid": stream,
                    "pid": ": Rank%d"%rank,
                    "cname": ev2cname(ev),
                }
        #
        if rank not in rank_stream_traces:
            rank_stream_traces[rank] = ODict()
        if stream not in rank_stream_traces[rank]:
            rank_stream_traces[rank][stream] = []
        rank_stream_traces[rank][stream].append(trace)
    ### Arrange display order
    global_traces = []
    for r in sorted(rank_stream_traces.keys()):
        streams = list(rank_stream_traces[r].keys())
        for o in STREAM_ORDER:
            for s in list(filter(lambda s: s.startswith(o), streams)):
                global_traces += rank_stream_traces[r][s]
    ### Write to a single json file
    assert os.path.exists(args.dir_jsons)
    path_json = os.path.join(args.dir_jsons, args.json)
    if '.gz' in args.json:
        with gzip.open(path_json, 'w') as f: # gzipped json
            f.write(json.dumps(global_traces).encode('utf-8'))
        # why need to compress?
        # > google trace-viewer has 256MB file limit: https://github.com/google/trace-viewer/issues/627
        # > google trace-viewer has Large File Silent Failure: https://github.com/google/trace-viewer/issues/298
        # If compressed too small (<3KB), trace-viewr doesn't accept it either.
        if os.stat(path_json).st_size < 5000:
            path_json = path_json.replace(".gz", "")
            with open(path_json, 'w') as f: # pure json
                json.dump(global_traces, f)
    else:
        with open(path_json, 'w') as f: # pure json
            json.dump(global_traces, f)
    print("chrome-trace.json dumped to %s (%.3f MB)"%(path_json, float(os.stat(path_json).st_size/1024./1024.)))
