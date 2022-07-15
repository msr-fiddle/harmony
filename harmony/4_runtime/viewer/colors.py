# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict as ODict
COLORS = ODict({
    0: ["thread_state_uninterruptible", (182, 125, 143)],
    1: ["thread_state_iowait", (255, 140, 0)],
    2: ["thread_state_running", (126, 200, 148)],
    3: ["thread_state_runnable", (133, 160, 210)],
    4: ["thread_state_sleeping", (240, 240, 240)],
    5: ["thread_state_unknown", (199, 155, 125)],
    6: ["background_memory_dump", (0, 180, 180)],
    7: ["light_memory_dump", (0, 0, 180)],
    8: ["detailed_memory_dump", (180, 0, 180)],
    9: ["vsync_highlight_color", (0, 0, 255)],
    10: ["generic_work", (125, 125, 125)],
    11: ["good", (0, 125, 0)],
    12: ["bad", (180, 125, 0)],
    13: ["terrible", (180, 0, 0)],
    14: ["black", (0, 0, 0)],
    15: ["grey", (221, 221, 221)],
    16: ["white", (255, 255, 255)],
    17: ["yellow", (255, 255, 0)],
    18: ["olive", (100, 100, 0)],
    19: ["rail_response", (67, 135, 253)],
    20: ["rail_animation", (244, 74, 63)],
    21: ["rail_idle", (238, 142, 0)],
    22: ["rail_load", (13, 168, 97)],
    23: ["startup", (230, 230, 0)],
    24: ["heap_dump_stack_frame", (128, 128, 128)],
    25: ["heap_dump_object_type", (0, 0, 255)],
    26: ["heap_dump_child_node_arrow", (204, 102, 0)],
    27: ["cq_build_running", (255, 255, 119)],
    28: ["cq_build_passed", (153, 238, 102)],
    29: ["cq_build_failed", (238, 136, 136)],
    30: ["cq_build_abandoned", (187, 187, 187)],
    31: ["cq_build_attempt_runnig", (222, 222, 75)],
    32: ["cq_build_attempt_passed", (103, 218, 35)],
    33: ["cq_build_attempt_failed", (197, 81, 81)]
})

"""
##### NOTE #####
Profiling Overhead: 13

CPU Utilization:7
CPU Probe:7

Runtime API: 29 or 33 
Marker & Range: 28 or 11
Background: 6

SwapIn/SwapOut: 18
Compute: 2
P2PIn/Out_v: 0
P2PIn/Out_^: 3

Peak Allocated: 1
Peak Reserved: 5
Probe: 12
"""

CNT_RUNTIME = 0
CNT_MARKER = 0

def tid2cname(tid):
    assert isinstance(tid, str)
    if "Profiling" in tid:
        return COLORS[13][0]
    elif "CPU Util" in tid:
        return COLORS[7][0]    
    elif "CPU Probe" in tid:
        return COLORS[7][0]  
    elif ("CUDA API" in tid) or ("Runtime" in tid):
        global CNT_RUNTIME; CNT_RUNTIME += 1
        return COLORS[29][0] if CNT_RUNTIME%2 == 0 else COLORS[33][0]
    elif ("Main" in tid) or ("Marker" in tid):
        global CNT_MARKER; CNT_MARKER += 1
        return COLORS[28][0] if CNT_MARKER%2 == 0 else COLORS[11][0]
    elif "Background" in tid:
        return COLORS[6][0]
    elif "Swap" in tid:
        return COLORS[31][0]
    elif "Compute" in tid:
        return COLORS[2][0]
    elif "P2P" in tid and "v" in tid:
        return COLORS[0][0]
    elif "P2P" in tid and "^" in tid:
        return COLORS[3][0]
    elif "P2P" in tid:
        return COLORS[0][0]
    elif "Allocated" in tid:
        return COLORS[1][0]
    elif "Reserved" in tid:
        return COLORS[12][0]
    elif "GPU Mem Probe" in tid:
        return COLORS[5][0]
    else:
        assert False

# def name2cname(name):
#     if name.startswith("cudaFree"):
#         return COLORS[8][0]

##########################
# if __name__ == "__main__": # TEST
#     import numpy as np
#     import matplotlib.pyplot as plt
#     # create a dataset
#     y_height = [1]*len(COLORS)
#     x_pos = np.arange(len(COLORS))
#     x_color = [ tuple(np.array(rgb, dtype=np.float64)/255.0) 
#                 for name,rgb in COLORS.values()] 
#     x_ticks = [ id for id in COLORS.keys() ] 
#     # Create bars with different colors
#     plt.figure(figsize=(11,1))
#     plt.bar(x_pos, y_height, color=x_color)
#     # Create names on the x-axis
#     plt.xticks(x_pos, tuple(x_ticks))
#     # Show graph
#     plt.savefig("colors.pdf", bbox_inches='tight')
#     plt.close()
#     print("plot written")
