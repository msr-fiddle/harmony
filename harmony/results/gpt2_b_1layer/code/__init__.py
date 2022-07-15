# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .vlayer0 import vLayer0
from .vlayer1 import vLayer1
from .vlayer2 import vLayer2
from .vlayer3 import vLayer3

def arch():
    return "gpt2_b_1layer"

def model(config, criterion):
    vlayer0 = vLayer0(config)
    vlayer1 = vLayer1(config)
    vlayer2 = vLayer2(config)
    vlayer3 = vLayer3(config)
    # (vlayer_id, sorted [in_node_id], sorted [out_node_id])
    return [
        (vlayer0, ["input0"], ["out0"]),
        (vlayer1, ["out0"], ["out1"]),
        (vlayer2, ["out1"], ["out2"]),
        (vlayer3, ["out2"], ["out3"]),
        (criterion, ["out3"], ["loss"])
    ]
