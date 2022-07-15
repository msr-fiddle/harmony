# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .vlayer0 import vLayer0
from .vlayer1 import vLayer1
from .vlayer2 import vLayer2
from .vlayer3 import vLayer3
from .vlayer4 import vLayer4
from .vlayer5 import vLayer5
from .vlayer6 import vLayer6
from .vlayer7 import vLayer7
from .vlayer8 import vLayer8
from .vlayer9 import vLayer9
from .vlayer10 import vLayer10
from .vlayer11 import vLayer11
from .vlayer12 import vLayer12
from .vlayer13 import vLayer13
from .vlayer14 import vLayer14
from .vlayer15 import vLayer15
from .vlayer16 import vLayer16
from .vlayer17 import vLayer17
from .vlayer18 import vLayer18
from .vlayer19 import vLayer19
from .vlayer20 import vLayer20
from .vlayer21 import vLayer21
from .vlayer22 import vLayer22
from .vlayer23 import vLayer23
from .vlayer24 import vLayer24
from .vlayer25 import vLayer25
from .vlayer26 import vLayer26

def arch():
    return "bert_large"

def model(config, criterion):
    vlayer0 = vLayer0(config)
    vlayer1 = vLayer1(config)
    vlayer2 = vLayer2(config)
    vlayer3 = vLayer3(config)
    vlayer4 = vLayer4(config)
    vlayer5 = vLayer5(config)
    vlayer6 = vLayer6(config)
    vlayer7 = vLayer7(config)
    vlayer8 = vLayer8(config)
    vlayer9 = vLayer9(config)
    vlayer10 = vLayer10(config)
    vlayer11 = vLayer11(config)
    vlayer12 = vLayer12(config)
    vlayer13 = vLayer13(config)
    vlayer14 = vLayer14(config)
    vlayer15 = vLayer15(config)
    vlayer16 = vLayer16(config)
    vlayer17 = vLayer17(config)
    vlayer18 = vLayer18(config)
    vlayer19 = vLayer19(config)
    vlayer20 = vLayer20(config)
    vlayer21 = vLayer21(config)
    vlayer22 = vLayer22(config)
    vlayer23 = vLayer23(config)
    vlayer24 = vLayer24(config)
    vlayer25 = vLayer25(config)
    vlayer26 = vLayer26(config)
    # (vlayer_id, sorted [in_node_id], sorted [out_node_id])
    return [
        (vlayer0, ["input0", "input1", "input2"], ["out0", "out1"]),
        (vlayer1, ["out0", "out1"], ["out2", "out3"]),
        (vlayer2, ["out2", "out3"], ["out4", "out5"]),
        (vlayer3, ["out4", "out5"], ["out6", "out7"]),
        (vlayer4, ["out6", "out7"], ["out8", "out9"]),
        (vlayer5, ["out8", "out9"], ["out10", "out11"]),
        (vlayer6, ["out10", "out11"], ["out12", "out13"]),
        (vlayer7, ["out12", "out13"], ["out14", "out15"]),
        (vlayer8, ["out14", "out15"], ["out16", "out17"]),
        (vlayer9, ["out16", "out17"], ["out18", "out19"]),
        (vlayer10, ["out18", "out19"], ["out20", "out21"]),
        (vlayer11, ["out20", "out21"], ["out22", "out23"]),
        (vlayer12, ["out22", "out23"], ["out24", "out25"]),
        (vlayer13, ["out24", "out25"], ["out26", "out27"]),
        (vlayer14, ["out26", "out27"], ["out28", "out29"]),
        (vlayer15, ["out28", "out29"], ["out30", "out31"]),
        (vlayer16, ["out30", "out31"], ["out32", "out33"]),
        (vlayer17, ["out32", "out33"], ["out34", "out35"]),
        (vlayer18, ["out34", "out35"], ["out36", "out37"]),
        (vlayer19, ["out36", "out37"], ["out38", "out39"]),
        (vlayer20, ["out38", "out39"], ["out40", "out41"]),
        (vlayer21, ["out40", "out41"], ["out42", "out43"]),
        (vlayer22, ["out42", "out43"], ["out44", "out45"]),
        (vlayer23, ["out44", "out45"], ["out46", "out47"]),
        (vlayer24, ["out46", "out47"], ["out48"]),
        (vlayer25, ["out48"], ["out49"]),
        (vlayer26, ["out49"], ["out50"]),
        (criterion, ["out50"], ["loss"])
    ]
