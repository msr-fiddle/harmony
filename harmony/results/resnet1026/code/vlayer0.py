# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from vgg_resnet_torch.resnet1026 import ConvLayer

class vLayer0(torch.nn.Module):
    def __init__(self):
        super(vLayer0, self).__init__()
        self.layer2 = ConvLayer(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, input0): # predecessor (ordered by in_node_id)
        # 1 out = 1 layer_call(in1,in2,...) = 1 node
        out0 = input0
        out2 = self.layer2(out0)
        return out2 # successor (ordered by out_node_id)
