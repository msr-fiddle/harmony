# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from vgg_resnet_torch.vgg416 import ConvLayer

class vLayer159(torch.nn.Module):
    def __init__(self):
        super(vLayer159, self).__init__()
        self.layer1 = ConvLayer(in_channels=128, out_channels=128, batch_norm=False, inplace=True, max_pool=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, input0): # predecessor (ordered by in_node_id)
        # 1 out = 1 layer_call(in1,in2,...) = 1 node
        out0 = input0
        out1 = self.layer1(out0)
        return out1 # successor (ordered by out_node_id)
