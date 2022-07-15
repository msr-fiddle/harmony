# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from bert_thomwolf.modeling2 import BertLayerNorm

class vLayer26(torch.nn.Module):
    def __init__(self, config):
        super(vLayer26, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=1024, out_features=2, bias=True)
        self.config = config
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses 
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617.
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input0): # predecessor (ordered by in_node_id)
        # 1 out = 1 layer_call(in1,in2,...) = 1 node
        out0 = input0
        out1 = self.layer1(out0)
        return out1 # successor (ordered by out_node_id)
