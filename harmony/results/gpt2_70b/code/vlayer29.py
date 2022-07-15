# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from gpt2_huggingface.modeling2_gpt2 import GPT2Layer
from gpt2_huggingface.modeling_utils import Conv1D

class vLayer29(torch.nn.Module):
    def __init__(self, config):
        super(vLayer29, self).__init__()
        self.layer1 = GPT2Layer(config=config)
        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (torch.nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input0): # predecessor (ordered by in_node_id)
        # 1 out = 1 layer_call(in1,in2,...) = 1 node
        out0 = input0
        out1 = self.layer1(out0)
        return out1 # successor (ordered by out_node_id)
