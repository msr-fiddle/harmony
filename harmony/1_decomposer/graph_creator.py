# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import functools
import numpy as np
import os
import torch
from torch.autograd import Variable
from torch.autograd import Function

import graph

import inspect
def description(obj):
    """ Get one-line description of an Torch Module object.
        Format: <ClassName>(Arg1=Val1, Arg2=Val2) 
        E.g.  : Linear(in_features=512, out_features=1000, bias=True)
    """
    assert isinstance(obj, torch.nn.Module)
    if obj.extra_repr(): # most torch Modules
        return obj.__repr__()
    else: # customized Modules
        sig = inspect.signature(obj.__class__.__init__)
        names = [param.name for param in sig.parameters.values()]
        if "self" in names:
            names.remove("self")
        values = []
        for n in names: # in strict definition order
            if n in ['config']: # BERT, GPT2
                values.append(n)
            else:
                assert hasattr(obj, n)
                v = getattr(obj, n)
                if isinstance(v, (int, float, bool, None)):
                    values.append(v)
                else:
                    print("[warning] untested argument type: {}.__init__({}={})".format(obj.__class__.__name__, n, v))
                    values.append(v)
        main_str = obj._get_name() + '('
        # make simple one-liner info as most builtin Modules will use
        # e.g.
        #       'in_features={}, out_features={}, bias={}'.format(
        #        self.in_features, self.out_features, self.bias is not None )
        main_str += ", ".join(["{}={}".format(n,v) for n,v in zip(names,values)])
        main_str += ')'
        return main_str
    # ref: 
    # - https://pytorch.org/docs/1.5.0/_modules/torch/nn/modules/module.html#Module
    # - https://pytorch.org/docs/1.5.0/_modules/torch/nn/modules/linear.html#Linear
    # - https://pytorch.org/docs/1.5.0/_modules/torch/nn/modules/conv.html#Conv2d
    # - https://docs.python.org/3.8/library/inspect.html#introspecting-callables-with-the-signature-object
    # - https://stackoverflow.com/questions/36849837/introspecting-arguments-from-the-constructor-function-init-in-python

object_id = 0

class TensorWrapper(object):
    def __init__(self, tensor, node_desc, graph_creator, activation_size=None):
        self.tensor = tensor
        global object_id
        self.object_id = object_id
        object_id += 1
        self.node_desc = node_desc

        self._node = graph.Node("node%d" % object_id, node_desc=node_desc)
        self.graph_creator = graph_creator

    def size(self, dim=None):
        if dim is None:
            result = self.tensor.size()
            dim_str = ""
        else:
            result = self.tensor.size(dim)
            dim_str = "(%d)" % dim
        wrapped_result = TensorWrapper(result, "Size%s" % dim_str, self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        return wrapped_result

    def dim(self):
        return self.tensor.dim()

    def view(self, *wrapped_sizes):
        sizes = []
        in_edges = []
        for wrapped_size in wrapped_sizes:
            if isinstance(wrapped_size, TensorWrapper):
                sizes.append(wrapped_size.tensor)
                in_edges.append(wrapped_size)
            else:
                sizes.append(wrapped_size)
        result = self.tensor.view(*sizes)

        if len(sizes) == 1:
            wrapped_result = TensorWrapper(result, "View", self.graph_creator)
        else:
            wrapped_result = TensorWrapper(result,
                                           "View(%s)" % ", ".join([str(size) for size in sizes[1:]]),
                                           self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        for in_edge in in_edges:
            self.graph_creator.graph.add_edge(in_edge.node(), wrapped_result.node())
        return wrapped_result

    def __gt__(self, other):
        return self.tensor.__gt__(other)

    def __lt__(self, other):
        return self.tensor.__lt__(other)

    def __add__(self, other):
        
        if isinstance(other, TensorWrapper):
            result_tensor = self.tensor + other.tensor
        else:
            result_tensor = self.tensor + other
        wrapped_result = TensorWrapper(result_tensor, "Add", self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        if isinstance(other, TensorWrapper):
            self.graph_creator.graph.add_edge(other.node(), wrapped_result.node())
        return wrapped_result

    def __iadd__(self, other):
        wrapped_result = TensorWrapper(self.tensor, "Add(inplace)", self.graph_creator)
        self.tensor += other.tensor
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        self.graph_creator.graph.add_edge(other.node(), wrapped_result.node())
        return wrapped_result

    def __mul__(self, other):
        result = self.tensor * other.tensor
        wrapped_result = TensorWrapper(result, "Mul", self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        self.graph_creator.graph.add_edge(other.node(), wrapped_result.node())
        return wrapped_result

    def __getitem__(self, key):
        """ NOTE: slice is underdevelopment 
        see: 
            https://www.geeksforgeeks.org/implementing-slicing-in-__getitem__/
            https://stackoverflow.com/questions/43627405/understanding-getitem-method
            https://stackoverflow.com/questions/2936863/implementing-slicing-in-getitem
        """
        result_tensor = self.tensor[key]
        wrapped_result = TensorWrapper(result_tensor, "__getitem__(%s)" % key, self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        return wrapped_result

    def transpose(self, *args):
        result_tensor = self.tensor.transpose(*args)
        args_str = ", ".join([str(arg) for arg in args])
        wrapped_result = TensorWrapper(result_tensor, "Transpose(%s)" % args_str,
                                       self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        return wrapped_result

    def unsqueeze(self, *args):
        return self.tensor.unsqueeze(*args)

    def node(self):
        return self._node


def cat(wrapped_tensors, dim):
    tensors = []
    all_unwrapped_tensors = True
    graph_creator = None
    for wrapped_tensor in wrapped_tensors:
        if isinstance(wrapped_tensor, TensorWrapper):
            tensors.append(wrapped_tensor.tensor)
            graph_creator = wrapped_tensor.graph_creator
            all_unwrapped_tensors = False
        else:
            tensors.append(wrapped_tensor)
    # Simplifying assumption: if all tensors are "unwrapped", then we're not profiling,
    # and default to torch implementation.
    if all_unwrapped_tensors:
        return torch.cat(tensors, dim)
    result = torch.cat(tensors, dim)
    wrapped_result = TensorWrapper(result, "Concat(%d)" % dim, graph_creator)
    for wrapped_tensor in wrapped_tensors:
        if not isinstance(wrapped_tensor, TensorWrapper):
            wrapped_tensor = TensorWrapper(wrapped_tensor, "Input", graph_creator)
        graph_creator.graph.add_edge(wrapped_tensor.node(), wrapped_result.node())
    return wrapped_result


class GraphCreator(object):
    def __init__(self, model, module_whitelist=[], summary=[]):
        """
        Recursively create graph nodes from top to leaf module.
        Args:
            model: the top-level module
            module_whitelist: optional, contains module class names, if recursion hits this whitelist, just build the hit module as a graph node, without further recursion to leaf modules. Otherwise (miss this whitelist), recursion to leafs. 
            summary: optional, contains size and time of each module node
        """
        if isinstance(model, torch.nn.Module) is False:
            raise Exception("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.module_whitelist = module_whitelist
        self.summary = copy.deepcopy(summary)
        self.forward_original_methods = {}
        self.graph = graph.Graph()
        self.inputs = {}

    def hook_modules(self, module, root=False):
        this_creator = self
        sub_modules = module.__dict__['_modules']

        # Wrapper function to "forward()", keeping track of dependencies.
        def forward_wrapper(self, *wrapped_inputs):
            input = []
            wrapped_inputs_list = list(wrapped_inputs)
            for i in range(len(wrapped_inputs_list)):
                if isinstance(wrapped_inputs_list[i], TensorWrapper):
                    input.append(wrapped_inputs_list[i].tensor)
                else:
                    key = wrapped_inputs_list[i]
                    if key in this_creator.inputs:
                        wrapped_inputs_list[i] = this_creator.inputs[key]
                    else:
                        j = len(this_creator.inputs)
                        wrapped_inputs_list[i] = TensorWrapper(wrapped_inputs_list[i],
                                                               "Input%d" % j, this_creator)
                        this_creator.inputs[key] = wrapped_inputs_list[i]
                    input.append(wrapped_inputs_list[i].tensor)
            result = this_creator.forward_original_methods[self](*input)
            
            wrapped_result = TensorWrapper(result, description(self), this_creator) 
            
            for wrapped_input in wrapped_inputs_list:
                this_creator.graph.add_edge(wrapped_input.node(), wrapped_result.node())

            return wrapped_result

        # Wrapper function to "forward()", keeping track of dependencies.
        # (without creating self node)
        def forward_wrapper_root(self, *wrapped_inputs):
            input = []
            wrapped_inputs_list = list(wrapped_inputs)
            for i in range(len(wrapped_inputs_list)):
                if isinstance(wrapped_inputs_list[i], TensorWrapper):
                    input.append(wrapped_inputs_list[i].tensor)
                else:
                    key = wrapped_inputs_list[i]
                    if key in this_creator.inputs:
                        wrapped_inputs_list[i] = this_creator.inputs[key]
                    else:
                        j = len(this_creator.inputs)
                        wrapped_inputs_list[i] = TensorWrapper(wrapped_inputs_list[i],
                                                               "Input%d" % j, this_creator)
                        this_creator.inputs[key] = wrapped_inputs_list[i]
                    input.append(wrapped_inputs_list[i].tensor)
            result = this_creator.forward_original_methods[self](*input)

            return result

        for name, sub_module in sub_modules.items():
            # nn.Module is the only thing we care about.
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break

            sub_module_name = sub_module.__class__.__name__
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) == 0 or sub_module_name in self.module_whitelist:
                """
                In 'pre_hook.patch' for the 'torchprofiler'
                +    def reset_hooks(self):
                +        self._backward_hooks = OrderedDict()
                +        self._backward_pre_hooks = OrderedDict() # profiler specific
                +        self._forward_hooks = OrderedDict()
                +        self._backward_hooks = OrderedDict()
                """
                """
                To be patch free, we manually reset hooks:
                    sub_module._backward_hooks = OrderedDict()
                    sub_module._forward_hooks = OrderedDict()
                [optional] sub_module._forward_pre_hooks = OrderedDict()
                """
                # sub_module.reset_hooks()
                if hasattr(sub_module, 'reset_hooks'): # patched for 'torchprofiler'
                    sub_module.reset_hooks()
                else: # patch free
                    from collections import OrderedDict
                    sub_module._backward_hooks = OrderedDict()
                    sub_module._forward_hooks = OrderedDict()    
                #
                # Hook leaf nn.Module with no descendants, or just in whitelist.
                #

                # Replace "forward" with "wrapped_forward".
                if sub_module not in this_creator.forward_original_methods:
                    this_creator.forward_original_methods.update({sub_module:
                                                                   sub_module.forward})
                    sub_module.forward = forward_wrapper.__get__(sub_module, sub_module.__class__)

            if len(sub_sub_modules) > 0 and sub_module_name not in self.module_whitelist:
                #
                # Recursively visit this module's descendants, if not in whitelist
                #
                self.hook_modules(sub_module)
        
        if root:
            this_creator.forward_original_methods.update({module: module.forward})
            module.forward = forward_wrapper_root.__get__(module, module.__class__)

    def unhook_modules(self):
        for sub_module in self.forward_original_methods:
            sub_module.forward = self.forward_original_methods[sub_module]

    def persist_graph(self, directory): 
        graph.save_graph(self.graph, directory, graph.GRAPH_FILENAME)
