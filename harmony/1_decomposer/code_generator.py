# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import re
import subprocess
import sys

import graph
from graph_sequentializer import partition, sequentialize

DECLARATION_SKIPLIST = [
    "hidden",
    "__getitem__",
    "Add",
    "Mul",
    "Concat",
    "Identity",
    "Input",
    "Size",
    "View",
    "Transpose",
    "self.get_seq_lens"
]

DECLARATION_SPECIALCASE = [
    "EmuBidirLSTM",
    "RecurrentAttention",
    "Classifier",
    "MaskConv",
    "ResizeInput",
    "InferenceBatchSoftmax",
    "BatchRNN",
    "SequenceWise",
    "BertEmbeddings",
    "BertLayer",
    "BertPooler",
    "GPT2Embeddings",
    "GPT2Layer",
    "GPT2LayerNorm",
    "GPT2LMHead",
    "ConvLayer",
    "LinearLayer0",
    "LinearLayer1",
    "LinearLayer2",
    # "ConvLayer",
    "BasicBlock",
    "Bottleneck",
    "LinearLayer"
]

CONFIG = "config"
MODEL_TEMPLATE_FILE = "templates/model.py.template"
INIT_TEMPLATE_FILE = "templates/__init__.py.template"

def _assumption():
    # DECLARATION_SKIPLIST does not overlap with DECLARATION_SPECIALCASE
    # i.e., declaration cannot resides in both lists, but at most in 1 list.
    for dec in DECLARATION_SKIPLIST:
        assert dec not in DECLARATION_SPECIALCASE
    for dec in DECLARATION_SPECIALCASE:
        assert dec not in DECLARATION_SKIPLIST

def get_output_tuple_str(outputs):
    if len(outputs) == 1:
        return outputs[0]
    return "(%s)" % ", ".join(outputs)

def get_tensor_names_list(names): # input_names/output_names ={ in/out_node_id: input%d }
    return [names[node_id] for node_id in sorted(names.keys())]

def get_input_names(subgraph, full_graph, check_vlayers=True):
    ''' per-vlayer input (ordering doesn't matter), return input_names = { in_node_id: input%d } '''
    # Figure out the inputs to this sub-graph, which are the predecessors of
    # nodes in the sub-graph not in the sub-graph.
    # input_names is a dict mapping each predecessor's node_id to assigned
    # variable name. 
    nodes = subgraph.nodes
    input_names = {}
    counter = 0
    for node_id in nodes:
        if (node_id in full_graph.in_edges and len(full_graph.in_edges[node_id]) > 0):
            for in_node in full_graph.in_edges[node_id]:
                if in_node.vlayer_id != nodes[node_id].vlayer_id and check_vlayers:
                    # Skip hidden inputs.
                    if full_graph.nodes[in_node.node_id].node_desc.startswith("hidden"):
                        continue
                    input_names[in_node.node_id] = "input%d" % counter
                    counter += 1
        else:
            if subgraph.nodes[node_id].node_desc.startswith("Input"):
                input_names[node_id] = "input%d" % counter
                counter += 1
    return input_names

def get_output_names(subgraph, full_graph, counter):
    ''' per-vlayer output (ordering doesn't matter), return output_names = { out_node_id: out%d } '''
    # Figure out the outputs of this sub-graph, which are the nodes in the
    # sub-graph with edges out of the sub-graph.
    nodes = subgraph.nodes
    output_names = {}
    for node_id in nodes:
        if (node_id in full_graph.edges and len(full_graph.edges[node_id]) > 0):
            for out_node in full_graph.edges[node_id]:
                if out_node.vlayer_id != nodes[node_id].vlayer_id:
                    if full_graph.nodes[node_id].node_desc.startswith("hidden"):
                        continue
                    output_names[node_id] = "out%d" % counter
                    counter += 1
        else:
            output_names[node_id] = "out%d" % counter
            counter += 1
    return output_names, counter

def convert_subgraph_to_module(subgraph, full_graph, module_name, initialize_weights, arch, output_filename, verbose=False):
    """ convert a subgraph (vlayer) to a module file """
    import_statements = []
    extra_args = []
    layer_declarations = []
    module_methods = []
    function_definition = []

    nodes = subgraph.topological_sort() # a node is a self.layer
    counter = 0 # for out%d
    layer_names = {} # { node_id: "self.layer%d" }
    input_names = get_input_names(subgraph, full_graph) # { in_node_id: input%d }
    output_names = input_names.copy() # { node_id: input%d/out%d }
    
    # First, count the inputs to forward() before count each layer
    for node_id in input_names: # { in_node_id: "input0" } unordered
        output_name = "out%d" % counter
        function_definition.append("%s = %s" % (output_name, input_names[node_id]))
        output_names[node_id] = output_name # { in_node_id: "out0" }
        counter += 1
    
    # Now, generate expressions for each node.
    # Iterate through nodes in topological order, and add output_name mappings for
    # each expression. Use this output_name mapping when generating expressions
    # in the model's implementation file.
    # PipeDream todo: Make sure that nodes with multiple inputs have the inputs in the
    # right order (even though this probably does not matter in practice).
    for node in nodes:
        #---- Generate layer_declarations ----
        layer_name = "self.layer%d" % counter
        layer_names[node.node_id] = layer_name
        layer_declaration = "torch.nn.%s" % node.node_desc
        # NOTE: always make CONFIG extra_args for transformers
        #       always make CONFIG 1st extra_args
        if arch.startswith("bert") or arch.startswith("gpt2"):
            if CONFIG not in extra_args: 
                extra_args.insert(0, CONFIG)
        # Skip layers that don't need a declaration (example: '+=').
        skip = False
        for declaration in DECLARATION_SKIPLIST:
            if node.node_desc.startswith(declaration):
               skip = True
               break
        # Check special case for overwriting layer_declaration, add import_statements, add module_methods
        if not skip:
            declaration = node.node_desc.split("(")[0]
            if declaration in DECLARATION_SPECIALCASE: # exact match
                if declaration == "EmuBidirLSTM":
                    m = re.search(r'.*LSTM\((\d+), (\d+)\).*', node.node_desc)
                    input_size = int(m.group(1))
                    hidden_size = int(m.group(2))
                    layer_declaration = "EmuBidirLSTM(%d, %d)" % (input_size, hidden_size)
                    import_statements.append("from seq2seq.models.encoder import EmuBidirLSTM")
                elif declaration == "RecurrentAttention":
                    m = re.search(r'.*LSTM\((\d+), (\d+)\).*', node.node_desc)
                    input_size = int(m.group(1))
                    hidden_size = int(m.group(2))
                    m = re.search(r'.*in_features=(\d+), out_features=(\d+).*', node.node_desc)
                    context_size = int(m.group(1))
                    layer_declaration = "RecurrentAttention(%d, %d, %d)" % (input_size, hidden_size, context_size)
                    import_statements.append("from seq2seq.models.decoder import RecurrentAttention")
                elif declaration == "Classifier":
                    m = re.search(r'.*in_features=(\d+), out_features=(\d+).*', node.node_desc)
                    in_features = int(m.group(1))
                    out_features = int(m.group(2))
                    layer_declaration = "Classifier(%d, %d)" % (in_features, out_features)
                    import_statements.append("from seq2seq.models.decoder import Classifier")
                elif declaration == "MaskConv":
                    node_desc = node.node_desc
                    modules = node_desc.split("    ")[1:-1]
                    module_declarations = []
                    for module in modules:
                        module_declaration = "torch.nn." + module.split(": ")[1].replace("inplace", "inplace=False")
                        module_declarations.append(module_declaration)
                    layer_declaration = "MaskConv(torch.nn.Sequential(%s))" % ",\n            ".join(module_declarations)
                    import_statements.append("from model import MaskConv")
                    module_methods.append("""def get_seq_lens(self, input_length):
        seq_len = input_length
        for m in %s.modules():
            if type(m) == torch.nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()""" % layer_name)
                elif declaration == "BatchRNN":
                    if "batch_norm" in node.node_desc:
                        batch_norm = True
                    else:
                        batch_norm = False
                    if "LSTM" in node.node_desc:
                        rnn_type = "torch.nn.LSTM"
                        m = re.search(r'LSTM\((\d+), (\d+), bidirectional=([a-zA-Z]+)\)', node.node_desc)
                        input_size = int(m.group(1))
                        hidden_size = int(m.group(2))
                        bidirectional = m.group(3)
                    elif "GRU" in node.node_desc:
                        rnn_type = "torch.nn.GRU"
                        m = re.search(r'GRU\((\d+), (\d+), bidirectional=([a-zA-Z]+)\)', node.node_desc)
                        input_size = int(m.group(1))
                        hidden_size = int(m.group(2))
                        bidirectional = m.group(3)
                    else:
                        # PipeDream todo: Do something else?
                        pass
                    # PipeDream todo: Pass remaining arguments.
                    # PipeDream todo: Get hidden and input size.
                    layer_declaration = "BatchRNN(%d, %d, rnn_type=%s, batch_norm=%s, bidirectional=%s)" % (
                        input_size, hidden_size, rnn_type, batch_norm, bidirectional)
                    import_statements.append("from model import BatchRNN")
                elif declaration == "ResizeInput":
                    layer_declaration = "ResizeInput()"
                    import_statements.append("from model import ResizeInput") 
                elif declaration == "SequenceWise":
                    node_desc = node.node_desc
                    modules = node_desc[:-2].split("  ")[1:]
                    module_declarations = []
                    for module in modules:
                        module_declaration = "torch.nn." + module.split(": ")[1].replace("inplace", "inplace=False")
                        module_declarations.append(module_declaration)
                    layer_declaration = "SequenceWise(torch.nn.Sequential(%s))" % ",\n            ".join(module_declarations)
                    import_statements.append("from model import SequenceWise")
                elif declaration == "InferenceBatchSoftmax":
                    layer_declaration = "InferenceBatchSoftmax()"
                    import_statements.append("from model import InferenceBatchSoftmax")
                elif declaration.startswith("Bert"):
                    layer_declaration = "%s" % (node.node_desc)
                    import_statements.append("from bert_thomwolf.modeling2 import %s" % declaration)
                elif declaration.startswith("GPT2"):
                    layer_declaration = "%s" % (node.node_desc)
                    import_statements.append("from gpt2_huggingface.modeling2_gpt2 import %s" % declaration)
                elif declaration in ["LinearLayer0","LinearLayer1","LinearLayer2"]:
                    layer_declaration = "%s" % (node.node_desc)
                    import_statements.append("from vgg_resnet_torch.vgg416 import %s" % declaration)
                elif declaration in ["BasicBlock","Bottleneck","LinearLayer"]:
                    layer_declaration = "%s" % (node.node_desc)
                    import_statements.append("from vgg_resnet_torch.resnet1026 import %s" % declaration)
                elif declaration == "ConvLayer":
                    layer_declaration = "%s" % (node.node_desc)
                    if arch.startswith("vgg"):
                        import_statements.append("from vgg_resnet_torch.vgg416 import %s" % declaration)
                    elif arch.startswith("resnet"):
                        import_statements.append("from vgg_resnet_torch.resnet1026 import %s" % declaration)
                    else:
                        raise ValueError
                else:
                    raise ValueError("Shouldn't be here")

        import_statements = list(set(import_statements)) # dedup imports

        if not skip: # common case or special case
            layer_declarations.append("%s = %s" % (layer_name, layer_declaration))

        #---- Generate function_definition ----
        layer_call = None
        output_name = "out%d" % counter
        if node.node_id not in output_names:
            output_names[node.node_id] = output_name # { node_id: "out1" }
        
        if node.node_id in full_graph.in_edges:
            in_edges = full_graph.in_edges[node.node_id] # ordered
        else:
            in_edges = []
        if len(in_edges) == 0 and node.node_desc.startswith("Input"):
            pass  # Don't need to do anything for this case.
        else:
            if node.node_desc.startswith("Size"):
                assert(len(in_edges) == 1)
                m = re.search(r'Size\((-?\d+)\)', node.node_desc)
                idx = int(m.group(1))
                layer_call = "%s = %s.size(%d)" % (output_name,
                                                   output_names[in_edges[0].node_id],
                                                   idx)
            elif node.node_desc.startswith("View"):
                size_node_ids = []
                input_node_id = None
                for in_node in in_edges: # ordered
                    src_node_id = full_graph.track_source_of_new_node_chain(in_node.node_id) # identity chain in sequential graph
                    if full_graph.nodes[src_node_id].node_desc.startswith("Size"): 
                        size_node_id = in_node.node_id
                        size_node_ids.append(size_node_id)
                    else:
                        input_node_id = in_node.node_id
                m = re.search(r'View\((-?\d+)\)', node.node_desc)
                if m is None:
                    size_output_names = [output_names[size_node_id] for size_node_id in size_node_ids]
                    layer_call = "%s = %s.view(%s)" % (output_name,
                                                       output_names[input_node_id],
                                                       ", ".join(size_output_names))
                else:
                    size = int(m.group(1))
                    layer_call = "%s = %s.view(%s, %d)" % (output_name,
                                                           output_names[input_node_id],
                                                           output_names[size_node_id],
                                                           size)
            elif node.node_desc.startswith("__getitem__"):
                assert(len(in_edges) == 1)
                m = re.search(r'__getitem__\((\d+)\)', node.node_desc)
                idx = int(m.group(1))
                src_node_id = full_graph.track_source_of_new_node_chain(in_edges[0].node_id) # identity chain in sequential graph
                if "hidden" in full_graph.nodes[src_node_id].node_desc:
                    layer_call = "%s = None" % output_name
                else:
                    layer_call = "%s = %s[%d]" % (output_name,
                                                  output_names[in_edges[0].node_id],
                                                  idx)
            elif node.node_desc.startswith("Add"):
                assert(len(in_edges) == 2)
                node1 = in_edges[0]
                node2 = in_edges[1]
                if len(full_graph.edges[node1.node_id]) > 1:
                    tmp = node1
                    node1 = node2
                    node2 = tmp
                layer_call = "%s = %s + %s" % (output_names[node1.node_id],
                                               output_names[node1.node_id],
                                               output_names[node2.node_id])
                output_names[node.node_id] = output_names[node1.node_id]
            elif node.node_desc.startswith("Identity"):
                assert(len(in_edges) == 1)
                node1 = in_edges[0]
                layer_call = "%s = %s" % (output_name,
                                          output_names[node1.node_id])
            elif node.node_desc.startswith("Mul"):
                assert(len(in_edges) == 2)
                node1 = in_edges[0]
                node2 = in_edges[1]
                if len(full_graph.edges[node1.node_id]) > 1:
                    tmp = node1
                    node1 = node2
                    node2 = tmp
                layer_call = "%s = %s * %s" % (output_names[node1.node_id],
                                               output_names[node1.node_id],
                                               output_names[node2.node_id])
                output_names[node.node_id] = output_names[node1.node_id]
            elif node.node_desc.startswith("Concat"):
                m = re.search(r'Concat\((-?\d+)\)', node.node_desc)
                dim = int(m.group(1))
                layer_call = "%s = torch.cat([%s], %d)" % (
                    output_name,
                    ", ".join([output_names[in_node.node_id]
                               for in_node in in_edges]), dim)
            elif node.node_desc.startswith("Transpose"):
                m = re.search(r'Transpose\((.+)\)', node.node_desc)
                args = m.group(1)
                assert(len(in_edges) == 1)
                node1 = in_edges[0]
                layer_call = "%s = %s.transpose(%s)" % (output_name, output_names[node1.node_id],
                                                        args)
            elif node.node_desc.startswith("hidden"):
                pass
            elif node.node_desc == "self.get_seq_lens":
                assert(len(in_edges) == 1)
                in_node = in_edges[0]
                layer_call = "%s = %s(%s)" % (output_name, node.node_desc, output_names[in_node.node_id])
            else:
                layer_call = "%s = %s(%s)" % (output_name, layer_name,
                                              ", ".join([output_names[in_node.node_id]
                                                         for in_node in in_edges])) 
                # NOTE: in_edges must be ordered for current node
                # NOTE: it seems that each node/layer is limited to a single output (but with multiple input)
        if layer_call is not None:
            function_definition.append(layer_call)
        
        # next node
        counter += 1
    
    #---- Generate return ----
    # Ensure that outputs of a module are returned in the same order as
    # the original model implementation.
    # Pipedream todo: This might not work as intended for sub-graphs.
    full_graph.populate_depths()
    graph_output_names, _ = get_output_names(subgraph, full_graph, 0) # output_names = { out_node_id: out%d } # counter=0 is dummy
    for key in graph_output_names:
        graph_output_names[key] = output_names[key] # replace with real out%d
    output_names_list = get_tensor_names_list(graph_output_names) # sorted out_node_id's out%d
    function_definition.append("return %s" % get_output_tuple_str(output_names_list)) # (out%d,out%d)

    #---- Declare extra args ----
    for ea in extra_args:
        layer_declarations.append("self.%s = %s" % (ea, ea))

    #---- Generate initialze weight ----
    if initialize_weights and arch.startswith("bert"): # "bert_thomwolf"
        import_statements.append("from bert_thomwolf.modeling2 import BertLayerNorm")
        layer_declarations.append("self.apply(self.init_bert_weights)")
        module_methods.append("""def init_bert_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses 
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617.
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()""")
    elif initialize_weights and arch.startswith("gpt2"): # "gpt_huggingface"
        import_statements.append("from gpt2_huggingface.modeling_utils import Conv1D")
        layer_declarations.append("self.apply(self._init_weights)")
        module_methods.append("""def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (torch.nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)""")
    elif initialize_weights and arch.startswith("vgg"):
        layer_declarations.append("self._initialize_weights()")
        module_methods.append("""def _initialize_weights(self):
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
                torch.nn.init.constant_(m.bias, 0)""")
    elif initialize_weights and arch.startswith("resnet"):
        layer_declarations.append("self._initialize_weights()")
        module_methods.append("""def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)""")
    elif initialize_weights:
        raise NotImplementedError

    #---- Write to file ----
    # Layer declarations are added to the constructor of the module.
    # Function definitions are added to the `forward()' method of the
    # module.
    with open(MODEL_TEMPLATE_FILE, 'r') as f:
        model_template = f.read()
        model = model_template % {
           "import_statements": "\n".join(import_statements),
           "module_name": module_name,
           "extra_args": "" if len(extra_args) == 0 else ", " + ", ".join(extra_args),
           "layer_declarations": "\n        ".join(layer_declarations),
           "module_methods": "\n\n".join(module_methods),
           "inputs": ", ".join(get_tensor_names_list(input_names)), # sorted in_node_id's input%d
           "function_definition": "\n        ".join(function_definition)
        }
    with open(output_filename, 'w') as f:
        f.write(model)
    print("generated: %s" % output_filename)

    return layer_names, extra_args

def generate_init_code(subgraphs, full_graph, all_layer_names, all_extra_args, arch, output_filename, verbose=False):
    # PyTorch modules are the names given to the generated vlayers (which are
    # of type torch.nn.Module).
    # Python modules are the names given to the filenames containing these
    # generated torch.nn.Modules.
    pytorch_modules = [] 
    python_modules = [] 
    for i in range(len(subgraphs)):
        pytorch_modules.append("vLayer%d" % i)
        python_modules.append("vlayer%d" % i)

    #---- Generate per-vlayer import statements ---- 
    import_statements = ["from .%s import %s" % (python_module, pytorch_module)
                         for (python_module, pytorch_module) in zip(python_modules, pytorch_modules)]

    #---- Generate vlayer_declarations ---- 
    vlayer_declarations = []
    for i, (subgraph, layer_names, extra_args, pytorch_module) in enumerate(zip(subgraphs, all_layer_names, all_extra_args, pytorch_modules)):
        if len(extra_args) == 0:
            vlayer_declarations.append("vlayer%d = %s()" % (i, pytorch_module))
        elif len(extra_args) == 1 and extra_args[0] == CONFIG: # bert, gpt2
            vlayer_declarations.append("vlayer%d = %s(%s)" % (i, pytorch_module, CONFIG))
        else:
            raise NotImplementedError
    
    #---- Generate args ---- 
    args = []
    for extra_args in all_extra_args:
        if len(extra_args) == 0:
            pass
        elif len(extra_args) == 1 and extra_args[0] == CONFIG: # bert, gpt2
            if CONFIG not in args:
                args.insert(0, CONFIG) # always make CONFIG 1st args
        else:
            raise NotImplementedError

    #---- Generate per-vlayer [input%d], [out%d] ---- 
    output_counter = 0 # global counter for out%d
    output_names = {} # global { node_id: input%d/out%d }
    graph_input_names = get_input_names(full_graph, full_graph, check_vlayers=False) # full graph inputs
    for key in graph_input_names:
        output_names[key] = graph_input_names[key]
    subgraph_inputs = [] # per-subgraph [input%d]
    subgraph_outputs = [] # per-subgraph [out%d]
    for i, subgraph in enumerate(subgraphs):
        subgraph_input_names = get_input_names(subgraph, full_graph) # local input%d
        subgraph_output_names, output_counter = get_output_names(
            subgraph, full_graph, output_counter) # global out%d already
        for key in subgraph_input_names: # index by node_id
            subgraph_input_names[key] = output_names[key] # replace to global input%d/out%d
        for key in subgraph_output_names: # index by node_id
            output_names[key] = subgraph_output_names[key] # record to global dict
        #
        subgraph_inputs.append(get_tensor_names_list(subgraph_input_names)) 
        subgraph_outputs.append(get_tensor_names_list(subgraph_output_names)) 

    #---- Generate model [(vlayer%d, [input%d], [out%d])] ---- 
    model = ["(%s, [%s], [%s])" % (x[0],
                                    ", ".join(["\"%s\"" % y for y in x[1]]),
                                    ", ".join(["\"%s\"" % y for y in x[2]]))
                for x in zip(python_modules, subgraph_inputs, subgraph_outputs)]
    model.append("(criterion, [\"%s\"], [\"loss\"])" % subgraph_outputs[-1][0]) # NOTE: always assume single logit to criterion (although bert pretrained might have multiple outputs, they are packed into a single tuple)

    #---- Write __init__.py ---- 
    with open(output_filename, 'w') as f1, open(INIT_TEMPLATE_FILE, 'r') as f2:
        template = f2.read()
        init = template % {
            "import_statements": "\n".join(import_statements),
            "arch": arch,
            "args": "" if len(args) == 0 else ", ".join(args)+", ",
            "vlayer_declarations": "\n    ".join(vlayer_declarations).replace("self.", ""),
            "model": ",\n        ".join(model),
        }
        f1.write(init)
    print("generated: %s" % output_filename)

def generate(seq_graph, output_dir, arch, init_weight=True, verbose=False):
    _assumption()
    
    # Add base dir
    assert os.path.exists(output_dir)
    output_dir = os.path.join(output_dir, "code")
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input graph
    full_graph = graph.Graph.from_str(seq_graph)

    # Remove all unneeded sinks that are not used, makes code generation easier.
    sinks = full_graph.sinks()
    for sink in sinks:
        if sink.node_desc.startswith("__getitem__"):
            full_graph.remove_node(sink)
    
    # Partition into isolated subgraphs (each is a vlayer)
    subgraphs = full_graph.partition_graph()
    assert len(subgraphs) >= 1, "graph must be at least one vlayer"
    
    # Convert subgraph to vlayer code
    all_layer_names = [] # per-subgraph layer_names
    all_extra_args = [] # per-subgraph extra_args
    for i, subgraph in enumerate(subgraphs):
        module_name = "vLayer%d" % i
        module_filename = "vlayer%d.py" % i
        
        layer_names, extra_args = convert_subgraph_to_module(subgraph, full_graph, module_name, init_weight, arch, os.path.join(output_dir, module_filename), verbose=verbose)
        
        all_layer_names.append(layer_names)
        all_extra_args.append(extra_args)
    
    # Generate __init__ code
    generate_init_code(subgraphs, full_graph, all_layer_names, all_extra_args, arch, os.path.join(output_dir, "__init__.py"), verbose=verbose)

    print("--- code generated ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generating code from created graph")
    parser.add_argument("--input_dir", default="", type=str,
                        help="input directory to vanilla 'graph.txt'")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="output directory to sequential 'seq_graph.txt' and generated code")
    parser.add_argument("--arch", required=True, type=str,
                        help="model name in code")
    parser.add_argument('--no_init_weight', default=False, action="store_true",
                        help="not initialize weight in code")
    parser.add_argument("--verbose", default=False, action='store_true')
    args = parser.parse_args()
    
    if args.input_dir:
        ### partition the graph
        par_graph = partition(args.input_dir, verbose=args.verbose)
        
        ### sequentialize the graph
        seq_graph = sequentialize(par_graph, args.output_dir, verbose=args.verbose)
    else: 
        ### read seq_graph.txt directly
        seq_graph = graph.load_graph(args.output_dir, graph.SEQ_FILENAME)
        seq_graph = str(seq_graph)
        
    ### generate the code
    generate(seq_graph, args.output_dir, args.arch, init_weight=not args.no_init_weight, verbose=args.verbose)
    

    
