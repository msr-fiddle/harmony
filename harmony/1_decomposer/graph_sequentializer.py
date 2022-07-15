# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import OrderedDict
import os
import graph


def partition(input_dir, output_dir=None, verbose=False):
    gr = graph.load_graph(input_dir, graph.GRAPH_FILENAME)
    if verbose: 
        print("--- start partition ---")

    # Remove inputs in graph, since inputs should always be in the first vlayer. 
    # (We will add the inputs back after partition)
    # NOTE: assume sources are the last fan-in for their consumer nodes
    sources = gr.sources()
    nodes_to_remove = OrderedDict()
    for source in sources:
        if source.node_desc.startswith("Input"):
            nodes_to_remove[source] = []
            for out_node in gr.edges[source.node_id]:
                nodes_to_remove[source].append(out_node)
            gr.remove_node(source)
    if verbose: 
        print("sources to remove: {}".format([str(node) for node in nodes_to_remove.keys()]))

    # Remove all unneeded sinks that are not used, makes code generation easier.
    sinks = gr.sinks()
    for sink in sinks:
        if sink.node_desc.startswith("__getitem__"):
            gr.remove_node(sink)
            if verbose: 
                print("sink to remove: {}".format(sink))
    
    # Make DAG and sort it
    antichain_gr = gr.antichain_dag()
    if verbose: 
        print("Antichain Graph:\n{}".format(antichain_gr))
    states = antichain_gr.topological_sort() # out-of-place
    if verbose: 
        print("\nstates (sorted AntichainNodes):")
        for node in states:
            print(str(node))
        print("\nTotal number of states (sorted AntichainNodes): %d" % len(states))
    
    # Treat each node (of original graph) as an vlayer
    # Follow sorting index (of AntiChainNodes) to assign node's vlayer id (vLayer id)
    # (Results might have multiple nodes in the same vlayer, but still works.)
    # (Results of vlayer ids follows topological ordering.)
    partial_splits = range(1, len(states)+1)
    if verbose:
        print("\npartial_splits = {}".format(partial_splits))
        # > partial_splits=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 26, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 52, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 73, 78, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    start_point = 0
    vlayer_id = 0
    for split in partial_splits:
        if verbose:
            print("\tcurrent split = [{},{})".format(start_point, split))
        
        predecessors = gr.all_predecessors(states[split-1].antichain) # inclusive
        if verbose:
            print("\t{}'s predecessors = {}".format(
                    str(states[split-1]), 
                    [predecessor.node_id for predecessor in predecessors] ))
        
        set_vlayer = False
        for predecessor in predecessors:
            if predecessor.vlayer_id is None:
                predecessor.set_vlayer_id(vlayer_id)
                if verbose:
                    print("\t\t{} set_vlayer_id to {}".format(
                            predecessor.node_id, vlayer_id))
                set_vlayer = True
        
        start_point = split
        if set_vlayer: # make vlayer_id continous
            vlayer_id += 1
    if verbose:
        print("Total number of vlayers: %d" % vlayer_id)
    
    # Set inputs as first vlayer; Add back removed inputs to graph
    for source in nodes_to_remove:
        for out_node in nodes_to_remove[source]:
            source.set_vlayer_id(0)
            gr.add_edge(source, out_node)

    # Write result graph
    if output_dir is not None:
        graph.save_graph(gr, output_dir, graph.PAR_FILENAME)
    print("--- graph partitioned ---")

    return str(gr)
    
def sequentialize(par_graph, output_dir=None, verbose=False):
    
    gr = graph.Graph.from_str(par_graph) # graph.load_graph(input_dir, graph.PAR_FILENAME)
    if verbose:
        print("--- start sequentialize ---")
    
    # NOTE: different topological ordering results in different sequential Identity chains, which results in different performances.
    # Future work can be done to improve this performance by sorting differently.
    gr.sequentialize_graph(verbose)
    if verbose:
        gr.print_ordered_vlayer_nodes()
    # all vlayers are now sequential
        
    # Write back
    if output_dir is not None:
        graph.save_graph(gr, output_dir, graph.SEQ_FILENAME)
    print("--- graph sequentialized ---")

    return str(gr)


    
