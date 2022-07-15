# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import graphviz
import os
from collections import OrderedDict
from textwrap import shorten
import sys; sys.setrecursionlimit(5000)
# print("recursion limit={}".format(sys.getrecursionlimit()))

class Graph(object):
    def __init__(self, node=None):
        self.nodes = {} # { "node_id": node obj }
        if node is not None:
            self.nodes[node.node_id] = node
        self.edges = {} # { "node_id": [fan-out nodes] }
        self.in_edges = {} # { "node_id": [fan-in nodes] }

        self._predecessors = {}
        self._successors = {}
        self._augmented_antichains = {}
        self._deaugmented_augmented_antichains = {}
        self._next_antichains = {}
        self._antichain_dag = None

        self._colors = ['lightblue', 'green', 'grey', 'firebrick1',
                        'gold', 'chocolate1', 'beige']

        if node is not None:
            self.in_edges[node.node_id] = list()

    def copy(self):
        gr = Graph()
        for node_id in self.in_edges:
            for node2 in self.in_edges[node_id]:
                gr.add_edge(node2, self.nodes[node_id])
        # confirm fan-in ordering is kept
        for node_id, in_nodes in gr.in_edges.items(): # { "node_id": [fan-in nodes] }
            assert [n.node_id for n in in_nodes] == [n.node_id for n in self.in_edges[node_id]]
        
        return gr

    def sources(self):
        sources = []
        for node_id in self.nodes:
            if node_id not in self.in_edges or len(self.in_edges[node_id]) == 0:
                sources.append(self.nodes[node_id])
        return sources

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def remove_node(self, node):
        del self.nodes[node.node_id]
        if node.node_id in self.edges:
            out_nodes = self.edges[node.node_id]
            del self.edges[node.node_id]
            for out_node in out_nodes:
                self.in_edges[out_node.node_id].remove(node) # NOTE: can change the fan-in order
        if node.node_id in self.in_edges:
            in_nodes = self.in_edges[node.node_id]
            del self.in_edges[node.node_id]
            for in_node in in_nodes:
                self.edges[in_node.node_id].remove(node) # NOTE: can change the fan-out order

    def sinks(self):
        sinks = []
        for node_id in self.nodes:
            if node_id not in self.edges or len(self.edges[node_id]) == 0:
                sinks.append(self.nodes[node_id])
        return sinks

    def reset(self):
        self._predecessors = {}
        self._successors = {}

    def add_edge(self, node1, node2): # node1 -> node2
        if node1.node_id not in self.nodes:
            self.nodes[node1.node_id] = node1
        if node2.node_id not in self.nodes:
            self.nodes[node2.node_id] = node2

        if node2.node_id not in self.in_edges:
            self.in_edges[node2.node_id] = list()
        self.in_edges[node2.node_id].append(node1) # NOTE: always the last fan-in 
        if node1.node_id not in self.edges:
            self.edges[node1.node_id] = list()
        self.edges[node1.node_id].append(node2) # NOTE: always the last fan-out

    def remove_edge(self, node1, node2): # node1 -> node2
        self.edges[node1.node_id].remove(node2) # NOTE: can change the fan-out order
        self.in_edges[node2.node_id].remove(node1) # NOTE: can change the fan-in order
    
    def replace_in_edge(self, old_src, dst, new_src):
        """ (old_src->dst) to (new_src->dst), where in_edge order of dst node is kept."""
        assert isinstance(old_src, Node) and isinstance(dst, Node) and isinstance(new_src, Node)
    
        self.edges[old_src.node_id].remove(dst)
        
        if new_src.node_id not in self.nodes:
            self.nodes[new_src.node_id] = new_src
        if new_src.node_id not in self.edges:
            self.edges[new_src.node_id] = list()
        self.edges[new_src.node_id].append(dst)
        
        # inplace replace in_edge of dst
        for i, in_node in enumerate(self.in_edges[dst.node_id]):
            if in_node.node_id == old_src.node_id:
                self.in_edges[dst.node_id][i] = new_src
        
    def populate_depths(self):
        # Helper method that annotates each node in the graph with its depth from the sink.
        sources = self.sources()
        sources[0].depth = 1
        queue = [sources[0]]
        while len(queue) > 0:
            node = queue.pop(-1)
            if node.node_id not in self.edges: continue
            for out_node in self.edges[node.node_id]:
                if out_node.depth is None or out_node.depth < (node.depth + 1):
                    out_node.depth = node.depth + 1
                queue.append(out_node)

    def partition_graph(self): # generate a list of isolated subgraphes(vlayers)
        vlayer_ids = set()
        for node_id in self.nodes:
            vlayer_ids.add(self.nodes[node_id].vlayer_id)
        if len(vlayer_ids) == 1:
            return [self.copy()]
        subgraphs = []
        for vlayer_id in sorted(vlayer_ids):
            subgraphs.append(self.partition_graph_helper(vlayer_id))
        
        # confirm fan-in ordering is kept
        # for subgraph in subgraphs:
        #     for node_id, in_nodes in subgraph.in_edges.items(): # { "node_id": [fan-in nodes] }
        #         assert [n.node_id for n in in_nodes] == [n.node_id for n in self.in_edges[node_id]], "[fan-in ordering not kept during partition_graph] {} v.s. {}".format([n.node_id for n in in_nodes], [n.node_id for n in self.in_edges[node_id]])
        
        return subgraphs

    def partition_graph_helper(self, vlayer_id): # generate a copy of subgraph of vlayer_id; subgraphes are isolated from each other
        subgraph = Graph() # node and edge residing in this vlayer (excluding edging in/out from/to other vlayers)
        # traverse full-graph nodes to add my vlayer node
        for node_id in self.nodes:
            if self.nodes[node_id].vlayer_id == vlayer_id:
                subgraph.add_node(self.nodes[node_id]) 
        # traverse sub-graph nodes to add my vlayer edge
        for node_id in subgraph.nodes:
            if node_id not in self.in_edges: continue
            for in_node in self.in_edges[node_id]: # follow fan-in order
                if in_node.vlayer_id == vlayer_id:
                    subgraph.add_edge(in_node, self.nodes[node_id])
        return subgraph

    def chain_nodes(self):
        chain_nodes = list()
        for node in self.nodes.values():
            if node.node_id in self.edges and len(self.edges[node.node_id]) == 1 \
                and node.node_id in self.in_edges and len(self.in_edges[node.node_id]) == 1:
                chain_nodes.append(node)
        return chain_nodes

    def aggregate(self, sum_activations=False):
        forward_compute_time = 0.0
        backward_compute_time = 0.0
        parameter_size = 0.0
        activation_size = 0.0
        for node in self.nodes.values():
           forward_compute_time += node.forward_compute_time
           backward_compute_time += node.backward_compute_time
           parameter_size += node.parameter_size
           if sum_activations:
               activation_size += node.activation_size
           else:
               if node.node_id not in self.in_edges or len(self.in_edges[node.node_id]) == 0:
                   activation_size += node.activation_size
        return [forward_compute_time, backward_compute_time, parameter_size, activation_size]

    def topological_sort(self):
        # Algorithm from https://en.wikipedia.org/wiki/Topological_sorting
        self.sorted_nodes = []
        self.marked_nodes = set()
        self.temporarily_marked_nodes = set()
        nodes = list(self.nodes.values())
        nodes.sort(key=lambda x: x.node_desc)
        for node in nodes:
            if node.node_id in self.marked_nodes:
                continue
            self.topological_sort_helper(node.node_id)
        return [self.nodes[node_id] for node_id in self.sorted_nodes]

    def topological_sort_helper(self, node_id):
        if node_id in self.marked_nodes:
            return
        if node_id in self.temporarily_marked_nodes:
            raise Exception("Graph has a cycle")
        self.temporarily_marked_nodes.add(node_id)
        if node_id in self.edges:
            out_nodes = list(self.edges[node_id])
            out_nodes.sort(key=lambda x: (x.node_desc, x.height))
            for out_node in out_nodes:
                self.topological_sort_helper(out_node.node_id)
        self.marked_nodes.add(node_id)
        self.temporarily_marked_nodes.remove(node_id)
        self.sorted_nodes.insert(0, node_id)

    def predecessors(self, node):
        if node in self._predecessors:
            return self._predecessors[node]
        predecessors = set()
        if node not in self.in_edges:  # Source node
            return predecessors
        for in_node in self.in_edges[node]:
            predecessors.add(in_node)
            predecessors.update(self.predecessors(in_node.node_id))
        self._predecessors[node] = predecessors
        return self._predecessors[node]

    def all_predecessors(self, antichain):
        all_predecessors = set()
        for antichain_node in antichain:
            all_predecessors.update(self.predecessors(antichain_node))
            all_predecessors.add(self.nodes[antichain_node])
        return all_predecessors

    def successors(self, node):
        if node in self._successors:
            return self._successors[node]
        successors = set()
        if not node in self.edges:  # Sink node
            return successors
        for out_node in self.edges[node]:
            successors.add(out_node)
            successors.update(self.successors(out_node.node_id))
        self._successors[node] = successors
        return self._successors[node]

    def augment_antichain(self, antichain):
        antichain_key = tuple(sorted(antichain))
        if antichain_key in self._augmented_antichains:
            return self._augmented_antichains[antichain_key]
        extra_nodes = set()
        all_predecessors = set()
        for antichain_node in antichain:
            predecessors = self.predecessors(antichain_node)
            all_predecessors = all_predecessors.union(predecessors)
        for antichain_node in antichain:
            predecessors = self.predecessors(antichain_node)
            for predecessor in predecessors:
                for out_node in self.edges[predecessor.node_id]:
                    if out_node not in predecessors and out_node.node_id != antichain_node:
                        extra_nodes.add(predecessor.node_id)
        self._augmented_antichains[antichain_key] = list(extra_nodes) + antichain
        return self._augmented_antichains[antichain_key]

    def deaugment_augmented_antichain(self, augmented_antichain):
        augmented_antichain_key = tuple(sorted(augmented_antichain))
        if augmented_antichain_key in self._deaugmented_augmented_antichains:
            return self._deaugmented_augmented_antichains[augmented_antichain_key]
        nodes_to_remove = set()
        all_successors = set()
        for augmented_antichain_node in augmented_antichain:
            successors = self.successors(augmented_antichain_node)
            for augmented_antichain_node_prime in augmented_antichain:
                if self.nodes[augmented_antichain_node_prime] in successors:
                    nodes_to_remove.add(augmented_antichain_node)
        antichain = list()
        for augmented_antichain_node in augmented_antichain:
            if (augmented_antichain_node not in nodes_to_remove and \
                augmented_antichain_node not in antichain):
                antichain.append(augmented_antichain_node)
        self._deaugmented_augmented_antichains[augmented_antichain_key] = antichain
        return self._deaugmented_augmented_antichains[augmented_antichain_key]

    def is_next_antichain(self, augmented_antichain, new_node):
        successors = self.successors(new_node)
        augmented_antichain_set = set(augmented_antichain)
        for successor in successors:
            if successor.node_id in augmented_antichain_set:
                return False
        return True

    def construct_antichain(self, augmented_antichain, old_node, new_node):
        new_antichain = [x if x != old_node else new_node for x in augmented_antichain]
        return self.deaugment_augmented_antichain(new_antichain)

    def next_antichains(self, antichain):
        antichain_key = tuple(sorted(antichain))
        if antichain_key in self._next_antichains:
            return self._next_antichains[antichain_key]

        next_antichains = []
        antichain_set = set(antichain)
        augmented_antichain = self.augment_antichain(antichain)
        for augmented_antichain_node in augmented_antichain:
            next_nodes = self.edges[augmented_antichain_node] if augmented_antichain_node in self.edges else []
            for next_node in next_nodes:
                if next_node.node_id in antichain_set:
                    continue
                if self.is_next_antichain(augmented_antichain, next_node.node_id):
                    next_antichain = self.construct_antichain(augmented_antichain,
                                                              augmented_antichain_node,
                                                              next_node.node_id)
                    next_antichains.append(next_antichain)
        self._next_antichains[antichain_key] = next_antichains
        return self._next_antichains[antichain_key]

    def antichain_dag(self):
        if self._antichain_dag is not None:
            return self._antichain_dag

        antichain_dag = Graph()
        antichain_id = 0
        antichain = [self.sources()[0].node_id]
        source_node = AntichainNode("antichain_%d" % antichain_id, self.augment_antichain(antichain))
        antichain_dag.source = source_node
        antichain_queue = [antichain]
        antichain_mapping = {tuple(sorted(antichain)): source_node}

        while len(antichain_queue) > 0:
            antichain = antichain_queue.pop(0)
            antichain_key = tuple(sorted(antichain))
            if antichain_key in self._next_antichains:
                continue
            next_antichains = self.next_antichains(antichain)
            for next_antichain in next_antichains:
                next_antichain_key = tuple(sorted(next_antichain))
                if next_antichain_key not in antichain_mapping:
                    antichain_id += 1
                    next_antichain_node = AntichainNode("antichain_%d" % antichain_id, self.augment_antichain(next_antichain))
                    antichain_mapping[next_antichain_key] = next_antichain_node
                antichain_dag.add_edge(antichain_mapping[antichain_key],
                                       antichain_mapping[next_antichain_key])
                antichain_queue.append(next_antichain)

        self._antichain_dag = antichain_dag
        
        # confirm fan-in ordering is kept (maybe too strong here)
        # for node_id, in_nodes in antichain_dag.in_edges.items(): # { "node_id": [fan-in nodes] }
        #     assert [n.node_id for n in in_nodes] == [n.node_id for n in self.in_edges[node_id]] 
        
        return antichain_dag
    
    def __str__(self): # graph.txt
        strs = []
        for node in self.nodes.values():
            strs.append(str(node))
        for node in self.nodes.values():
            if node.node_id not in self.in_edges:
                continue
            for in_node in self.in_edges[node.node_id]: # fan-in order kept
                strs.append("\t%s -- %s" % (in_node.node_id, node.node_id))
        return "\n".join(strs)

    @staticmethod
    def from_str(graph_str): # graph.txt
        gr = Graph()
        graph_str_lines = graph_str.strip().split('\n')
        for graph_str_line in graph_str_lines:
            if not graph_str_line.startswith('\t'):
                node = Node.from_str(graph_str_line.strip())
                gr.nodes[node.node_id] = node
            else:
                [in_node_id, node_id] = graph_str_line.strip().split(" -- ")
                if node_id not in gr.in_edges:
                    gr.in_edges[node_id] = [gr.nodes[in_node_id]]
                else: # fan-in order kept
                    gr.in_edges[node_id].append(gr.nodes[in_node_id])
                if in_node_id not in gr.edges:
                    gr.edges[in_node_id] = [gr.nodes[node_id]]
                else:
                    gr.edges[in_node_id].append(gr.nodes[node_id])
        return gr
    
    def to_dot(self, arch): # graph.dot.pdf
        dot = graphviz.Digraph()
        for node in self.nodes.values():
            in_node_ids = ", ".join([in_node.node_id for in_node in self.in_edges[node.node_id]]) if node.node_id in self.in_edges else ""
            node_desc = "[in: %s]\n"%(in_node_ids)
            # node_desc += str(shorten("%s -- %s"%(node.node_id,node.node_desc), width=50, placeholder="..."))
            node_desc += "%s -- %s"%(node.node_id, node.node_desc)
            if node.vlayer_id is not None:
                node_desc += " (vLayer %d)" % node.vlayer_id
            # node_desc = shorten(node_desc, width=64, placeholder="...")
            if node.vlayer_id is not None:
                color = self._colors[node.vlayer_id % len(self._colors)]
                dot.node(node.node_id, node_desc,
                   color=color, style='filled')
            else:
                dot.node(node.node_id, node_desc)
        for node in self.nodes.values():
            if node.node_id not in self.in_edges:
                continue
            for in_node in self.in_edges[node.node_id]: # fan-in order kept
                dot.edge(in_node.node_id, node.node_id) # NOTE: can not show ordering
        dot.render(arch)

    def to_dot_legacy(self, arch): # graph.dot.legacy.pdf
        dot = graphviz.Digraph()
        for node in self.nodes.values():
            node_desc = "%s\n[forward_compute_time=%.3f,backward_compute_time=%.3f,activation_size=%s,parameter_size=%.1f]" % (
                node.node_desc, node.forward_compute_time, node.backward_compute_time,
                node.activation_size, node.parameter_size)
            if node.vlayer_id is not None:
                color = self._colors[node.vlayer_id % len(self._colors)]
                dot.node(node.node_id, node_desc,
                   color=color, style='filled')
            else:
                dot.node(node.node_id, node_desc)
        for node in self.nodes.values():
            if node.node_id not in self.edges:
                continue
            for out_node in self.edges[node.node_id]:
                dot.edge(node.node_id, out_node.node_id)
        dot.render(arch)

    def get_ordered_vlayer_node_ids(self):
        vlayer_node_ids = {} # { vlayer_id: [node_id, node_id, ..., new_node_id] }
        for node in self.nodes.values():
            assert node.vlayer_id is not None, "graph node must be vlayered"
            if node.vlayer_id not in vlayer_node_ids:
                vlayer_node_ids[node.vlayer_id] = []
            vlayer_node_ids[node.vlayer_id].append(node.node_id)
        ordered_vlayer_node_ids = OrderedDict() # { ordered vlayer_id: ordered [node_id, ..., new_node_id] }
        for vlayer_id in sorted(list(vlayer_node_ids.keys())):        
            # oem node_id
            nids = sorted([int(node_id.split("node")[-1]) for node_id in vlayer_node_ids[vlayer_id] if node_id.startswith("node")])
            # new_node_id
            new_nids = sorted([int(node_id.split("node")[-1]) for node_id in vlayer_node_ids[vlayer_id] if node_id.startswith("nw_node")])
            # recreate two
            ordered_vlayer_node_ids[vlayer_id] = \
            ["node%d"%nid for nid in nids] + ["nw_node%d"%nid for nid in new_nids]
        return ordered_vlayer_node_ids
    
    def print_ordered_vlayer_nodes(self):
        vlayer_node_ids = self.get_ordered_vlayer_node_ids()
        print("[vlayer_id : nodes] =")
        for vlayer_id, node_ids in vlayer_node_ids.items():
            print("{} : {}".format(vlayer_id,node_ids))
            for node_id in node_ids:
                print("\t{} -- {}".format(node_id, self.nodes[node_id].node_desc))
   
    def sequentialize_graph(self, verbose=False):                
        vlayer_node_ids = self.get_ordered_vlayer_node_ids() # { ordered vlayer_id: ordered [node_id, ...] }
        
        # find branch outs and seqentialize with Identiy nodes
        new_node_id = 1 # max([int(node_id.split("node")[-1]) for node_id in self.nodes.keys()]) + 1
        for vlayer_id, node_ids in vlayer_node_ids.items():
            for node_id in node_ids:
                if ("nw" not in node_id) and (node_id in self.edges): # not identity node && fan-out exists
                    # record current node's fan-outs
                    out_vlayer_node_ids = {} # { out_vlayer_id: [out_node_ids] }
                    for out_node in self.edges[node_id]:
                        if out_node.vlayer_id not in out_vlayer_node_ids:
                            out_vlayer_node_ids[out_node.vlayer_id] = []
                        out_vlayer_node_ids[out_node.vlayer_id].append(out_node.node_id)
                    # leave only out vlayers
                    if vlayer_id in out_vlayer_node_ids:
                        del out_vlayer_node_ids[vlayer_id]
                    for out_vlayer_id in out_vlayer_node_ids.keys():
                        assert out_vlayer_id > vlayer_id # no circular vlayers
                    
                    # distinct_vlayer_ids = set()
                    # for out_node in self.edges[node_id]:
                    #     distinct_vlayer_ids.add(out_node.vlayer_id)
                    # distinct_vlayer_ids.discard(node.node_id) # pure out vlayer_ids
                    # distinct_vlayer_ids = set(out_vlayer_node_ids.keys()).discard(node_id) # pure out vlayer_ids
                    # confirm no circular fan-out
                    # for distinct_vlayer_id in distinct_vlayer_ids:
                    #     assert distinct_vlayer_id > vlayer_id:
                    
                    # check whether all fan-out are sequential
                    seq = True
                    for out_vlayer_id in out_vlayer_node_ids.keys():
                        if out_vlayer_id != vlayer_id + 1:
                            seq = False
                            break
                    if seq:
                        continue # next node
                    else: # non-sequential fan-out exists                 
                        if verbose:
                            print("non-sequential fan-out on {}. seqentializing.".format(node_id))
                        
                        # create an Identity chain from current node to the farest vlayer
                        vlayer_new_node = {} # {vlayer: new node}
                        prev_node = self.nodes[node_id]
                        for identity_vlayer_id in range(vlayer_id+1, max(out_vlayer_node_ids.keys())+1):
                            new_node = Node("nw_node%d" % new_node_id,
                                            node_desc="Identity",
                                            vlayer_id=identity_vlayer_id)
                            new_node_id += 1
                            self.add_edge(prev_node, new_node)
                            prev_node = new_node
                            vlayer_new_node[identity_vlayer_id] = new_node 
                        
                        # replace edges (current node -> out node) to (Identity node -> out node)
                        for out_vlayer_id, out_node_ids in out_vlayer_node_ids.items():
                            for out_node_id in out_node_ids:
                                self.replace_in_edge(self.nodes[node_id], self.nodes[out_node_id], vlayer_new_node[out_vlayer_id])
                        
                        # # remove edges of (current node -> out nodes) 
                        # for out_node_ids in out_vlayer_node_ids.values():
                        #     for out_node_id in out_node_ids:
                        #         self.remove_edge(self.nodes[node_id], self.nodes[out_node_id])
                        # # add edge for (Identity node -> out node) 
                        # for out_vlayer_id, out_node_ids in out_vlayer_node_ids.items():
                        #     for out_node_id in out_node_ids:
                        #         self.add_edge(vlayer_new_node[out_vlayer_id], self.nodes[out_node_id])
                     
    def track_source_of_new_node_chain(self, cur_node_id, verbose=False):
        """ 'source' node -> new node: Identity -> new node: Identity -> ... -> curent new node Identity """
        if cur_node_id.startswith("node"):
            return cur_node_id
        else: # current is new node
            node_id = cur_node_id
            while True: # loop through new nodes
                if verbose: print("[track source] node_id = {}".format(node_id))
                if node_id.startswith("node"):
                    break
                assert self.nodes[node_id].node_desc.startswith("Identity")
                assert len(self.in_edges[node_id])==1
                node_id = self.in_edges[node_id][0].node_id
            return node_id

class Node(object):
    def __init__(self, node_id, node_desc="", forward_compute_time=0.0,
                 backward_compute_time=0.0, activation_size=0.0, parameter_size=0.0,
                 vlayer_id=None): # TODO: remove forward_compute_time/backward_compute_time/activation_size/parameter_size
        self.node_id = node_id
        self.node_desc = node_desc
        self.forward_compute_time = forward_compute_time
        self.backward_compute_time = backward_compute_time
        self.activation_size = activation_size
        self.parameter_size = parameter_size
        self.vlayer_id = vlayer_id
        self.depth = None
        self.height = None

    def set_vlayer_id(self, vlayer_id):
        self.vlayer_id = vlayer_id

    def __str__(self):
        vlayer_id_str = " -- vlayer_id=%d" % self.vlayer_id if self.vlayer_id is not None else ""
        node_desc = self.node_desc.replace('\n', "")
        activation_size = ("%s" % self.activation_size).replace(", ", "; ")
        return "%s -- %s -- forward_compute_time=%.3f, backward_compute_time=%.3f, activation_size=%s, parameter_size=%.3f%s" % (
            self.node_id, node_desc, self.forward_compute_time, self.backward_compute_time,
            activation_size, self.parameter_size, vlayer_id_str)

    @staticmethod
    def from_str(node_str):
        node_str_tokens = node_str.strip().split(" -- ")
        node_id = node_str_tokens[0]
        node_desc = node_str_tokens[1]
        node_metadata = node_str_tokens[2]
        vlayer_id = None
        if len(node_str_tokens) > 3:
            vlayer_id = int(node_str_tokens[3].split("=")[1])
        [forward_compute_time, backward_compute_time, activation_size, parameter_size] = node_metadata.split(", ")
        forward_compute_time = float(forward_compute_time.split("=")[1])
        backward_compute_time = float(backward_compute_time.split("=")[1])
        if "[" in activation_size:
            activation_size = activation_size.split("=")[1]
            activation_size = sum([float(x) for x in activation_size.lstrip("[").rstrip("]").split("; ")])
        else:
            activation_size = float(activation_size.split("=")[1])
        parameter_size = float(parameter_size.split("=")[1])
        return Node(node_id, node_desc, forward_compute_time=forward_compute_time,
                    backward_compute_time=backward_compute_time, activation_size=activation_size,
                    parameter_size=parameter_size, vlayer_id=vlayer_id)

class AntichainNode(Node):
    def __init__(self, node_id, antichain, node_desc=""):
        self.antichain = antichain
        self.output_activation_size = 0.0
        super(AntichainNode, self).__init__(node_id, node_desc)

    def __str__(self):
        return "%s -- %s" % (self.node_id, self.antichain)


GRAPH_FILENAME = "graph"
PAR_FILENAME = "par_graph"
SEQ_FILENAME = "seq_graph"

def save_graph(graph, path_dir, fname="graph", base_dir="graph", verbose=True): 
    assert isinstance(graph, Graph)
    assert os.path.exists(path_dir)
    fname = fname.split(".")[0] # strip off extension
    if base_dir is None:
        full_dir = path_dir
    else:
        full_dir = os.path.join(path_dir, base_dir)
        os.makedirs(full_dir, exist_ok=True)

    # graph.to_dot_legacy(os.path.join(full_dir, fname + ".dot.legacy")) 
    graph.to_dot(os.path.join(full_dir, fname + ".dot"))
    with open(os.path.join(full_dir, fname + ".txt"), 'w') as f:
        f.write(str(graph))

    if verbose: print("graph saved to: {}/{}".format(full_dir, fname + ".txt"))

def load_graph(path_dir, fname="graph.txt", base_dir="graph", verbose=True):
    if ".txt" not in fname:
        fname += ".txt"
    if base_dir is None:
        full_path = os.path.join(path_dir, fname)
    else:
        full_path = os.path.join(path_dir, base_dir, fname)
    assert os.path.exists(full_path)
    
    with open(full_path, 'r') as f:
        graph = Graph.from_str(f.read())
    if verbose: print("graph loaded from: {}".format(full_path))
    
    return graph
