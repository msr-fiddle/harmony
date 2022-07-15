# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict as ODict
import numpy as np
import random
from collections import deque

import sys; sys.path.append("../2_profiler")
from prof_data_struct import *
from analyze_profiles import time_of_pack, memory_of_pack, model_size_of_pack

from task_data_struct import Medium, vTask

def find_last_subpack_idx(vt):
    if vt.type == "FWD":
        if not vt.Out['X']:
            spi = 0
        else:
            spi = len(vt.Out['X'].keys()) + 1 - 1
            if vt.layers[0] in vt.Out['X']:
                spi -= 1
    else: # "BWD", "UPD"
        spi = 0
    assert spi >= 0
    return spi

def find_stash_subpack_idx(vt, TASKS):
    l, m = vt.layers[0], vt.In['X'][vt.layers[0]]; assert m.medium == "MSG"
    src_vt = TASKS[m.idx]; assert src_vt.idx == m.idx, "TASKS should be global"
    layers_stash = list(src_vt.Out['X'].keys())
    src_spi = layers_stash.index(l) + 1
    if src_vt.layers[0] in layers_stash:
        src_spi -= 1
    assert src_spi >= 0
    return src_vt, src_spi

def find_output_subpack_idx(vt, TASKS):
    l, m = vt.layers[0], vt.In['X'][vt.layers[0]]; assert m.medium == "MSG"
    src_vt = TASKS[m.idx]; assert src_vt.idx == m.idx, "TASKS should be global"
    src_spi = find_last_subpack_idx(src_vt)
    return src_vt, src_spi

def find_subpack_layers(vt, spi):
    if vt.type == "FWD":
        if not vt.Out['X']:
            assert spi == 0
            layers = vt.layers
        else:
            assert spi >= 0
            layers_stash = list(vt.Out['X'].keys()) + [vt.layers[-1]+1]
            if vt.layers[0] not in layers_stash:
                layers_stash = [vt.layers[0]] + layers_stash
            layers = list(range(layers_stash[spi], layers_stash[spi+1]))
    else: # "BWD", "UPD"
        assert spi == 0
        layers = vt.layers
    return layers

class Event(object):
    def __init__(self, vt, ui, spi, stream, kind, ubs=None):
        self.vt = vt # affiliated vtask
        self.ui = ui # affiliated ubatch index
        self.spi = spi # affiliated subpack index
        self.stream = stream # "SwapIn"/"P2PIn"/"Compute" # for per-stream event queue
        self.kind = kind # "W"/"X"/"Y"/"FWD"/"BWD"
        # five tuple as id
        self.id = "%d.%d.%d.%s.%s"%(self.vt.idx, self.ui, self.spi, self.stream, self.kind)
        # microbatch size
        if ubs is None:
            self._find_ubs() # self.ubs = None or 2
        else:
            self.ubs = int(ubs) # override for MSG
        # subpack as event layers
        self._find_layers() # self.layers = [1] or [1,2,3]
        # dependency
        self.inputs = [] # required input events from other streams
        self.is_done = False # whether executed
        # p2p
        self.peer_id = None # peer event id
        self.pos = None # position idx in event queue
        # exec
        self.begin = 0.0 # begin time in sec
        self.dur = 0.0 # duration time in sec

    def _find_ubs(self):
        if self.kind in ["WB","DEL","ARD","dWB","Update"]:
            self.ubs = None
        elif self.kind in ["FWD","REC","BWD","sX","MSGsX","X","dX","MSGdX","Y","dY","MSGY"]:
            self.ubs = self.vt.ubatchszs[self.ui]
        else:
            raise NotImplementedError

    def _find_layers(self):
        if self.kind in ["WB","REC","BWD","DEL","ARD","dWB","Update"]:
            assert self.spi == 0
            self.layers = self.vt.layers
        elif self.kind in ["FWD"]:
            self.layers = find_subpack_layers(self.vt, self.spi)
        elif self.kind in ["sX","MSGsX"]:
            self.layers = [find_subpack_layers(self.vt, self.spi)[0]]
        elif self.kind in ["X","dX","MSGdX"]:
            assert self.spi == 0
            self.layers = [self.vt.layers[0]]
        elif self.kind in ["Y","dY","MSGY"]:
            assert self.spi >= 0
            self.layers = [self.vt.layers[-1]]
        else:
            raise NotImplementedError
            
    def inputs_add(self, vt, ui, spi, stream, kind):
        idx = vt.idx # if isinstance(vt, vTask) else vt
        spi = find_last_subpack_idx(vt) if spi == -1 else spi
        self.inputs.append("%d.%d.%d.%s.%s"%(idx, ui, spi, stream, kind))
    def inputs_add_ev(self, ev):
        assert isinstance(ev, Event)
        self.inputs.append(ev)
            
    def add_to(self, events, per_stream_events):
        events[self.id] = self
        if self.stream not in per_stream_events:
            per_stream_events[self.stream] = deque([])
        per_stream_events[self.stream].append(self)
        self.pos = len(per_stream_events[self.stream])-1
                
    def register_peer(self, TASKS, ui, spi, stream, kind):
        """ register the peer Event ID """
        assert self.peer_id is None, "only one peer"
        # find peer vt
        if stream == "P2PIn":
            assert self.stream == "P2POut"
            if kind == "dY":
                m = self.vt.Out['dX'][self.vt.layers[0]]   
            elif kind == "X":
                m = self.vt.Out['Y'][self.vt.layers[-1]]
            else:
                raise ValueError
            assert m.medium == "P2P"
        elif stream == "P2POut":
            assert self.stream == "P2PIn"
            if kind == "Y":
                m = self.vt.In['X'][self.vt.layers[0]]
            elif kind == "dX":
                m = self.vt.In['dY'][self.vt.layers[-1]]
            else:
                raise ValueError
            assert m.medium == "P2P"
        else:
            raise ValueError
        peer_vt = TASKS[m.idx]; assert peer_vt.idx == m.idx, "TASKS is global"
        
        idx = peer_vt.idx
        spi = find_last_subpack_idx(peer_vt) if spi == -1 else spi
        self.peer_id = "%d.%d.%d.%s.%s"%(idx, ui, spi, stream, kind)

    def solve_peer(self, events, rank_stream_events):
        """ add peer's inputs and its previous stream Event to my inputs """
        if self.peer_id is None:
            return
        peer = events[self.peer_id]
        self.inputs += peer.inputs 
        if peer.pos-1 >= 0:
            self.inputs.append( 
                rank_stream_events[peer.vt.rank][peer.stream][peer.pos-1] )
        self.inputs = list(set(self.inputs)) # remove double counting
        self.inputs = sorted(self.inputs, key=lambda e: e.id) # for result matching
                   
    @property
    def end(self): # end time in sec
        return self.begin + self.dur

    def show_layers(self):
        # assert list(range(self.layers[0], self.layers[-1]+1)) == list(self.layers)
        if len(self.layers) == 0:
            return "L--"
        elif len(self.layers) == 1:
            return "L%d" % (self.layers[0])
        else:
            return "L%d-%d" % (self.layers[0], self.layers[-1])
    def __str__(self):
        ### id, ubs, layers, [inputs ids], done
        ss = "%s, %s, %s, [%s], %s"%(
              self.id, 
              'U--' if self.ubs is None else 'U%d'%self.ubs,
              self.show_layers(),
              ",".join(inev.id for inev in self.inputs),
              "@%.3f_dur%.3f"%(self.begin, self.dur) if self.is_done else "-")
        return ss
    @property
    def name(self): # for display in chrome
        return "t{} {} {} {}".format(
                self.vt.idx, 
                '' if self.ubs is None else '#%d(%d)'%(self.ui, self.ubs),
                self.show_layers(), 
                self.kind)
        # return "t{} #{} {} {}".format(
        #         self.vt.idx, self.ui, self.show_layers(), self.kind)
        
class UBSConverter(object):
    def __init__(self, ubatchszs_fwd, ubatchszs_bwd, u_bwd, verbose=True):
        """ examples: 
                [4,4,1], [4,4,1], 4
                [4,4,1], [2,2,2,2,1], 2
                [4,4,4,1], [3,3,3,3,1], 3
                [3,3,3,3,1], [4,4,4,1], 4
                [2,2,2,2,1], [4,4,1], 4
        """
        assert sum(ubatchszs_fwd) == sum(ubatchszs_bwd)
        self.ubatchszs_fwd = ubatchszs_fwd
        self.ubatchszs_bwd = ubatchszs_bwd
        self.u_bwd = u_bwd
        self.verbose = verbose
        # convertion
        self._convert_ubatchsize()
        self._map_idx_ufwd_to_ubwd()
        self._map_idx_ubwd_to_ufwd()
            
    def _concat(self, Cs):
        return int(sum(Cs))
    def _split(self, C, U):
        assert isinstance(C, int) and isinstance(U, int)
        if C >= U:
            if C % U == 0:
                ubatchszs = [U] * int(C/U)
            else:
                ubatchszs = [U] * int(C/U) + [ C%U ]
            assert sum(ubatchszs) == C
        else: # not sufficient to split
            ubatchszs = [C]
        return tuple(ubatchszs)
    def _convert_ubatchsize(self):
        """ given ubatchszs, return per-ufwd converted list """
        per_ufwd_converted = [] # [[ubwd, ubwd], [ubwd, ubwd], [ubwd, 1]]
        
        residual = 0
        cnt_converted_ubwd = 0
        for ufwd in self.ubatchszs_fwd:
            # Use previous residual for split
            split_u = self._split(self._concat((residual,ufwd)), self.u_bwd) # (c1,c2) or (c1,res) or (c1,) or (res,)
            residual = 0
            # Within split, check if each u is desired ubwd, if so make them as converted
            converted = []
            for j, u in enumerate(split_u):
                assert cnt_converted_ubwd < len(self.ubatchszs_bwd)
                desired_ubwd = self.ubatchszs_bwd[cnt_converted_ubwd]
                if u == desired_ubwd: # match
                    converted.append(u)
                    cnt_converted_ubwd += 1
                elif u < desired_ubwd: # residual
                    assert j == len(split_u)-1, "residual must be the last split"
                    residual = u
                else:
                    raise ValueError
            # Return converted
            # print("ufwd {} => ubwd {}".format(ufwd, converted))
            per_ufwd_converted.append(converted)
        self.per_ufwd_converted = per_ufwd_converted
        if self.verbose:
            print("[UBSConverter] per_ufwd_converted={}".format(self.per_ufwd_converted))

    def find_converted(self, idx_ufwd):
        """ given microbatch idx of ufwd, return list of converted ubwd """
        return self.per_ufwd_converted[idx_ufwd]

    def _map_idx_ufwd_to_ubwd(self):
        ufwd_to_ubwd = [] # [[ubwd#0, ubwd#1], [ubwd#2,ubwd#3], [ubwd#4,ubwd#5]]
        assert len(self.ubatchszs_fwd) == len(self.per_ufwd_converted)
        cnt = 0
        for converted in self.per_ufwd_converted:
            ufwd_to_ubwd.append([cnt+i for i, _ in enumerate(converted)])
            cnt += len(converted)
        self.ufwd_to_ubwd = ufwd_to_ubwd
        assert cnt == len(self.ubatchszs_bwd)
        if self.verbose:
            print("[UBSConverter] idx of ufwd_to_ubwd={}".format(self.ufwd_to_ubwd))
    def find_idx_ubwd(self, idx_ufwd):
        """ given microbatch idx of ufwd, return list of converted ubwd idx"""
        return self.ufwd_to_ubwd[idx_ufwd]    

    def _map_idx_ubwd_to_ufwd(self):
        ubwd_to_ufwd = [] # [ufwd#0, ufwd#0, ufwd#1, ufwd#1, ufwd#2, ufwd#2]
        assert len(self.ubatchszs_fwd) == len(self.per_ufwd_converted)
        for idx_ufwd, converted in enumerate(self.per_ufwd_converted):
            for _ in range(len(converted)):
                ubwd_to_ufwd.append(idx_ufwd)
        self.ubwd_to_ufwd = ubwd_to_ufwd
        assert len(self.ubatchszs_bwd) == len(self.ubwd_to_ufwd)
        if self.verbose:
            print("[UBSConverter] idx of ubwd_to_ufwd={}".format(self.ubwd_to_ufwd))
    def find_idx_ufwd(self, idx_ubwd):
        """ given microbatch idx of ubwd, return its producer idx of ufwd """
        return self.ubwd_to_ufwd[idx_ubwd]    

def In_WB(vt, events, stream_events, left_vt, delay_enqueue, prefetch_offload):
    ev_w = Event(vt, 0, 0, "SwapIn", "WB")
    if left_vt is None:
        ev_w.add_to(events, stream_events)
    elif prefetch_offload: # prefetch at left vt
        ev_w.inputs_add(left_vt, len(left_vt.ubatchszs)-1, -1, "Compute", left_vt.type)
        if not delay_enqueue: # vPP-["F", "Bc"]
            ev_w.add_to(events, stream_events)
        else: # vDP or vPP-"Bn"
            pass
    else: # no prefetch offload
        ev_w.inputs_add(left_vt, 0, 0, "Compute", "DEL")
        ev_w.add_to(events, stream_events)
    
    return ev_w

def In_X(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ubscvt, ev_w, prefetch_offload, ev_w_comp_out):
    # assert ['F','Bc']
    if vt.has_data: # "Data"
        ev_x = Event(vt, i, 0, "SwapIn", "X")
        assert left_vt is None # 1st task
    elif vt.In["X"][vt.layers[0]].medium == "P2P": # "P2PX"
        ev_x = Event(vt, i, 0, "P2PIn", "X")
        ev_x.register_peer(TASKS, i, -1, "P2POut", "Y") # p2p dependency
        assert ubscvt is None
    elif vt.In["X"][vt.layers[0]].medium == "MSG": # last fwd "MSG" of vPP
        ev_x = Event(vt, i, 0, "SwapIn", "X")
        src_vt, src_spi = find_output_subpack_idx(vt, TASKS)
        ev_x.inputs_add(src_vt, i, src_spi, "CPU", "MSGY") # msg dependency
    elif vt.In["X"][vt.layers[0]].medium == "SWP": # "LocalX" of vDP
        ev_x = Event(vt, i, 0, "SwapIn", "X")
        # if 'Bc':
        #     ui = i if ubscvt is None else ubscvt.find_idx_ufwd(i)
        # elif 'F':
        #     ui = i
        ui = i if ubscvt is None else ubscvt.find_idx_ufwd(i)
        ev_x.inputs_add(left_vt, ui, -1, "SwapOut", "Y") # swap dependency
    else:
        raise ValueError
    
    if prefetch_offload:
        if i == 0:
            if left_vt is None: # 1st task
                ev_x.add_to(events, stream_events) 
            else: # prefetch at left vt
                if len(left_vt.ubatchszs) >= 2: # double buffer
                    ev_x.inputs_add(left_vt, len(left_vt.ubatchszs)-2, -1, "Compute", left_vt.type)
                else: # left_vt has only 1 ubatch
                    if left2_vt is not None:
                        ev_x.inputs_add(left2_vt, 0, 0, "Compute", "DEL")
                ev_x.add_to(events, stream_events)
                if ev_w is not None: # 'vDP'-['F','Bc']
                    ev_w.add_to(events, stream_events) 
                else: # vPP
                    pass
        elif i == 1:
            if left_vt is not None:
                ev_x.inputs_add(left_vt, 0, 0, "Compute", "DEL")
            ev_x.add_to(events, stream_events)
        else: # i >= 2
            ev_x.inputs_add(vt, i-2, -1, "Compute", vt.type)
            ev_x.add_to(events, stream_events)
    else: # no prefetch offload
        # if i == 0:
        #     assert ev_w is not None
        #     ev_x.inputs_add_ev(ev_w)
        # else: # i >= 1
        #     if ev_comp_out is not None:
        #         ev_x.inputs_add_ev(ev_comp_out) # if vDP, ev_out; if vPP, ev_comp
        #     else:
        #         ev_x.inputs_add(vt, i-1, -1, "Compute", vt.type) # if None dX
        if ev_w_comp_out is not None:
            ev_x.inputs_add_ev(ev_w_comp_out)
        else: # if None dX
            ev_x.inputs_add(vt, i-1, -1, "Compute", vt.type)
        ev_x.add_to(events, stream_events)
    
    return ev_x

def In_sX(vt, i, events, stream_events, left_vt, left2_vt, src_vt, src_spi, prefetch_offload, ev_w_comp_out):
    # assert 'Bn'
    ev_sx = Event(vt, i, 0, "SwapIn", "sX")
    ev_sx.inputs_add(src_vt, i, src_spi, "CPU", "MSGsX") # swap dependency
    
    if prefetch_offload:
        if i == 0:
            if left_vt is None: # 1st task
                ev_sx.add_to(events, stream_events)
            else: # prefetch at left vt
                if len(left_vt.ubatchszs) >= 2: # double buffer
                    ev_sx.inputs_add(left_vt, len(left_vt.ubatchszs)-2, -1, "Compute", left_vt.type)
                else: # left_vt has only 1 ubatch
                    if left2_vt is not None:
                        ev_sx.inputs_add(left2_vt, 0, 0, "Compute", "DEL")
                ev_sx.add_to(events, stream_events)
        elif i == 1:
            if left_vt is not None:
                ev_sx.inputs_add(left_vt, 0, 0, "Compute", "DEL")
            ev_sx.add_to(events, stream_events)
        else: # i >= 2
            ev_sx.inputs_add(vt, i-2, 0, "Compute", vt.type)
            ev_sx.add_to(events, stream_events)
    else: # no prefetch offload
        # if i == 0:
        #     assert ev_w is not None
        #     ev_sx.inputs_add_ev(ev_w)
        # else: # i >= 1
        #     if ev_comp_out is not None:
        #         ev_sx.inputs_add_ev(ev_comp_out) # if vDP, ev_out; if vPP, ev_comp
        #     else:
        #         ev_x.inputs_add(vt, i-1, -1, "Compute", vt.type) # if None dX
        if ev_w_comp_out is not None:
            ev_sx.inputs_add_ev(ev_w_comp_out)
        else: # if None dX
            ev_sx.inputs_add(vt, i-1, -1, "Compute", vt.type)
        ev_sx.add_to(events, stream_events)
    
    return ev_sx

def Compute_FWD(vt, i, events, stream_events, ev_w, ev_x, prefetch_offload):
    # 'F'
    spi = 0
    ev_fwd = Event(vt, i, spi, "Compute", "FWD") # subpack[0]
    if i == 0:
        ev_fwd.inputs_add_ev(ev_w)
    ev_fwd.inputs_add_ev(ev_x)
    ev_fwd.add_to(events, stream_events)
    # sub-packing by StashX
    for l, m in vt.Out['X'].items(): # layers in ascending order
        spi += 1
        if l == vt.layers[0]: # reuse subpack[0]
            spi -= 1; assert spi == 0
        else: # create a subpack[1+]
            ev_fwd = Event(vt, i, spi, "Compute", "FWD")
            ev_fwd.add_to(events, stream_events)
        if not prefetch_offload:
            ev_fwd.inputs_add(vt, i, spi, "SwapOut", "sX")
    
    return ev_fwd

def Out_sX(vt, i, u, events, stream_events, left_vt, ubscvt, ev_x, prefetch_offload):
    # 'F'
    ev_msg = None
    spi = 0
    # sub-packing by StashX
    for l, m in vt.Out['X'].items(): # layers in ascending order
        spi += 1
        if l == vt.layers[0]: # reuse subpack[0]
            spi -= 1; assert spi == 0
            ev_sx = Event(vt, i, spi, "SwapOut", "sX")
            ev_sx.inputs_add_ev(ev_x)
            if prefetch_offload:
                if i == 0:
                    if left_vt is not None:
                        ev_sx.inputs_add(left_vt, 0, 0, "Compute", "DEL")
                else:
                    ev_sx.inputs_add(vt, i-1, -1, "Compute", "FWD")
            ev_sx.add_to(events, stream_events)
        else: # create a subpack[1+]
            ev_sx = Event(vt, i, spi, "SwapOut", "sX")
            ev_sx.inputs_add(vt, i, spi-1, "Compute", "FWD")
            ev_sx.add_to(events, stream_events)
        # "MSGsX"
        assert m.medium == "MSG"
        indice_bwd = [i] if ubscvt is None else ubscvt.find_idx_ubwd(i)
        us_bwd = [u] if ubscvt is None else ubscvt.find_converted(i)
        for i_bwd, u_bwd in zip(indice_bwd, us_bwd):
            ev_msg = Event(vt, i_bwd, spi, "CPU", "MSGsX", ubs=u_bwd)            
            ev_msg.inputs_add_ev(ev_sx)
            ev_msg.add_to(events, stream_events)
    
    return ev_msg

def Out_Y(vt, i, u, events, stream_events, ubscvt, TASKS, ev_fwd):
    # 'F'
    if vt.Out["Y"][vt.layers[-1]].medium == "P2P": # P2PY
        ev_y = Event(vt, i, ev_fwd.spi, "P2POut", "Y")
        ev_y.register_peer(TASKS, i, 0, "P2PIn", "X")
        ev_y.inputs_add_ev(ev_fwd) 
        ev_y.add_to(events, stream_events)
    elif vt.Out["Y"][vt.layers[-1]].medium == "MSG": # last fwd MSG
        ev_y = Event(vt, i, ev_fwd.spi, "SwapOut", "Y")
        ev_y.inputs_add_ev(ev_fwd) 
        ev_y.add_to(events, stream_events)
        # "MSGY"
        indice_bwd = [i] if ubscvt is None else ubscvt.find_idx_ubwd(i)
        us_bwd = [u] if ubscvt is None else ubscvt.find_converted(i)
        for i_bwd, u_bwd in zip(indice_bwd, us_bwd):
            ev_msg = Event(vt, i_bwd, ev_fwd.spi, "CPU", "MSGY", ubs=u_bwd)            
            ev_msg.inputs_add_ev(ev_y)
            ev_msg.add_to(events, stream_events)
    elif vt.Out["Y"][vt.layers[-1]].medium == "SWP": # vDP only
        ev_y = Event(vt, i, ev_fwd.spi, "SwapOut", "Y")
        ev_y.inputs_add_ev(ev_fwd) 
        ev_y.add_to(events, stream_events)
    else:
        raise ValueError
    
    return ev_y

def In_dY(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ev_w, prefetch_after_rec, prefetch_offload):
    # assert 'Bn'
    if vt.In["dY"][vt.layers[-1]].medium == "P2P": # P2PdY
        ev_dy = Event(vt, i, 0, "P2PIn", "dY")
        ev_dy.register_peer(TASKS, i, 0, "P2POut", "dX")
    elif vt.In["dY"][vt.layers[-1]].medium == "SWP": # LocaldY of vDP only
        ev_dy = Event(vt, i, 0, "SwapIn", "dY")
        ev_dy.inputs_add(left_vt, i, 0, "SwapOut", "dX") # swap dependency
        assert left_vt is not None
    else:
        assert False
    
    if prefetch_offload:
        if i == 0:
            if left_vt is None: # 1st task
                ev_dy.add_to(events, stream_events)
            else: # prefetch at left vt
                if prefetch_after_rec or (left_vt.type == "BWD" and not left_vt.has_criterion): # prefetch_after_rec: True for vDP, False for vPP
                    ev_dy.inputs_add(left_vt, len(left_vt.ubatchszs)-1, -1, "Compute", "REC")
                else: # FWD or BWD(criterion)
                    if len(left_vt.ubatchszs) >= 2: # double buffer
                        ev_dy.inputs_add(left_vt, len(left_vt.ubatchszs)-2, -1, "Compute", left_vt.type)
                    else: # left_vt has only 1 ubatch
                        if left2_vt is not None:
                            ev_dy.inputs_add(left2_vt, 0, 0, "Compute", "DEL")
                ev_dy.add_to(events, stream_events)
                ev_w.add_to(events, stream_events)
        else: 
            ev_dy.inputs_add(vt, i-1, 0, "Compute", "REC")
            ev_dy.add_to(events, stream_events)
    else:
        ev_dy.inputs_add(vt, i, 0, "Compute", "REC")
        ev_dy.add_to(events, stream_events)
    
    return ev_dy

def Compute_BWD(vt, i, events, stream_events, ev_w, ev_x, ev_dy):
    ev_rec = Event(vt, i, 0, "Compute", "REC")
    if i == 0:
        ev_rec.inputs_add_ev(ev_w)
    ev_rec.inputs_add_ev(ev_x) # Bc: inputX, Bn: stashX
    ev_rec.add_to(events, stream_events)
    
    ev_bwd = Event(vt, i, 0, "Compute", "BWD")
    if ev_dy is not None: # "Bn"
        ev_bwd.inputs_add_ev(ev_dy)
    ev_bwd.add_to(events, stream_events)
    
    return ev_rec, ev_bwd
               
def Out_dX(vt, i, events, stream_events, TASKS, ev_bwd):
    if vt.layers[0] in vt.Out["dX"]:
        # assert not vt.has_data
        if vt.Out["dX"][vt.layers[0]].medium == "P2P": # P2PdX
            ev_dx = Event(vt, i, 0, "P2POut", "dX")
            ev_dx.register_peer(TASKS, i, -1, "P2PIn", "dY")
            ev_dx.inputs_add_ev(ev_bwd)
            ev_dx.add_to(events, stream_events)
        elif vt.Out["dX"][vt.layers[0]].medium == "SWP": # vDP only
            ev_dx = Event(vt, i, 0, "SwapOut", "dX")
            ev_dx.inputs_add_ev(ev_bwd)
            ev_dx.add_to(events, stream_events)
        else:
            assert False
    else:
        ev_dx = None
    
    return ev_dx

def Compute_ARD(vt, events, stream_events, num_gpus, prefetch_offload, ev_dx):
    # assert 'vDP'-['Bc','Bn']
    if num_gpus > 1:
        ev_ard = Event(vt, 0, 0, "Compute", "ARD")
        ev_ard.add_to(events, stream_events)
        if not prefetch_offload and ev_dx is not None:
            ev_ard.inputs_add_ev(ev_dx)
    else:
        ev_ard = None
    
    return ev_ard
    
def Out_dWB(vt, events, stream_events, ev_bwd, ev_ard, prefetch_offload, ev_dx):
    # assert ["Bc",'Bn']
    ev_dw = Event(vt, 0, 0, "SwapOut", "dWB")
    ev_dw.inputs_add_ev(ev_bwd)
    if ev_ard is not None: # 'vDP' and num_gpus > 1
        ev_dw.inputs_add_ev(ev_ard)
    elif not prefetch_offload and ev_dx is not None: # 'vDP' and num_gpus == 1
        ev_dw.inputs_add_ev(ev_dx)
    ev_dw.add_to(events, stream_events)
    
    return ev_dw

def Compute_DEL(vt, events, stream_events, ev_dw, prefetch_offload, ev_y):
    ev_del = Event(vt, 0, 0, "Compute", "DEL")
    if ev_dw is not None: # 'Bc','Bn'
        ev_del.inputs_add_ev(ev_dw)
    elif not prefetch_offload and ev_y is not None: # 'F' 
        ev_del.inputs_add_ev(ev_y)
    ev_del.add_to(events, stream_events)
    
    return ev_del

def CPU_Update(vt, events, stream_events, left_vt):
    assert left_vt is not None and left_vt.layers == vt.layers
    ev = Event(vt, 0, 0, "CPU", "Update")
    ev.inputs_add(left_vt, 0, 0, "SwapOut", "dWB")
    ev.add_to(events, stream_events)
    
    return ev

class Dispatcher(object):
    def __init__(self, rank_stream_events):
        self.event_queues = deque([]) # non-empty event queues across all ranks
        for stream_events in rank_stream_events.values(): # can be empty rank
            for eq in stream_events.values():
                assert len(eq) != 0
                self.event_queues.append(eq)
        # for rank, stream_events in rank_stream_events.items(): 
        #     for k in [k for k, v in stream_events.items() if v == []]:
        #         del stream_events[k]
        # for k in [k for k, v in rank_stream_events.items() if not v]:
        #     del rank_stream_events[k]
        self.num_streams = len(self.event_queues) # statisics

    def _check_inputs(self, ev):
        if ev.inputs == []:
            return True
        else:
            return min([inev.is_done for inev in ev.inputs]) 
            # return min([self.events[id].is_done for id in ev.inputs]) 
            # Any False -> False; All True -> True
            
    def dispatch(self):
        if len(self.event_queues) == 0:
            return "done" # all events dispatched
        max_step = len(self.event_queues)*2
        for _ in range(max_step):
            # try a non-empty queue
            events = self.event_queues.popleft() # round-robin abitration
            if self._check_inputs(events[0]): # event found
                ev = events.popleft()
                if len(events) != 0:
                    self.event_queues.append(events)
                return ev # dispatch a single event
            self.event_queues.append(events)
        # deadlock 
        return [events[0] for events in self.event_queues] # dealock events

class Executor(object):
    def __init__(self, args, non_empty_gpus, CONFIGS, TASKS, rank_stream_events):
        self.prof = args.prof
        self.bw_swap = args.bw_swap
        self.bw_p2p = args.bw_p2p
        self.bw_msg = args.bw_msg
        self.time_del = args.time_del
        self.mode = CONFIGS["mode"]
        self.N = non_empty_gpus # CONFIGS["N"]
        self.R = CONFIGS["R"]
        self.TASKS = TASKS; assert TASKS[-1].idx == len(TASKS)-1, "TASKS is global"
        self.use_random = args.use_random
        if self.use_random:
            random.seed(args.seed)
            np.random.seed(args.seed)
        # { rank : { stream : last Event's end time } or {} }
        self.rank_stream_endtime = ODict()
        for r, stream_events in rank_stream_events.items(): # can be empty rank
            self.rank_stream_endtime[r] = ODict()
            for s in stream_events.keys():
                self.rank_stream_endtime[r][s] = 0.0 # sec
        # { rank : compute time accumulated }
        self.rank_compute = ODict()
        for r in rank_stream_events.keys(): # can be empty rank
            self.rank_compute[r] = 0.0 # sec
        # count executed events
        self.cnt = 0 
    
    def _duration(self, ev):
        ubs = ev.ubs
        l_start, l_end = ev.layers[0], ev.layers[-1] # subpack layers
        if ev.kind in ["FWD","REC"]:
            return time_of_pack(self.prof, "FWD", ubs, l_start, l_end, interp_ubatchsize=True) # sec
        elif ev.kind == "BWD":
            return time_of_pack(self.prof, "BWD", ubs, l_start, l_end,     interp_ubatchsize=True) - \
            time_of_pack(self.prof, "FWD", ubs, l_start, l_end,     interp_ubatchsize=True)
        elif ev.kind == "Update":
            return time_of_pack(self.prof, 'UPD', None, l_start, l_end, offload_optim=True)
        elif ev.kind == "DEL":
            return self.time_del # sec # empirical value
        elif ev.kind == "ARD":
            W = model_size_of_pack(self.prof, 'W', l_start, l_end) # bytes
            B = model_size_of_pack(self.prof, 'B', l_start, l_end) # bytes
            return 2.*(self.N-1)/self.N*(W+B)/self.bw_p2p # sec
        elif ev.kind in ["WB","dWB"]:
            W = model_size_of_pack(self.prof, 'W', l_start, l_end) # bytes
            B = model_size_of_pack(self.prof, 'B', l_start, l_end) # bytes
            return float(W+B)/(self.bw_swap/self.N) # FWD-SwapIn/BWD-SwapIn/BWD-SwapOut in vDP/vPP
        elif ev.kind in ["sX","X","dX"]:
            X = self.prof["XMETA"].get_bytes(ubs, l_start, interp=True)
            if ev.stream.startswith("Swap"):
                return float(X)/(self.bw_swap/self.N) # FWD/BWD SwapIn/SwapOut in vDP/vPP
            elif ev.stream.startswith("P2P"):
                return float(X)/(self.bw_p2p) # FWD-P2PIn/BWD-P2POut in vPP
            else:
                raise NotImplementedError 
        elif ev.kind in ["Y","dY"]:
            Y = self.prof["XMETA"].get_bytes(ubs, l_end+1, interp=True) \
                if l_end+1 < self.R else 0. 
            if ev.stream.startswith("Swap"):
                return float(Y)/(self.bw_swap/self.N) # FWD-SwapOut/BWD-SwapIn in vDP
            elif ev.stream.startswith("P2P"):
                return float(Y)/(self.bw_p2p) # FWD-P2PIn/BWD-P2POut in vPP
            else:
                raise NotImplementedError 
        elif ev.kind.startswith("MSG"): # ["MSGsX","MSGdX","MSGY"]
            kind = ev.kind.replace("MSG","")
            if kind == "sX":
                if self.TASKS[ev.vt.Out["X"][l_start].idx].rank == ev.vt.rank:
                    M = 0. # self send
                else:
                    M = self.prof["XMETA"].get_bytes(ubs, l_start, interp=True)
            elif kind == "dX":
                if self.TASKS[ev.vt.Out["dX"][l_start].idx].rank == ev.vt.rank:
                    M = 0. # self send
                else:
                    M = self.prof["XMETA"].get_bytes(ubs, l_start, interp=True)
            elif kind in "Y":
                if self.TASKS[ev.vt.Out["Y"][l_end].idx].rank == ev.vt.rank:
                    M = 0. # self send
                else:
                    M = self.prof["XMETA"].get_bytes(ubs, l_end+1, interp=True) \
                        if l_end+1 < self.R else 0.
            else:
                raise NotImplementedError
            return float(M)/(self.bw_msg/self.N)*2. # CPU memory is half-duplex
        else:
            raise NotImplementedError 
            
    def execute(self, ev):
        ev.begin = max([inev.end for inev in ev.inputs] + 
                       [self.rank_stream_endtime[ev.vt.rank][ev.stream]])
        if self.use_random:
            ev.dur = random.uniform(0, 1.0) # sec
            if ev.stream.startswith("P2P"):
                ev.dur = 0.5
        else:
            ev.dur = self._duration(ev) # sec
        ev.is_done = True
        self.rank_stream_endtime[ev.vt.rank][ev.stream] = ev.end
        if ev.kind in ["FWD","REC","BWD"]:
            self.rank_compute[ev.vt.rank] += ev.dur
        self.cnt += 1
    
    def end(self):
        ### end time
        self.per_rank_endtime = []
        for stream_endtime in self.rank_stream_endtime.values(): # can be empty rank
            if stream_endtime:
                et = max([endtime for endtime in stream_endtime.values()])
            else: # empty rank
                et = 1.E-10 # zero
            self.per_rank_endtime.append(et)
        self.global_endtime = max(self.per_rank_endtime)
        ### end idle ratio
        self.per_rank_endidle = [ (self.global_endtime - et) / self.global_endtime 
                                    for et in self.per_rank_endtime ]
        self.max_endidle = max(self.per_rank_endidle)
        self.avg_endidle = sum(self.per_rank_endidle)/len(self.per_rank_endidle)
        ### compute ratio
        num_ranks = len(self.rank_compute)
        self.avg_compute_to_globaltime = sum([ ct/self.global_endtime 
                                            for ct in self.rank_compute.values() ]) \
                                            / num_ranks
        self.avg_compute_to_ranktime = sum([ct/et 
                                            for ct, et in zip(self.rank_compute.values(), self.per_rank_endtime)]) \
                                            / num_ranks
    
def print_events(events):
    print("------- Event: ID, Inputs, Done, Name -------")
    for id, ev in events.items(): 
        assert id == ev.id
        print("%s, '%s'" % (ev, ev.name))
    print()

def print_rank_stream_events(rank_stream_events):
    for rank, per_stream_events in rank_stream_events.items(): 
        print("------- Rank %d's Stream : [Events] -------" % rank)
        for stream, events in per_stream_events.items():
            print("%s: [%s]" % (stream, ", ".join(ev.id for ev in events) ))
    print()

def debug(non_empty_gpus, events, rank_stream_events):
    non_empty_rank = 0
    for stream_events in rank_stream_events.values():
        if stream_events:
            non_empty_rank += 1
    assert non_empty_rank == non_empty_gpus
    print("[DEBUG] non_empty_gpus check: passed")
    for stream_events in rank_stream_events.values(): 
        for eq in stream_events.values():
            for e in eq:
                assert id(events[e.id]) == id(e), "%s vs %s"%(events[e.id], e)
    print("[DEBUG] same reference test: passed")
    for ev in events.values(): 
        for inev in ev.inputs:
            assert isinstance(inev, Event), "!! {} is not Event".format(inev)
    for stream_events in rank_stream_events.values():
        for eq in stream_events.values():
            for e in eq:
                for inev in e.inputs:
                    assert isinstance(inev, Event), "!! {} is not Event".format(inev)
    print("[DEBUG] inputs are events: passed")

class CofTask(object):
    """ The memory cost of a task.
        Assumption:
        0) math equation modeled cost
        1) optimizer on CPU
        2) equal ubatch size in a group, FWD or BWD
        3) vPP always use P2P for input and output
        4) ignore T
        base version:
        5) fetch W
        6) prefetch X with double buffering
        7) no prefetch X across Tasks
        8) offload Y with double buffering
        9) offload X with double buffering
    """
    def __init__(self, prof, mode, num_layers):
        self.prof = prof
        self.mode = mode
        self.R = num_layers
                
    def __call__(self, vt):
        l_start, l_end = vt.layers[0], vt.layers[-1]
        if vt.type == 'UPD':
            return 0.
            # Ctask = memory_of_pack(self.prof, 'UPD', None, l_start, l_end, offload_optim=self.offload_optim) # bytes
            # return float(Ctask)
        
        assert vt.type in ['FWD','BWD']
        ubs = vt.ubatchszs[0]
        # size
        InputX = self.prof["XMETA"].get_bytes(ubs, l_start, interp=True)
        dX = InputX if l_start != 0 else 0. 
        W = model_size_of_pack(self.prof, 'W', l_start, l_end) # bytes
        B = model_size_of_pack(self.prof, 'B', l_start, l_end) # bytes
        Y = self.prof["XMETA"].get_bytes(ubs, l_end+1, interp=True) \
            if l_end != self.R-1 else 0. 
        dY = Y
        if vt.type == 'FWD':
            StashX = sum([ self.prof["XMETA"].get_bytes(ubs, l, interp=True) 
                            for l in vt.Out['X'].keys() ])       
        else:
            StashX = InputX # critierion or non-criterion
        # memory
        Ccompute = memory_of_pack(self.prof, vt.type, ubs, l_start, l_end, interp_ubatchsize=True) # bytes
        # if vt.type == 'FWD' and vt.has_data:
        #     Ccompute -= InputX
        
        Ctask = 0.
        if self.mode == 'vPP' and vt.type == 'FWD':
            Ctask += InputX
            Ctask += Ccompute
            Ctask += Y
            Ctask += StashX    
        elif self.mode == 'vPP' and vt.type == 'BWD':
            Ctask += StashX*2 # include grad
            Ctask += dY
            Ctask += Ccompute
            Ctask += dX
        elif self.mode == 'vDP' and vt.type == 'FWD':
            Ctask += InputX
            Ctask += Ccompute
            Ctask += Y
            Ctask += StashX
        elif self.mode == 'vDP' and vt.type == 'BWD': 
            Ctask += StashX*2 # include grad
            Ctask += dY
            Ctask += Ccompute
            Ctask += dX
        else:
            raise NotImplementedError
        return Ctask # bytes
