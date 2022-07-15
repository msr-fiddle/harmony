# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division, print_function
import os
import argparse
import json
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict as ODict
import sys
import pickle
from copy import deepcopy

def _piecewise_linear_interplote(x_points, y_points, x):
    assert x_points[0] <= x and x <= x_points[-1]
    return np.interp(x, x_points, y_points) # https://numpy.org/doc/stable/reference/generated/numpy.interp.html # float, np.float64

class Time(object): 
    # FWD/BWD(include recompute): { 'FWD'/'BWD' : { ubatchsize: { vlayer_id: { 'device': 0.xxx sec } } } }
    # or
    # UPD                       : { 'UPD'       :               { vlayer_id: { 'device': 0.xxx sec } }   }
    def __init__(self, fwd_ubatchsize_range, bwd_ubatchsize_range, num_vlayers): 
        if isinstance(fwd_ubatchsize_range, list) and isinstance(bwd_ubatchsize_range, list):
            self.stats = ODict({ 'FWD':ODict(), 'BWD':ODict() })
            # init FWD/BWD
            for ubatchsize in range(*fwd_ubatchsize_range): 
                self.stats['FWD'][ubatchsize] = ODict()
                for vlayer_id in range(num_vlayers):
                    self.stats['FWD'][ubatchsize][vlayer_id] = ODict({'GPU':0.0, 'CPU':0.0})
            for ubatchsize in range(*bwd_ubatchsize_range): 
                self.stats['BWD'][ubatchsize] = ODict()
                for vlayer_id in range(num_vlayers):
                    self.stats['BWD'][ubatchsize][vlayer_id] = ODict({'GPU':0.0, 'CPU':0.0})
        elif fwd_ubatchsize_range is None and bwd_ubatchsize_range is None:
            self.stats = ODict({ 'UPD':ODict() })
            # init UPD
            for vlayer_id in range(num_vlayers):
                self.stats['UPD'][vlayer_id] = ODict({'GPU':0.0, 'CPU':0.0})
        
    def __repr__(self, which='Time (shown in milli-sec)'):
        def time2show(t):
            assert isinstance(t,float)
            return "-" if t == 0.0 else "%.3f" % (t*1000) # milli-sec    
        repre  = "\n------------- %s ----------------\n"%(which)
        repre += "type: ubatchsize/vlayer_id: GPU time, CPU time\n"
        for type, ubatchsize_time in self.stats.items():
            if type in ['FWD','BWD']:
                for ubatchsize, vlayer_time in ubatchsize_time.items():
                    for vlayer_id, device_time in vlayer_time.items():
                        repre += "%s: %d/%d: %s, %s\n" % (
                                type, ubatchsize,vlayer_id, 
                                time2show(device_time['GPU']), 
                                time2show(device_time['CPU']))
            elif type in ['UPD']:
                vlayer_time = ubatchsize_time
                for vlayer_id, device_time in vlayer_time.items():
                    repre += "%s: -/%d: %s, %s\n" % (
                            type, vlayer_id, 
                            time2show(device_time['GPU']), 
                            time2show(device_time['CPU']))
        repre += "-----------------------------------------------------------"
        return repre

    def reset(self, type, ubatchsize, vlayer_id, device, time=0.0):
        assert isinstance(time,float)
        if type in ["FWD","BWD"]:
            self.stats[type][ubatchsize][vlayer_id][device] = time
        elif type in ["UPD"]:
            self.stats[type][vlayer_id][device] = time
        else:
            raise ValueError("unknown type={}".format(type))

    def add(self, type, ubatchsize, vlayer_id, device, time):
        assert isinstance(time,float)
        if type in ["FWD","BWD"]:
            self.stats[type][ubatchsize][vlayer_id][device] += time
        elif type in ["UPD"]:
            self.stats[type][vlayer_id][device] += time
        else:
            raise ValueError("unknown type={}".format(type))
    
    def avg_trials(self, num_trials):
        for type, ubatchsize_time in self.stats.items():
            if type in ['FWD','BWD']:
                for ubatchsize, vlayer_time in ubatchsize_time.items():
                    for vlayer_id, device_time in vlayer_time.items():
                        self.stats[type][ubatchsize][vlayer_id]['GPU'] /= float(num_trials)
                        self.stats[type][ubatchsize][vlayer_id]['CPU'] /= float(num_trials)
            elif type in ['UPD']:
                vlayer_time = ubatchsize_time
                for vlayer_id, device_time in vlayer_time.items():
                    self.stats[type][vlayer_id]['GPU'] /= float(num_trials)
                    self.stats[type][vlayer_id]['CPU'] /= float(num_trials)
        print("Time averaged over %d trials"%num_trials)

    def get(self, type, ubatchsize, vlayer_id, device="GPU", interp=False):
        if type in ["FWD","BWD"]:
            if not interp or ubatchsize in self.stats[type]:
                return self.stats[type][ubatchsize][vlayer_id][device]
            else: # linear interpolate unknown ubatchsize
                xp = list(self.stats[type].keys())
                yp = [self.stats[type][ubsize][vlayer_id][device] for ubsize in xp]
                return float(_piecewise_linear_interplote(xp, yp, ubatchsize))
        elif type in ["UPD"]:
            return self.stats[type][vlayer_id][device]
        else:
            raise ValueError("unknown type={}".format(type))

    def get_ubatchsizes(self, type):
        assert type in ['FWD','BWD']
        return list(self.stats[type].keys())
    
    def get_vlayers(self):
        for type, ubatchsize_time in self.stats.items():
            if type in ['FWD','BWD']:
                for ubatchsize, vlayer_time in ubatchsize_time.items():
                    return list(vlayer_time.keys())
            elif type in ['UPD']:
                vlayer_time = ubatchsize_time
                return list(vlayer_time.keys())

    def fit_poly_over_ubatchsize(self, degree=1): # { 'FWD'/'BWD' : { ubatchsize: { vlayer_id: { 'device': 0.xxx sec } } } }
        if hasattr(self, 'already_fit'):
            return
        assert 'FWD' in self.stats and 'BWD' in self.stats, "only for FWD and BWD time"
        #
        def fit_poly(x, y, degree=degree):
            z = np.polyfit(x, y, degree) # calculate polynomial
            f = np.poly1d(z) # get fit function
            x_new = np.array(range(x[0], x[-1]+1, 1)) # calculate new x's and y's
            y_new = f(x_new)
            return x_new, y_new
        # fit time for all
        tmp = ODict({ 'FWD':ODict(), 'BWD':ODict() }) # { 'FWD'/'BWD' : { vlayer_id: [x, y1, y2] } } 
        for type in ['FWD','BWD']:
            tmp[type] = ODict()
            for vlayer_id in self.get_vlayers():
                # fit time
                x = self.get_ubatchsizes(type)
                y1 = [self.stats[type][ubatchsize][vlayer_id]['GPU'] for ubatchsize in x]
                y2 = [self.stats[type][ubatchsize][vlayer_id]['CPU'] for ubatchsize in x]
                x_new, y1_new = fit_poly(x, y1, degree)
                _, y2_new = fit_poly(x, y2, degree)
                # save fit time
                tmp[type][vlayer_id] = [x_new, y1_new, y2_new]
        # creat new stats
        self.stats_fit = ODict({ 'FWD':ODict(), 'BWD':ODict() })
        for type in self.stats_fit.keys():
            for ubatchsize in tmp[type][0][0]: # traverse new microbatch size
                self.stats_fit[type][ubatchsize] = ODict()
                for vlayer_id in self.get_vlayers():
                    self.stats_fit[type][ubatchsize][vlayer_id] = ODict({'GPU': 0.0, 'CPU': 0.0})
        # save tmp to new stats
        for type, vlayer_xy in tmp.items():
            for vlayer_id, (x,y1,y2) in vlayer_xy.items():
                for ubatchsize, gpu_time, cpu_time in zip(x,y1,y2):
                    self.stats_fit[type][ubatchsize][vlayer_id] = ODict({'GPU': gpu_time, 'CPU': cpu_time})
        print("[INFO] time's stats got fit over ubatchsize with degree={}".format(degree))
        # flag for poly fit
        self.already_fit = True

    def get_ubatchsizes_fit(self, type):
        assert type in ['FWD','BWD']
        self.fit_poly_over_ubatchsize()
        return list(self.stats_fit[type].keys())
    
    def get_fit(self, type, ubatchsize, vlayer_id, device="GPU"):
        assert type in ["FWD","BWD"]
        self.fit_poly_over_ubatchsize()
        return self.stats_fit[type][ubatchsize][vlayer_id][device]
      
class Memory(object): 
    # FWD/BWD(include recompute): { 'FWD'/'BWD' : { ubatchsize: { vlayer_id: xxx bytes } } }
    # UPD                       : { 'UPD'       :               { vlayer_id: xxx bytes }   }
    def __init__(self, fwd_ubatchsize_range, bwd_ubatchsize_range, num_vlayers): 
        if isinstance(fwd_ubatchsize_range, list) and isinstance(bwd_ubatchsize_range, list):
            self.stats = ODict({ 'FWD':ODict(), 'BWD':ODict() })
            # init FWD/BWD
            for ubatchsize in range(*fwd_ubatchsize_range): 
                self.stats['FWD'][ubatchsize] = ODict()
                for vlayer_id in range(num_vlayers):
                    self.stats['FWD'][ubatchsize][vlayer_id] = None
            for ubatchsize in range(*bwd_ubatchsize_range): 
                self.stats['BWD'][ubatchsize] = ODict()
                for vlayer_id in range(num_vlayers):
                    self.stats['BWD'][ubatchsize][vlayer_id] = None
        elif fwd_ubatchsize_range is None and bwd_ubatchsize_range is None:
            self.stats = ODict({ 'UPD':ODict() })
            # init UPD
            for vlayer_id in range(num_vlayers):
                self.stats['UPD'][vlayer_id] = None

    def __repr__(self, which='Memory (shown in MB)'):
        def memory2show(m): # byte
            assert m is None or isinstance(m, int)
            return "-" if m is None else "%d"%( int(float(m)/1024/1024) ) # MB
        repre  = "\n------------- %s ----------------\n"%(which)
        repre += "type: ubatchsize/vlayer_id: memory usage\n"
        for type, ubatchsize_mem in self.stats.items():
            if type in ['FWD','BWD']:
                for ubatchsize, vlayer_mem in ubatchsize_mem.items():
                    for vlayer_id, mem in vlayer_mem.items():
                        repre += "%s: %d/%d: %s\n" % (
                                type, ubatchsize,vlayer_id, memory2show(mem))
            elif type in ['UPD']:
                vlayer_mem = ubatchsize_mem
                for vlayer_id, mem in vlayer_mem.items():
                    repre += "%s: -/%d: %s\n" % (
                              type, vlayer_id, memory2show(mem))
        repre += "-----------------------------------------------------------"
        return repre

    def set(self, type, ubatchsize, vlayer_id, mem):
        assert isinstance(mem,int)
        if type in ["FWD","BWD"]:
            self.stats[type][ubatchsize][vlayer_id] = mem
        elif type in ["UPD"]:
            self.stats[type][vlayer_id] = mem
        else:
            raise ValueError("unknown type={}".format(type))

    def get(self, type, ubatchsize, vlayer_id, interp=False):
        if type in ["FWD","BWD"]:
            if not interp or ubatchsize in self.stats[type]:
                return self.stats[type][ubatchsize][vlayer_id]
            else: # linear interpolate unknown ubatchsize
                xp = list(self.stats[type].keys())
                yp = [self.stats[type][ubsize][vlayer_id] for ubsize in xp]
                return int(_piecewise_linear_interplote(xp, yp, ubatchsize)) 
        elif type in ["UPD"]:
            return self.stats[type][vlayer_id]
        else:
            raise ValueError("unknown type={}".format(type))

    def get_ubatchsizes(self, type):
        assert type in ['FWD','BWD']
        return list(self.stats[type].keys())

class ConstMeta():
    def __init__(self, name, const): 
        self.name = name # input0 or out1
        self.const = const # int, float
    
    def __repr__(self):
        return "<'{}':{}>".format(self.name,self.const)
    
    @property
    def bytes(self):
        return 0
    
    @property
    def numel(self):
        return 1
    
    # def __eq__(self, other): # For Test
    #     # calling dir on the object gives you back all the attributes of that object (sorted): ['__class__', '__delattr__', ..., 'bar', 'foo', 'func'] 
    #     # https://stackoverflow.com/questions/11637293/iterate-over-object-attributes-in-python
    #     if dir(self) != dir(other): 
    #         return False
    #     for attr_name in dir(self): # traverse all attributes, compare non-callables
    #         self_attr, other_attr = getattr(self, attr_name), getattr(other, attr_name)
    #         if not callable(self_attr) or not callable(other_attr):
    #             if self_attr != other_attr:
    #                 return False
    #     # compare __str__() results
    #     if self.__repr__() != other.__repr__():
    #         return False
    #     return True # both non-callable and __str__ equals

class TensorMeta():
    def __init__(self, name, shape, dtype=torch.FloatTensor, is_ubatch=True): 
        self.name = name # input0 or out1
        self.shape = shape # torch.Size([]) or torch.Size([1]) or torch.Size([1, 3])
        self.dtype = dtype # torch.int64 or torch.float32 or torch.uint8
        self.is_ubatch = is_ubatch # True: shape includes ubatchsize, False: a single sample
    
    def __repr__(self, show_size=False):
        if show_size:
            return "<'{}':{}>".format(self.name,self.bytes)
        else:
            return "<'{}':{},{}>".format(self.name,tuple(self.shape),self.dtype)
    
    def add_ubatch(self, ubatchsize):
        if not self.is_ubatch: # e.g. un-batched data sample
            self.shape = tuple([ubatchsize]+list(self.shape)) # insert ubatchsize to the left most dimension
            self.is_ubatch = True
    
    def set_ubatch(self, ubatchsize):
        if not self.is_ubatch: # e.g. un-batched data sample
            self.add_ubatch(ubatchsize)
        else:
            assert len(self.shape) > 0, "set ubatch of scalar tensor"
            shape = list(self.shape)
            shape[0] = ubatchsize # set ubatchsize to the left most dimension
            self.shape = tuple(shape) 
    
    @property
    def bytes(self):
        if self.dtype in [torch.float32]:
            elem_size = 4
        elif self.dtype in [torch.int64]:
            elem_size = 8
        elif self.dtype in [torch.uint8, torch.int8]:
            elem_size = 1
        else:
            raise ValueError("unknown dtype={}".format(self.dtype))
        return int( np.prod(list(self.shape)) * elem_size )
        # if scalar tensor, np.prod(list(scalar_tensor.shape)) = np.prod([]) => 1.0 
        # else non-scalar tensor, np.prod(list(scalar_tensor.shape)) = np.prod([1] or [1,2,3]) => 1 or 6
    
    @property
    def numel(self):
        return int(np.prod(list(self.shape)))
    
    # def __eq__(self, other): # For Test
    #     # calling dir on the object gives you back all the attributes of that object (sorted): ['__class__', '__delattr__', ..., 'bar', 'foo', 'func'] 
    #     # https://stackoverflow.com/questions/11637293/iterate-over-object-attributes-in-python
    #     if dir(self) != dir(other): 
    #         return False
    #     for attr_name in dir(self): # traverse all attributes, compare non-callables
    #         self_attr, other_attr = getattr(self, attr_name), getattr(other, attr_name)
    #         if not callable(self_attr) or not callable(other_attr):
    #             if self_attr != other_attr:
    #                 return False
    #     # compare __str__() results
    #     if self.__repr__() != other.__repr__():
    #         return False
    #     return True # both non-callable and __str__ equals

class XMeta(object): 
    # { ubatchsize: { vlayer_id: { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] } } } }
    def __init__(self, ubatchsizes, num_vlayers):
        self.stats = ODict() 
        for ubatchsize in range(*ubatchsizes): 
            self.stats[ubatchsize] = ODict()
            for vlayer_id in range(num_vlayers):
                self.stats[ubatchsize][vlayer_id] = None

    def __repr__(self, which='X'):
        repre  = "\n----------------------- %s meta-data -----------------------\n"%(which)
        repre += "ubatchsize/vlayer_id: TensorMeta, ConstMeta, [TensorMeta,TensorMeta] \n"
        for ubatchsize, vlayer_metas in self.stats.items():
            for vlayer_id, name_metas in vlayer_metas.items():
                if name_metas is None: 
                    ss = "-"
                else:
                    ss = []
                    for name, meta in name_metas.items():
                        if isinstance(meta, (TensorMeta,ConstMeta)):
                            assert name == meta.name
                            ss.append( str(meta) )
                        elif isinstance(meta, list):
                            for m in meta:
                                assert name == m.name
                            ss.append(  "[%s]" % ", ".join(str(m) for m in meta) )
                        else:
                            raise ValueError("unknown meta={}".format(meta))
                    ss = ", ".join(ss)
                repre += "%d/%d: %s\n" % (ubatchsize, vlayer_id, ss)
        repre += "-----------------------------------------------------------"
        return repre

    def init_data_meta_on_vlayer0(self, data_names, data_tensors):
        for ubatchsize in self.stats.keys():
            self.set(ubatchsize, 0, data_names, data_tensors, is_ubatch=False)

    def set(self, ubatchsize, vlayer_id, X_names, X_tensors, is_ubatch=True):
        assert isinstance(X_names, list)
        if type(X_tensors) in [torch.Tensor, Variable, int]:
            X_tensors_list = [X_tensors]
        elif type(X_tensors) in [tuple,list]:
            X_tensors_list = list(X_tensors)
        else:
            raise ValueError("unknown X={}".format(X_tensors))
        
        self.stats[ubatchsize][vlayer_id] = ODict()
        if len(X_names) == 1 and len(X_tensors_list) > 1:
            self.stats[ubatchsize][vlayer_id][X_names[0]] = [TensorMeta(X_names[0], t.shape, t.dtype, is_ubatch=True) for t in X_tensors_list]
        else:
            for name, tensor in zip(X_names, X_tensors_list):
                if type(tensor) in [torch.Tensor, Variable]:
                    self.stats[ubatchsize][vlayer_id][name] = TensorMeta(name, tensor.shape, tensor.dtype, is_ubatch)
                elif type(tensor) in [int]:
                    self.stats[ubatchsize][vlayer_id][name] = ConstMeta(name, tensor)
                # elif type(tensor) in [list]:
                #     self.stats[ubatchsize][vlayer_id][name] = [TensorMeta(name, t.shape, t.dtype, is_ubatch) for t in tensor]
                else:
                    raise ValueError("set error at {}: tensor={}".format(type(tensor), tensor))
        
    def get(self, ubatchsize, vlayer_id, interp=True):
        if not interp or ubatchsize in self.stats:
            return self.stats[ubatchsize][vlayer_id]
        else: # interoplate unknown ubatchsize by replacement
            # { ubatchsize: { vlayer_id: { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] } } } }
            u, vlayer_metas = next(iter( self.stats.items() ))
            name_metas = deepcopy(vlayer_metas[vlayer_id])
            for name, meta in name_metas.items(): # named metas
                if isinstance(meta, TensorMeta):
                    # named_metas[name] = TensorMeta(name, meta.shape, meta.dtype, meta.is_ubatch)
                    meta.set_ubatch(ubatchsize)
                elif isinstance(meta, ConstMeta):
                    assert isinstance(meta.const, int) and meta.const == u, "unsupported const type"
                    meta.const = ubatchsize
                elif isinstance(meta, list): # output tuple of bert pretrainheader
                    # named_metas[name] = [ TensorMeta(name, m.shape, m.dtype, m.is_ubatch) for m in meta ]
                    [ m.set_ubatch(ubatchsize) for m in meta ]
            return name_metas
    
    def get_names(self, ubatchsize, vlayer_id):
        if ubatchsize in self.stats:
            return list(self.stats[ubatchsize][vlayer_id].keys())
        else:
            _, vlayer_metas = next(iter( self.stats.items() ))
            return list(vlayer_metas[vlayer_id].keys())

    def _total_bytes(self, named_metas):
        total_bytes = 0
        for metas in named_metas.values(): # { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
            if isinstance(metas, (TensorMeta, ConstMeta)):
                total_bytes += metas.bytes
            elif isinstance(metas, list):
                total_bytes += sum([meta.bytes for meta in metas])
            else:
                raise ValueError("type error: {}".format(type(metas)))
        return int(total_bytes)

    def get_bytes(self, ubatchsize, vlayer_id, interp=True):
        if not interp or ubatchsize in self.stats:
            return self._total_bytes(self.stats[ubatchsize][vlayer_id])
        else: # interpolate unknown ubatchsize by replacement
            named_metas = self.get(ubatchsize, vlayer_id, interp=True)
            return self._total_bytes(named_metas)
        # else: # linear interpolation
        #         xp = list(self.stats.keys())
        #         yp = [self.get_bytes(ubsize, vlayer_id, interp=False) for ubsize in xp]
        #         assert xp[0] <= ubatchsize and ubatchsize <= xp[-1]
        #         # print("[INFO] XMeta.get({},{}) is interpolated.".format(ubatchsize, vlayer_id))
        #         return int(np.interp(ubatchsize, xp, yp))

    def get_ubatchsizes(self):
        return list(self.stats.keys())
    
    def get_vlayer_ids(self):
        _, vlayer_metas = next(iter( self.stats.items() ))
        return list(vlayer_metas.keys())
    
    # def fill_dummy_meta(self): # For TEST
    #     for ubatchsize, vlayer_metas in self.stats.items():
    #         for vlayer_id, name_metas in vlayer_metas.items():
    #             assert name_metas == None
    #             self.stats[ubatchsize][vlayer_id] = ODict()
    #             # { name1: tensor, name2: const, name3: [tensors] }
    #             name1 = "X%d_1"%vlayer_id
    #             self.stats[ubatchsize][vlayer_id][name1] = TensorMeta(name1, torch.Size([ubatchsize,vlayer_id+1]), torch.int64, True)
    #             name2 = "X%d_2"%vlayer_id
    #             self.stats[ubatchsize][vlayer_id][name2] = ConstMeta(name2, int(ubatchsize))
    #             name3 = "X%d_3"%vlayer_id
    #             self.stats[ubatchsize][vlayer_id][name3] = [ TensorMeta(name3, torch.Size([ubatchsize,vlayer_id+1]), torch.float32, True) for _ in range(2) ]
    
    # def fill_dummy_meta_enlarge(self, enlarge=10000000): # For TEST
    #     for ubatchsize, vlayer_metas in self.stats.items():
    #         for vlayer_id, name_metas in vlayer_metas.items():
    #             assert name_metas == None
    #             self.stats[ubatchsize][vlayer_id] = ODict()
    #             # { name1: tensor, name2: const, name3: [tensors] }
    #             name1 = "X%d_1"%vlayer_id
    #             self.stats[ubatchsize][vlayer_id][name1] = TensorMeta(name1, torch.Size([ubatchsize, (vlayer_id+1)*enlarge]), torch.int64, True)
    #             name2 = "X%d_2"%vlayer_id
    #             self.stats[ubatchsize][vlayer_id][name2] = ConstMeta(name2, int(ubatchsize))
    #             name3 = "X%d_3"%vlayer_id
    #             self.stats[ubatchsize][vlayer_id][name3] = [ TensorMeta(name3, torch.Size([ubatchsize,(vlayer_id+1)*enlarge]), torch.float32, True) for _ in range(2) ]

    # def fill_dummy_meta_debug(self, contain=[1,2,3], dtype=torch.float32): # For TEST
    #     for ubatchsize, vlayer_metas in self.stats.items():
    #         for vlayer_id, name_metas in vlayer_metas.items():
    #             assert name_metas == None
    #             self.stats[ubatchsize][vlayer_id] = ODict()
    #             # { name1: tensor, name2: const, name3: [tensors] }
    #             if 1 in contain:
    #                 name1 = "X%d_1"%vlayer_id
    #                 self.stats[ubatchsize][vlayer_id][name1] = TensorMeta(name1, torch.Size([ubatchsize,vlayer_id+1]), dtype, True)
    #             if 2 in contain:
    #                 name2 = "X%d_2"%vlayer_id
    #                 self.stats[ubatchsize][vlayer_id][name2] = ConstMeta(name2, int(ubatchsize))
    #             if 3 in contain:
    #                 name3 = "X%d_3"%vlayer_id
    #                 self.stats[ubatchsize][vlayer_id][name3] = [ TensorMeta(name3, torch.Size([ubatchsize,vlayer_id+1]), dtype, True) for _ in range(2) ]

class TMeta(XMeta): 
    # { ubatchsize: { last_vlayer_id: {name:TensorMeta} } } }
    def __init__(self, ubatchsizes, num_vlayers):
        super().__init__(ubatchsizes, num_vlayers)
        self.last_vlayer_id = num_vlayers-1
        for ubatchsize, vlayer_metas in self.stats.items():
            for vlayer_id in list(vlayer_metas.keys()):
                if vlayer_id != self.last_vlayer_id:
                    del vlayer_metas[vlayer_id]
    
    @property
    def last_vlayer_id(self):
        return self.last_stage_id

    @last_vlayer_id.setter
    def last_vlayer_id(self, value):
        self.last_stage_id = value

    def __repr__(self, which='X'):
        return super().__repr__('T')

    def init_target_meta_on_last_vlayer(self, target_names, target_tensors):
        self.target_names = target_names
        for ubatchsize in self.stats.keys():
            self.set(ubatchsize, self.last_vlayer_id, target_names, target_tensors, is_ubatch=False)

class WMeta(object): 
    # TODO: move __init__ functionality out side of this data struct
    def __init__(self, model, attr_name="named_parameters"):
        assert isinstance(model, list)
        assert attr_name in ["named_parameters", "named_buffers"]
        # per parameter size
        self.param_size = ODict() # { vlayer_id: { name:TensorMeta, name:ConstMeta } or None }        
        for vlayer_id, (vlayer, _, _) in enumerate(model):
            if len(list( getattr(vlayer, attr_name)() )) == 0:
                self.param_size[vlayer_id] = None
            else:
                self.param_size[vlayer_id] = ODict()
                for name, param in getattr(vlayer, attr_name)():
                    if isinstance(param, (torch.Tensor, Variable)):
                        self.param_size[vlayer_id][name] = TensorMeta(name, param.data.shape, dtype=param.data.dtype, is_ubatch=False)
                    elif isinstance(param, (int, float)):
                        self.param_size[vlayer_id][name] = ConstMeta(name, param)
                    else:
                        raise ValueError("unknown type of parameter: {}".format(param))
        # per vlayer size
        self.vlayer_size = ODict() # { vlayer_id: 100 bytes or 0 bytes }
        for vlayer_id, name_metas in self.param_size.items():
            if name_metas is None:
                self.vlayer_size[vlayer_id] = 0
            else:
                self.vlayer_size[vlayer_id] = int(np.sum([meta.bytes for meta in name_metas.values()]))
    @property
    def vlayer_size(self):
        return self.stage_size

    @vlayer_size.setter
    def vlayer_size(self, value):
        self.stage_size = value

    def __repr__(self, which='W'):
        repre  = "\n----------------------- %s meta-data -----------------------\n"%(which)
        repre += "vlayer_id: vlayer_size bytes (TensorMeta bytes, ConstMeta bytes)\n"
        for vlayer_id, name_metas in self.param_size.items():
            if name_metas is None: 
                repre += "%d: -\n" % (vlayer_id)
            else:
                repre += "%d: %d (%s)\n" % (vlayer_id, self.vlayer_size[vlayer_id], 
                                ", ".join([meta.__repr__(show_size=True) for meta in name_metas.values()]) )
                # repre += "%d: %d\n" % (vlayer_id, self.vlayer_size[vlayer_id]) # DEBUG
        repre += "-----------------------------------------------------------"
        return repre    

    def get_bytes(self, vlayer_id):
        return self.vlayer_size[vlayer_id]

class BMeta(WMeta):
    def __init__(self, model):
        super().__init__(model, attr_name="named_buffers")
    def __repr__(self):
        return super().__repr__('B')

class KMeta(object): 
    # TODO: move __init__ functionality out side of this data struct
    def __init__(self, model, optimizer):
        assert isinstance(model, list) and isinstance(optimizer, list) and len(model)==len(optimizer)
        # confirm model and optimizer are on CPU
        for (vlayer,_,_), optim in zip(model, optimizer):
            if optim is not None:
                assert not next(vlayer.parameters()).is_cuda
                for k, v in optim.state.items():
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
        # create zero gradient 
        for vlayer,_,_ in model:
            if len(list(vlayer.parameters())) != 0:
                for param in vlayer.parameters():
                    param.grad = torch.zeros_like(param.data)
        # force initialization of optimizer states (Bert, GPT2, Adam, SGD)
        for optim in optimizer:
            if optim is not None:
                optim.step() 
        # get per state size
        self.state_size = ODict() # { vlayer_id: { param.id : { state.name:TensorMeta, state.name:ConstMeta } } or None }        
        for vlayer_id, optim in enumerate(optimizer):
            if optim is None:
                self.state_size[vlayer_id] = None
            else: # optim of this vlayer
                self.state_size[vlayer_id] = ODict()
                assert optim.state, "optimizer state must be initialized by dummy grad and step" # non-empty
                for pid, (param, states) in enumerate(optim.state.items()): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                    self.state_size[vlayer_id][pid] = ODict()
                    for k, v in states.items():
                        if isinstance(v, (torch.Tensor, Variable)):
                            self.state_size[vlayer_id][pid][k] = TensorMeta(k, v.shape, dtype=v.dtype, is_ubatch=False)
                        elif isinstance(v, (int, float)):
                            self.state_size[vlayer_id][pid][k] = ConstMeta(k, v)
                        else:
                            raise ValueError("unknown state {}:{}".format(k,v))
        # print("self.state_size={}".format(self.state_size)) # DEBUG
        # get per vlayer size
        self.vlayer_size = ODict() # { vlayer_id: 100 bytes or 0 bytes }
        for vlayer_id, param_states in self.state_size.items():
            if param_states is None:
                self.vlayer_size[vlayer_id] = 0
            else:
                cnt = 0
                for named_states in param_states.values():
                    for state in named_states.values():
                        cnt += state.bytes
                self.vlayer_size[vlayer_id] = cnt
        # print("self.vlayer_size={}".format(self.vlayer_size)) # DEBUG

    @property
    def vlayer_size(self):
        return self.stage_size

    @vlayer_size.setter
    def vlayer_size(self, value):
        self.stage_size = value

    def __repr__(self, which='K'):
        repre  = "\n----------------------- %s meta-data -----------------------\n"%(which)
        repre += "vlayer_id: vlayer_size bytes\n"
        for vlayer_id, param_states in self.state_size.items():
            if param_states is None: 
                repre += "%d: -\n" % (vlayer_id)
            else:
                repre += "%d: %d\n" % (vlayer_id, self.vlayer_size[vlayer_id])
        repre += "-----------------------------------------------------------"
        return repre

    def get_bytes(self, vlayer_id):
        return self.vlayer_size[vlayer_id]

def save_prof_data_struct(data_struct, path_dir, fname, base_dir="prof", verbose=True):
    if ".pickle" not in fname:
            fname += ".pickle"
    assert os.path.exists(path_dir)
    if base_dir is None:
        full_path = os.path.join(path_dir, fname)
    else:
        full_dir = os.path.join(path_dir, base_dir)
        os.makedirs(full_dir, exist_ok=True)
        full_path = os.path.join(full_dir, fname)
           
    with open(full_path,'wb') as f:
        pickle.dump(data_struct, f)
    if isinstance(data_struct,int):
        full_path = full_path.replace(".pickle", ".json")
        with open(full_path, 'w') as f:
            json.dump(data_struct, f)
    if verbose: print("prof_data_struct saved to: {}".format(full_path))

def load_prof_data_struct(path_dir, fname, base_dir="prof", verbose=True):
    if ".pickle" not in fname:
        fname += ".pickle"
    if base_dir is None:
        full_path = os.path.join(path_dir, fname)
    else:
        full_path = os.path.join(path_dir, base_dir, fname)
    assert os.path.exists(full_path)
    
    with open(full_path,'rb') as f:
        fdata = pickle.load(f)
    if verbose: print("prof_data_struct load from: {}".format(full_path))
    
    return fdata


