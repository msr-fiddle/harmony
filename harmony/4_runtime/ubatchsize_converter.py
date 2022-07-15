# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading
from collections import OrderedDict as ODict

import torch

from torch.autograd import Variable
from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

import threadsafe_data_struct

class UBatchSizeConverter(object):
    """ Convert different microbatch sizes from forward to backward tasks.
        E.g. stashing X in vPP/vDP (including X to last layer pack in vDP)
        Assumption:
            0) know D, Ufwd, Ubwd in advance
            1) only CPU tensor and no grad
    """
    def __init__(self, rank, data_batchsize, u_fwd, ubatchszs_fwd, u_bwd, ubatchszs_bwd, output_method, pack_ordering=True, pin_memory=True, nvprof=False):
        self.rank = rank
        self.data_batchsize = data_batchsize
        self.u_fwd = u_fwd
        self.ubatchszs_fwd = ubatchszs_fwd
        self.u_bwd = u_bwd
        self.ubatchszs_bwd = ubatchszs_bwd
        if u_fwd == u_bwd: # assert u_fwd != u_bwd
            print("[UBatchSizeConverter] --- Warning: Ufwd = Ubwd ! ---") 
        assert data_batchsize >= u_fwd and data_batchsize >= u_bwd
        self.pin_memory = pin_memory
        self.nvprof = nvprof
        
        self._initialize(output_method, pack_ordering)
        self._start_helper_thread()

        # print("[UBatchSizeConverter] __init__: rank {} has D={}, Ufwd={} ({}), Ubwd={} ({})".format(self.rank, self.data_batchsize, self.u_fwd, self.ubatchszs_fwd, self.u_bwd, self.ubatchszs_bwd))

    def _initialize(self, output_method, pack_ordering=True):
        """
        Initialize state needed for sub-thread. 
        Argument: output_method(layer_id, named_tensor)
                  pack_ordering = bool : whether convert in layer or pack ordering (this ordering is self contained)
        """
        self.input_queue = threadsafe_data_struct.Queue()
        self.residual = ODict() # { layer_id: named_tensors (with ubatchsize < Ubwd) }
        self.cnt_converted_ubatch = ODict() # { layer_id: cnt }
        self.output_method = output_method # can be dict
        assert self.output_method is not None
        self.pack_ordering = pack_ordering
        assert isinstance(pack_ordering, bool)

    def _start_helper_thread(self):
        helper_thread = threading.Thread(target=self._helper_thread)
        helper_thread.daemon = True
        helper_thread.start()
        # print("[UBatchSizeConverter] rank{} started converter helper thread".format(self.rank))
    
    def _helper_thread(self):
        """ This method is to be executed from a helper daemon thread. """
        if self.pack_ordering: # output in pack ordering: [X0:#1, X1:#1], [X0:#2, X1:#2]
            # print("[UBatchSizeConverter] uses pack ordering")
            while True:
                packed_named_tensors, is_convert = self.input_queue.remove() # [ [ layer_id, named_tensors ] ] # a pack at an ubatch
                if not is_convert:
                    for layer_id, named_tensors in packed_named_tensors:
                        self.output_method(layer_id, named_tensors)
                    continue
                # converting
                layer_converted = ODict() # { layer_id: [#1 cvt_named_tensor, #2 cvt_named_tensor] } 
                for layer_id, named_tensors in packed_named_tensors:
                    converted = self._convert_ubatchsize(layer_id, named_tensors) # this layer's [ { cvt_named_tensor } ]
                    if converted == []:
                        # print("[UBatchSizeConverter] rank{}: converted is empty".format(self.rank))
                        continue
                    else:
                        layer_converted[layer_id] = converted
                #
                if layer_converted:
                    num_ubwds = set()
                    for converted in layer_converted.values():
                        num_ubwds.add(len(converted))
                    assert len(num_ubwds) == 1, "layers in a pack must have equal num of ubwd to send"
                    for idx in range(list(num_ubwds)[0]):
                        for layer_id, converted in layer_converted.items():
                            self.output_method(layer_id, converted[idx])
                            # print("[UBatchSizeConverter] rank{}: outputed L{}".format(self.rank, layer_id))
        else: # output in layer ordering: X0:[#1,#2,#3] then X1:[#1,#2,#3]
            # print("[UBatchSizeConverter] uses layer ordering")
            while True:
                layer_id, named_tensors, is_convert = self.input_queue.remove() # a layer at an ubatch
                if not is_convert:
                    self.output_method(layer_id, named_tensors)
                    continue
                # converting
                if self.nvprof: nvtx_range_push("__L{} ConvertU(X)".format(layer_id)) 
                converted = self._convert_ubatchsize(layer_id, named_tensors) # this layer's [ { named_tensors of Ubwd } ]
                if converted == []:
                    # print("[UBatchSizeConverter] rank{}: converted is empty".format(self.rank))
                    if self.nvprof: nvtx_range_pop() 
                    continue
                else:
                    for cvt_named_tensor in converted:
                        self.output_method(layer_id, cvt_named_tensor)
                        # print("[UBatchSizeConverter] rank{}: outputed L{}".format(self.rank, layer_id))
                    if self.nvprof: nvtx_range_pop() 
                          
    def _convert_ubatchsize(self, layer_id, named_tensors):
        """
        Helper thread converts one layer's tensors from Ufwd to Ubwd.
        Use previously stored residual tensors (not sufficient for Ubwd) for each convert call.
        Store residual tensors of this convert call for the next one.
        Return converted = [ { named_tensors of Ubwd } ] or []

        Note: the actually residual memory in pytorch == a residual Ubwd + an extra Ufwd == the concat'ed size 
              (i.e., _concat_tensors create an atomic big tensor. Even if a split of it gets deleted, the entire concat'ed memory is still there. Unless all splits gets deleted.)
        """
        # find new split
        named_split = ODict() # { name: (t1,t2), name: (c1,c2), name: [ (t1,t2), (t1,t2) ] }
        num_split = set()
        # self.residual = ODict() # { layer_id: named_tensors (with ubatchsize < Ubwd) }
        for name,tensor in named_tensors.items(): # { name: tensor/const, name: [tensors] }
            if isinstance(tensor, (torch.Tensor,Variable)):
                # assert not tensor.is_cuda and not tensor.requires_grad
                if layer_id in self.residual: # and name in self.residual[layer_id]:
                    concat_tensor = self._concat_tensors((self.residual[layer_id][name],tensor))
                else:
                    concat_tensor = tensor
                named_split[name] = self._split_tensor(concat_tensor, self.u_bwd) # (t1,t2) or (t1,res) or (t1,) or (res,)
                num_split.add(len(named_split[name]))
            elif isinstance(tensor, int):
                assert tensor in self.ubatchszs_fwd, "convert ubatchsize on unknown int value={}".format(tensor) # TODO: can use repeated int const
                if layer_id in self.residual: # and name in self.residual[layer_id]:
                    concat_tensor = self._concat_const_ubatchsizes((self.residual[layer_id][name],tensor))
                else:
                    concat_tensor = tensor
                named_split[name] = self._split_const_ubatchsize(concat_tensor, self.u_bwd) # (c1,c2) or (c1,res) or (c1,) or (res,)
                num_split.add(len(named_split[name]))
            elif isinstance(tensor, list): # output tuple of bert pretrainhead 
                tmp = []
                for i,t in enumerate(tensor):
                    # assert not t.is_cuda and not t.requires_grad
                    if layer_id in self.residual: # and name in self.residual[layer_id]:
                        concat_t = self._concat_tensors((self.residual[layer_id][name][i],t))
                    else:
                        concat_t = t
                    tmp.append(self._split_tensor(concat_t, self.u_bwd)) # (t1,t2) or (t1,res) or (t1,) or (res,)
                    num_split.add(len(tmp[-1]))
                named_split[name] = tmp
            else:
                raise ValueError("unknown tensor type to convert ={}".format(type(tensor)))
        # save residual and return converted
        assert len(num_split) == 1, "num_split must be unique"
        #
        if layer_id in self.residual: # { layer_id: named_tensors (with ubatchsize < Ubwd) }
            del self.residual[layer_id] 
        if not layer_id in self.cnt_converted_ubatch: # { layer_id: cnt }
            self.cnt_converted_ubatch[layer_id] = 0
        u_bwd = self.ubatchszs_bwd[self.cnt_converted_ubatch[layer_id]]
        converted = []
        for j in range(list(num_split)[0]):
            ready = ODict() # { name: t1, name: c1, name: [t1,t1] }
            not_ready = ODict() 
            for name, split in named_split.items(): # { name: (t1,t2), name: (c1,c2), name: [ (t1,t2), (t1,t2) ] }
                # print("[UBatchSizeConverter] rank{}'s named_split has {}:{}".format(self.rank, name, split)) 
                if isinstance(split,tuple) and isinstance(split[j], (torch.Tensor,Variable)):
                    tensor = split[j]
                    if tensor.size(0) == u_bwd: # 0-dim matches desired ubatchsize
                        ready[name] = tensor
                    elif tensor.size(0) < u_bwd: # residual
                        not_ready[name] = tensor
                    else:
                        raise ValueError
                elif isinstance(split,tuple) and isinstance(split[j], int):
                    tensor = split[j]
                    if tensor == u_bwd: # 0-dim matches desired ubatchsize
                        ready[name] = tensor
                    elif tensor < u_bwd: # residual
                        not_ready[name] = tensor
                    else:
                        raise ValueError
                elif isinstance(split,list):
                    # tmp_tensor, match_flag = [], []
                    # for s in split:
                    #     tensor = s[j]
                    #     tmp_tensor.append(tensor)
                    #     if tensor.size(0) == u_bwd: # 0-dim matches desired ubatchsize
                    #         match_flag.append(True)
                    #     elif tensor.size(0) < u_bwd: # residual
                    #         match_flag.append(False)
                    #     else:
                    #         raise ValueError
                    # if match_flag == [True]*len(match_flag):
                    #     ready[name] = tmp_tensor
                    # elif match_flag == [False]*len(match_flag):
                    #     not_ready[name] = tmp_tensor
                    # else:
                    #     raise ValueError
                    tmp1, tmp2 = [], []
                    for s in split:
                        tensor = s[j]
                        if tensor.size(0) == u_bwd: # 0-dim matches desired ubatchsize
                            tmp1.append(tensor)
                        elif tensor.size(0) < u_bwd: # residual
                            tmp2.append(tensor)
                        else:
                            raise ValueError
                    if tmp1 != []:
                        ready[name] = tmp1
                    elif tmp2 != []:
                        not_ready[name] = tmp2
                    else:
                        raise ValueError
                else:
                    raise ValueError
            #
            if ready:
                assert list(ready.keys()) == list(named_tensors.keys()), "{} vs. {}".format(list(ready.keys()), list(named_tensors.keys()))
                converted.append(ready)
                self.cnt_converted_ubatch[layer_id] += 1
                cnt = self.cnt_converted_ubatch[layer_id]
                if cnt < len(self.ubatchszs_bwd): # not last ubatch yet
                    u_bwd = self.ubatchszs_bwd[cnt]
                    # print("[UBatchSizeConverter] rank{}: converted L{}'s {} ubatches".format(self.rank,layer_id,cnt))
                else: # last ubatch done (of this iteration)
                    u_bwd = -1 # prevent keep looping
                    self.cnt_converted_ubatch[layer_id] = 0 # reset for next iteration
                    assert not layer_id in self.residual, "no more residual left"
                    # print("[UBatchSizeConverter] rank{}: converted L{}'s All {} ubatches".format(self.rank,layer_id,cnt))
            elif not_ready:
                assert j == list(num_split)[0]-1, "residual must be the last split"
                assert list(not_ready.keys()) == list(named_tensors.keys())
                self.residual[layer_id] = not_ready
            else:
                raise ValueError
        # clean up
        del named_split
        
        return converted
                
    def _concat_tensors(self, tensors):
        for t in tensors:
            # assert isinstance(t, (torch.Tensor,Variable))
            # assert not t.is_cuda and not t.requires_grad
            assert t.ndim > 0, "scalar tensor cannot be concat'ed"
        # dim=0 must be ubatchsize
        if self.pin_memory:
            return torch.cat(tensors, dim=0).pin_memory() # create new memory # inherit tensor's device
        else:
            return torch.cat(tensors, dim=0)

        
    def _split_tensor(self, t, split_size):
        # assert isinstance(t, (torch.Tensor,Variable))
        # assert not t.is_cuda and not t.requires_grad
        assert t.ndim > 0, "scalar tensor cannot be split'ed"
        # dim=0 must be ubatchsize
        return torch.split(t, split_size, dim=0) # share the same underlying memory # inherit tensor's device
        # tensor will be split into equally sized chunks (if possible). 
        # Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.
    
    def _concat_const_ubatchsizes(self, Cs):
        return int(sum(Cs))

    def _split_const_ubatchsize(self, C, U):
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
    
    def isend(self, input, input2=None, is_convert=True):
        ''' 
        Call by upstream thread. Nonblocking send.
        Argument: 
            input, input2 = 
                [ [ layer_id, named_tensors ] ] -- a pack at an ubatch
                or [ layer_id, named_tensors ]  -- a layer at an ubatch
                or layer_id, named_tensors      -- a layer at an ubatch
            is_convert = whether to convert this ubatch
        '''
        if self.pack_ordering:
            assert isinstance(input, list) and isinstance(input[0], list) and len(input[0])==2
            self.input_queue.add([input, is_convert])
        else:
            if input2 is None:
                assert isinstance(input, list) and len(input)==2
                self.input_queue.add( input+is_convert ) 
            else:
                self.input_queue.add([input, input2, is_convert]) 
