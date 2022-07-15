# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import threading
import gc
from collections import OrderedDict as ODict

import torch
import torch.distributed as dist
from torch.autograd import Variable

from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

from prof_data_struct import ConstMeta, TensorMeta


class P2PX(object):
    """ Handle P2P send-recv for X/dX of vPP with NCCL

        Feature: 
            0) Stateless recv (GPU memory are free'ed each microbatch) and Stateful prerecv (with double buffering)
            1) NCCL broadcast emulated send-recv
            2) Everything in the "Main" thread 
            3) Send-recv rank pairs use their own groups/communicators/cudaStreams
                --> Deadlock free, Full Duplex, Unblocking send and recv
            4) Support concurrent isend/recv/irecv for multiple tensors 
                *Plan-A: use multiple isend unblockingly (catch: PyTorch send-recv only works for a single tensor --> blocking)
                Plan-B: move multiple isend-recv to background thread, just like Gloo simpleCommHandler (bad: P2P(NCCL) is not thread-safe! )
                Plan-C: concatenate multiple tensors into one big tensor, then isend (catch: extra memory/time for concatenate/split)
            5) Support microbatch size conversion by using 'UBatchSizeConverterP2P'
                last-FWD sending to 1st-BWD needs converting microbatch sizes on GPUs. (catch: extra memory/time for concatenate/split)
            6) Built-in cuda event:
                <cudaEventRecord & cudaStreamWaitEvent> -> 
                launch nccl kernel (broadcast) -> 
                <cudaEventRecord, cudaEventCreateWithFlags, cudaEventRecord, cudaStreamWaitEvent> (ireq.wait)

        Assumption:
            0) distributed environment has already been initialized with gloo
            1) during send-recv, X/dX has no grad (after send-recv, can set them to requires_grad for BWD)
    """
    def __init__(self, rank, world_size, reverse_bwd=True, verbose=False, nvprof=False):
        assert dist.get_backend() == "gloo"
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.nvprof = nvprof
        assert self.rank == torch.cuda.current_device() # torch.cuda.set_device(rank)
        # build two-process group for NCCL broadcast (r1, r2)  
        self.groups = ODict() # { "r1->r2": dist.group_obj }
        # in-order round-robin
        for r1 in range(self.world_size):
            r2 = (r1+1) % self.world_size
            pgroup = dist.new_group(ranks=[r1, r2], backend='nccl')
            # This function requires that 1) all processes in the main group (i.e. all processes that are part of the distributed job) enter this function, even if they are not going to be members of the group; 2) groups should be created in the same order in all processes.
            # This new_group only creates empty NCCL groups (communicator and cudaStream are not initialized yet)
            if self.rank in [r1,r2]:
                self.groups["{}->{}".format(r1,r2)] = pgroup
        # reverse round-robin
        if reverse_bwd:
            for r1 in range(self.world_size):
                r2 = (r1-1) % self.world_size
                pgroup = dist.new_group(ranks=[r1, r2], backend='nccl')
                if self.rank in [r1,r2]:
                    self.groups["{}->{}".format(r1,r2)] = pgroup
        # print("[P2P] rank={}, world_size={}, self.groups = {}".format(self.rank, self.world_size, self.groups))
        # initialize NCCL communicator and its cudaStream in mainthread
        tensor = torch.tensor(1.0, dtype=torch.float32, device="cuda:%d"%(self.rank))
        for key, group in self.groups.items():
            dist.broadcast(tensor, group=group, src=int(key.split("->")[0])) # init communicator should be in-order
            # print("[P2P] rank={} init'ed NCCL communicator[{}] and its cudaStream".format(self.rank, key))
        # clean up
        del tensor; gc.collect(); torch.cuda.empty_cache()
        # for pre-recv
        self.is_irecving = False
        self.ubatch_idx = int(0)
        self.double_bufs = [None, None]
        # for bytes counter
        if self.verbose: 
            self.send_byte_cnt = 0
            self.recv_byte_cnt = 0

    @torch.no_grad()
    def _isend_tensor(self, tensor, dst):
        """ Non-Blocking send a tensor via NCCL broadcast """
        # print("[P2P]\trank{}: _isend_tensor({},{}) to dst:{}".format(self.rank, tensor.shape, tensor.dtype, dst))
        assert tensor.is_cuda
        group_key = "{}->{}".format(self.rank,dst)
        ireq = dist.broadcast(tensor, src=self.rank, group=self.groups[group_key], async_op=True)
        # print("[P2P]\trank{}: _isend_tensor'ed".format(self.rank))
        if self.verbose: self.send_byte_cnt += tensor.nelement()*tensor.element_size()
        return ireq
    
    @torch.no_grad()
    def _irecv_tensor(self, tensor=None, shape=None, dtype=torch.float32, src=-1):
        """ Non-Blocking recv a tensor via NCCL broadcast.
            If tensor is None, then its shape (e.g. () or (1,) or (2,2)) must be given to create a tensor, to receive, and to return this GPU tensor. 
            Else, return filled GPU tensor.

            case-1: _irecv_tensor(shape=(1,2), src=123) # create new
            case-2: _irecv_tensor(tensor=cuda_tensor, src=123) # reuse existing
        """
        assert (tensor is None and shape is not None) or (tensor is not None and shape is None)
        tensor = torch.empty(shape, dtype=dtype, device="cuda:%d"%self.rank) if tensor is None else tensor
        assert tensor.is_cuda
        # print("[P2P]\trank{}: _irecv_tensor({},{}) from src:{}".format(self.rank, tensor.shape, tensor.dtype, src))
        group_key = "{}->{}".format(src, self.rank)
        ireq = dist.broadcast(tensor, src=src, group=self.groups[group_key], async_op=True)
        # ireq.wait() # blocking
        # print("[P2P]\trank{}: _irecv_tensor'ed".format(self.rank))
        if self.verbose: self.recv_byte_cnt += tensor.nelement()*tensor.element_size()
        return tensor, ireq
    
    @torch.no_grad()
    def _isend_const(self, const, dst):
        """ Non-Blocking send a const int/float via NCCL """
        # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
        assert isinstance(const, (int,float))
        tensor = torch.tensor(const, device="cuda:%d"%self.rank)
        ireq = self._isend_tensor(tensor, dst) # """ nccl unblocking isend works better than gloo """
        del tensor # It works
        # print("[P2PHandler] rank={}: dist.broadcast'ed".format(self.rank))
        return ireq

    @torch.no_grad()
    def _irecv_const(self, tensor=None, const=None, src=-1):
        """ Non-Blocking send a const scalar via NCCL.
            If tensor is None, then const must be given (int/float) to create a tensor, to receive, and to return this GPU tensor.
            Else, return filled GPU tensor.
            
            case-1: _irecv_const(const=123, src=123) # create new
            case-2: _irecv_const(tensor=cuda_tensor, src=123) # reuse existing
        """
        assert (tensor is None and isinstance(const, (int,float))) or (tensor is not None and const is None)
        # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
        tensor = torch.tensor(const, device="cuda:%d"%self.rank) if tensor is None else tensor
        tensor, ireq = self._irecv_tensor(tensor=tensor, src=src)
        # ireq.wait() # blocking
        # print("[rank{}]\tP2PX._irecv_const: irecv'ed {}".format(self.rank, tensor))
        return tensor, ireq # Need tensor.item() to convert a 0-dim tensor to a python number, once received
        
    # @torch.no_grad() # Deprecated
    # def _isend_const(self, const, dst, tag=7777777):
    #     """ Non-Blocking send a const int/float via Gloo """
    #     # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
    #     assert isinstance(const, (int,float))
    #     ireq = dist.isend(torch.tensor(const), dst=dst, tag=tag)
    #     ireq.wait() 
    #     """ must blocking for const, otherwise the CPU const can be deleted too soon, causing irecv wait forever """
    #     """ But if really blocks, it causes P2P deadlock. """
    #     # print("[P2PHandler] rank={}: dist.broadcast'ed".format(self.rank))
    #     return ireq

    # @torch.no_grad() # Deprecated
    # def _irecv_const(self, const, src, tag=7777777):
    #     """ Non-Blocking send a const int/float via Gloo """
    #     # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
    #     assert isinstance(const, (int,float))
    #     tensor = torch.tensor(const, device="cpu", pin_memory=True)
    #     ireq = dist.irecv(tensor, src=src, tag=tag)
    #     # ireq.wait() # blocking
    #     # print("[rank{}]\tP2PX._irecv_const: irecv'ed {}".format(self.rank, tensor))
    #     return tensor, ireq # Need tensor.item() to convert a 0-dim tensor to a python number
    
    def isend(self, named_tensors, dst):
        ''' Call by main thread. Nonblocking send. '''    
        # print("[P2P]\trank{}: isend entered".format(self.rank))
        for name,tensor in named_tensors.items(): # { name: tensor/const, name: [tensors] }
            if isinstance(tensor, (torch.Tensor,Variable)):
                assert tensor.is_cuda and not tensor.requires_grad
                self._isend_tensor(tensor, dst)
            elif isinstance(tensor, (float,int)):
                self._isend_const(tensor, dst)
            elif isinstance(tensor, list): # output tuple of bert pretrainhead 
                for t in tensor:
                    assert t.is_cuda and not t.requires_grad
                    self._isend_tensor(t, dst)
            else:
                raise ValueError("unknown tensor={}".format(tensor))
            # print("[P2P]\trank{}: isend'ed {}:{} to dst:{}".format(self.rank, name, type(tensor), dst))
    
    def recv(self, named_metas, src=-1):
        ''' Call by main thread. Blocking recv. 
            Assumption: 
                0) Stateless: always create allocate new tensor, and let it delete by runtime (no double buffering)
                1) Blocking compute stream by cuda events
            Argument: named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      src = src rank
            Return: recved named_tensors. '''    
        # print("[P2P]\trank{}: recv entered".format(self.rank))
        named_tensors = ODict()
        named_ireq = ODict()
        for name, meta in named_metas.items():
            if isinstance(meta, TensorMeta):
                named_tensors[name], named_ireq[name] = self._irecv_tensor(shape=meta.shape, dtype=meta.dtype, src=src)
            elif isinstance(meta, ConstMeta):
                named_tensors[name], named_ireq[name] = self._irecv_const(const=meta.const, src=src)
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                tmp_tensor, tmp_ireq = [], []
                for m in meta:
                    tensor, ireq = self._irecv_tensor(shape=m.shape, dtype=m.dtype, src=src)
                    tmp_tensor.append(tensor); tmp_ireq.append(ireq)
                named_tensors[name] = tmp_tensor
                named_ireq[name] = tmp_ireq
            else:
                raise ValueError("unknown meta={}".format(meta))
            # print("[P2P]\trank{}: recv's irecv'ed {}:{}".format(self.rank, name, meta))
        # wait all tensors recved (by built-in cuda event)
        for name, meta in named_metas.items():
            if isinstance(meta, TensorMeta):
                named_ireq[name].wait()
            elif isinstance(meta, ConstMeta):
                named_ireq[name].wait()
                named_tensors[name] = named_tensors[name].item() # convert a 0-dim cpu/cuda tensor to a python number # let python do the gc on cuda tensor
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                # for ireq, tensor in zip(named_ireq[name],named_tensors[name]):
                #     ireq.wait()
                [ ireq.wait() for ireq in named_ireq[name] ]  
            else:
                raise ValueError("unknown meta={}".format(meta))
            # print("[P2P]\trank{}: recv's ireq.waited {}:{} from src:{}".format(self.rank, name, meta, src))
        # print("[P2P]\trank{}: recv's all ireqs waited".format(self.rank))
        # clean up
        return named_tensors
    
    def _irecv(self, named_metas, src, buffer=None): 
        ''' Non-Blocking recv. 
            Assumption: only one irecv can be running.
            Argument: named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      src = src rank
                      buffer = None or (named_tensors, named_ireq)
            Return: created or input (named_tensors, named_ireq)
        '''
        assert not self.is_irecving, "the irecv is still running"
        self.is_irecving = True
        # print("[P2P]\trank{}: _irecv'ing".format(self.rank))
        if buffer is None: # allocate new tensors
            if self.nvprof: nvtx_range_push("P2PIn Alloc & iBcast") 
            named_tensors = ODict()
            named_ireq = ODict()
            for name, meta in named_metas.items():
                if isinstance(meta, TensorMeta):
                    named_tensors[name], named_ireq[name] = self._irecv_tensor(shape=meta.shape, dtype=meta.dtype, src=src)
                elif isinstance(meta, ConstMeta):
                    named_tensors[name], named_ireq[name] = self._irecv_const(const=meta.const, src=src)
                elif isinstance(meta, list): # output tuple of bert pretrainhead 
                    tmp_tensor, tmp_ireq = [], []
                    for m in meta:
                        tensor, ireq = self._irecv_tensor(shape=m.shape, dtype=m.dtype, src=src)
                        tmp_tensor.append(tensor); tmp_ireq.append(ireq)
                    named_tensors[name] = tmp_tensor
                    named_ireq[name] = tmp_ireq
                else:
                    raise ValueError("unknown meta={}".format(meta))
            if self.nvprof: nvtx_range_pop() 
            # print("[P2P]\trank{}: _irecv allocated new tensors and requested all".format(self.rank))
        else: # reuse buffered tensors
            named_tensors, named_ireq = buffer
            for name, meta in named_metas.items():
                if isinstance(meta, TensorMeta):
                    named_tensors[name], named_ireq[name] = self._irecv_tensor(tensor=named_tensors[name], src=src)
                elif isinstance(meta, ConstMeta):
                    named_tensors[name], named_ireq[name] = self._irecv_const(tensor=named_tensors[name], src=src)
                elif isinstance(meta, list): # output tuple of bert pretrainhead 
                    assert isinstance(named_tensors[name], list) and isinstance(named_ireq[name], list)
                    for i in range(len(meta)):
                        named_tensors[name][i], named_ireq[name][i] = self._irecv_tensor(tensor=named_tensors[name][i], src=src)
                else:
                    raise ValueError("unknown meta={}".format(meta))
            # print("[P2P]\trank{}: _irecv reused buffered tensors and requested all".format(self.rank))
        return named_tensors, named_ireq
        
    def _wait_irecv(self, named_metas, buffer=None): 
        ''' Wait for the running irecv. 
            Assumption: only one irecv can be running.
            Arguments: the same as _irecv, except buffer is not None
            Return: recved named_tensors.
        '''
        assert self.is_irecving, "no running irecv"
        self.is_irecving = False
        # wait all tensors recved
        assert buffer is not None
        named_tensors, named_ireq = buffer
        recved_named_tensors = ODict() # ref to buffer
        for name, meta in named_metas.items():
            if isinstance(meta, TensorMeta):
                named_ireq[name].wait()
                recved_named_tensors[name] = named_tensors[name] # ref
            elif isinstance(meta, ConstMeta):
                named_ireq[name].wait()
                # named_tensors[name] = named_tensors[name].item() # convert a 0-dim cpu/cuda tensor to a python number
                recved_named_tensors[name] = named_tensors[name].item()
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                tmp_tensor = []
                for tensor, ireq in zip(named_tensors[name], named_ireq[name]):
                    ireq.wait()
                    tmp_tensor.append(tensor)
                recved_named_tensors[name] = tmp_tensor # ref
            else:
                raise ValueError("unknown meta={}".format(meta))
        return recved_named_tensors 

    def prerecv(self, named_metas, src, is_end=False): 
        ''' Call by main thread. Blocking recv current one and unblocking pre-recv next one. 
            Assumption: 
                    1) use double buffering for parallel compute and irecv
                    2) double buffering doesn't change in shape TODO: fix
                    3) use cudaEvent to sync with compute stream, and event calls are still on CPU async w.r.t GPU streams
                    4) no prerecv for successor group
            Argument: named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      src = src rank
                      is_end = whether ends prerecv after current recv
            Return: recved current named_tensors. '''    
        #
        cur_buf_idx = self.ubatch_idx % 2 # for compute
        next_buf_idx = (self.ubatch_idx+1) % 2 # for pre irecv
        # wait current one (if no current one, _irecv & _wait_irecv one)
        if not self.is_irecving:
            self.double_bufs[cur_buf_idx] = self._irecv(named_metas, src, None)
        cur_named_tensors = self._wait_irecv(named_metas, self.double_bufs[cur_buf_idx])
        # irecv next one if exists
        if not is_end:
            self.double_bufs[next_buf_idx] = self._irecv(named_metas, src, self.double_bufs[next_buf_idx])
            self.ubatch_idx += 1
        else: # clean up
            self.ubatch_idx = 0
            del self.double_bufs 
            self.double_bufs = [None, None]            
        return cur_named_tensors # reference only; to be deleted by runtime

    def prerecv_suc(self, suc_info): 
        ''' Prerecv successor group's 1st ubatch if exists. Call by main thread. 
            Assumption: same 1) 3) above
            Argument: suc_info = None or successor group's (named_metas, src_rank)
        '''  
        if suc_info is None:
            return
        suc_named_metas, suc_src = suc_info
        # print("\trank{}: P2P.prerecv_suc({}, src{})".format(self.rank, suc_named_metas, suc_src))
        # must after is_end
        assert self.ubatch_idx == 0 and self.double_bufs == [None, None]
        # record previous compute event (built-in)
        self.double_bufs[0] = self._irecv(suc_named_metas, suc_src, buffer=None)

class P2PModel(object):
    """ Handle P2P allreduce for dW/B of vDP with NCCL 
        
        Feature: 
            0) Stateless (GPU memory are free'ed each microbatch)
            1) Everything in the "Main" thread 
            2) Use a new group/communicator/cudaStream
            3) Blocking
        
        Assumption:
            0) distributed environment has already been initialized with gloo
    """
    def __init__(self, rank, world_size, verbose=False):
        assert dist.get_backend() == "gloo"
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        assert self.rank == torch.cuda.current_device() # torch.cuda.set_device(rank)
        # build a world group for NCCL collectives
        tensor = torch.tensor(1.0, dtype=torch.float32, device="cuda:%d"%(self.rank))
        self.world_group = dist.new_group(ranks=list(range(self.world_size)), backend='nccl')
        dist.all_reduce(tensor, group=self.world_group, op=dist.ReduceOp.SUM)
        # print("[P2PM] rank={} init'ed world communicator of NCCL and its cudaStream".format(self.rank))
        del tensor; gc.collect(); torch.cuda.empty_cache()
        if self.verbose: self.allreduce_byte_cnt = 0

    @torch.no_grad()
    def average_gradients(self, model):
        """ GPU Gradient averaging. """
        size = float(self.world_size)
        for param in model.parameters():
            dist.all_reduce(param.grad.data, group=self.world_group, op=dist.ReduceOp.SUM)
            param.grad.data /= size
            if self.verbose: self.allreduce_byte_cnt += param.grad.data.nelement()*param.grad.data.element_size()
    
    @torch.no_grad()
    def average_buffers(self, model):
        """ Buffer averaging. """
        size = float(self.world_size)
        for buf in model.buffers():
            if isinstance(buf.data, torch.Tensor) and buf.data.dtype in [torch.float16, torch.float32, torch.float64]: 
                if buf.is_cuda:
                    dist.all_reduce(buf.data, group=self.world_group, op=dist.ReduceOp.SUM) # NCCL
                else: 
                    dist.all_reduce(buf.data, op=dist.ReduceOp.SUM) # Gloo
                buf.data /= size
                if self.verbose: self.allreduce_byte_cnt += buf.data.nelement()*buf.data.element_size()
