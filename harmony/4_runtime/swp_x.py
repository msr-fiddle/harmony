# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from collections import OrderedDict as ODict
import threading

import torch
from torch.autograd import Variable

from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

from prof_data_struct import ConstMeta, TensorMeta
from task_data_struct import Medium, vTask
import threadsafe_data_struct

if os.environ.get('CUDA_LAUNCH_BLOCKING') in ['1','True', True]:
    MEMCPY_NONBLK = False
else:
    MEMCPY_NONBLK = True

""" Handles local swap-in/out of stashX/X/dX. 

    Assumption:
        0) stateless
        1) during swap, stashX/X/dX has no grad
"""

@torch.no_grad()
def swapout(cuda_named_tensors, pin_memory=True): 
    """ Argument: cuda_named_tensors (StashX of vPP, LocalX of vDP)
        Return: cpu_named_tensors in pinned memory
    """
    cpu_named_tensors = ODict()
    for name,tensor in cuda_named_tensors.items(): # { name: tensor/const, name: [tensors] }
        if isinstance(tensor, (torch.Tensor,Variable)):
            assert tensor.is_cuda and not tensor.requires_grad
            cpu_named_tensors[name] = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=pin_memory)
            cpu_named_tensors[name].copy_(tensor, non_blocking=MEMCPY_NONBLK) # inplace copy from cuda tensor to pinned memory 
            # assert cpu_named_tensors[name].is_pinned()
        elif isinstance(tensor, (float,int)):
            cpu_named_tensors[name] = tensor
        elif isinstance(tensor, list): # output tuple of bert pretrainhead 
            tmp = []
            for t in tensor:
                assert t.is_cuda and not t.requires_grad
                tmp.append( torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=pin_memory) )
                tmp[-1].copy_(t, non_blocking=MEMCPY_NONBLK)
                # assert tmp[-1].is_pinned()
            cpu_named_tensors[name] = tmp
        else:
            raise ValueError("unknown tensor={}".format(tensor))
    del cuda_named_tensors
    return cpu_named_tensors

@torch.no_grad()
def swapin(cpu_named_tensors): 
    """ Argument: cpu_named_tensors (stashX of vPP, stashX/X/dX of vDP) 
        Return: cuda_named_tensors
    """
    cuda_named_tensors = ODict()
    for name,tensor in cpu_named_tensors.items(): # { name: tensor/const, name: [tensors] }
        if isinstance(tensor, (torch.Tensor,Variable)):
            assert not tensor.is_cuda and not tensor.requires_grad
            # assert tensor.is_pinned()
            cuda_named_tensors[name] = tensor.cuda(non_blocking=MEMCPY_NONBLK) # to(device='cuda', non_blocking=True) # create new cuda tensor
        elif isinstance(tensor, (float,int)):
            cuda_named_tensors[name] = tensor
        elif isinstance(tensor, list): # output tuple of bert pretrainhead 
            tmp = []
            for t in tensor:
                assert not t.is_cuda and not t.requires_grad
                # assert t.is_pinned()
                tmp.append( t.cuda(non_blocking=MEMCPY_NONBLK) )
            cuda_named_tensors[name] = tmp
        else:
            raise ValueError("unknown tensor={}".format(tensor))
    # del cpu_named_tensors
    return cuda_named_tensors

""" Prefetch StashX/X/dX """

class SwapIn(object):
    """ Handle prefetch StashX in vPP/vDP and LocalX (X/dX) in vDP in background thread.
        Simliar to P2P.prerecv with double buffering.
        
        Step:
        1) main thread: waits for the running prefetch finish
        2) main thread: allocate or reuse buffer on GPU
        3) swapin thread: synchronize X on CPU
        4) swapin thread: copy in X in swapin_cudastream

        Assumption:
        0) statefull
        1) during swap, StashX/X/dX has no grad (but double buffering can have grad) 
        2) FIFO ordering. put each layerId, and get prefetched X. 
    """
    def __init__(self, sync_fn, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.sync_fn = sync_fn
        self.rank = rank
        self.swapin_stream = swapin_stream if swapin_stream is not None else torch.cuda.Stream(device=rank)
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert self.rank == torch.cuda.current_device() 
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [ layerId, layerId, ...]  # between main and swapin thread
        self.get_queue = threadsafe_data_struct.Queue() # [ #0X, #1X, ...] # between swapin and main thread
        self.swapin_thread = threading.Thread(target=self._thread_func)
        self.swapin_thread.daemon = True
        self.swapin_thread.start()
        # for preget
        self.is_running = False
        self.ubatch_idx = int(0)
        self.double_bufs = [None, None]
        # print("[SwapIn] rank{} started swapin_thread".format(self.rank))
    
    def _wait(self): 
        ''' Wait for the running swapin. Called in main thread.
            Assumption: only one swapin can be running.
            Return: swapined cuda_named_tensors.
        '''
        assert self.is_running, "no running swapin"
        self.is_running = False
        return self.get_queue.remove()

    # def _wait(self): # Deprecated: causing strange slowdown
    #     ''' Wait for the running swapin (iput). Called in main thread.
    #         Assumption: only one swapin can be running.
    #         Return: swapined cuda_named_tensors.
    #     '''
    #     assert self.is_running, "no running swapin"
    #     self.is_running = False
    #     # if self.nvprof: nvtx_range_push("L{} Wait(X)".format(layer_id)) 
    #     cuda_named_tensors, ev_swapin = self.get_queue.remove()
    #     self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
    #     # torch.cuda.default_stream(self.rank).synchronize() # wait Compute (PlanC)
    #     # if self.nvprof: nvtx_range_pop() 
    #     return cuda_named_tensors
    
    def _allocate(self, layer_id, named_metas, buffer): 
        ''' Allocate or reuse the buffer for next swapin. Called in main thread.
            Argument: named_metas = XMETA.get(ubatchsize, layer_id) # { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                       buffer = allocated cuda_named_tensors
            Return: newly allocated or reused buffer
        '''
        if buffer is None: # allocate new tensors
            if self.nvprof: nvtx_range_push("L{} Alloc(X)".format(layer_id)) 
            cuda_named_tensors = ODict()
            for name, meta in named_metas.items():
                if isinstance(meta, TensorMeta):
                    assert meta.is_ubatch
                    cuda_named_tensors[name] = torch.empty(meta.shape, dtype=meta.dtype, device="cuda:%d"%self.rank) 
                elif isinstance(meta, ConstMeta): 
                    cuda_named_tensors[name] = meta.const
                elif isinstance(meta, list): # output tuple of bert pretrainhead 
                    cuda_named_tensors[name] = [ torch.empty(m.shape, dtype=m.dtype, device="cuda:%d"%self.rank) for m in meta ]
                else:
                    raise ValueError("unknown meta={}".format(meta))   
            if self.nvprof: nvtx_range_pop() 
            return cuda_named_tensors
        else: # reuse buffered tensors
            # TODO: confirm named_metas matches buffer
            return buffer

    def _sync_copyin(self, layer_id, cuda_named_tensors, ev_compute=None):
        ''' Put allocated buffer to background swapin. Call by main thread thread. 
            Assumption: only one swapin can be running.
            Return: swapined cuda_named_tensors. '''
        assert not self.is_running, "the swapin is still running"
        self.is_running = True
        self.put_queue.add((layer_id, cuda_named_tensors, ev_compute))

    @torch.no_grad() 
    def _copyin(self, cpu_named_tensors, cuda_named_tensors):
        ''' Call by background swapin thread.
            Argument: cpu_named_tensors = src buffer on CPU
                      cuda_named_tensors = dst buffers on GPU
            (Return: this cuda_named_tensors with filled data)
        '''
        with torch.cuda.stream(self.swapin_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
            for (cname,ctensor), (gname, gtensor) in zip(cpu_named_tensors.items(), cuda_named_tensors.items()): # { name: tensor/const, name: [tensors] }
                assert cname == gname and type(ctensor) == type(gtensor)
                if isinstance(ctensor, (torch.Tensor,Variable)):
                    assert not ctensor.requires_grad, \
                    "{} (requires_grad:{})".format(ctensor, ctensor.requires_grad)
                    # assert ctensor.is_pinned() 
                    gtensor.data.copy_(ctensor.data, non_blocking=MEMCPY_NONBLK)
                elif isinstance(ctensor, (float,int)):
                    assert ctensor == gtensor
                elif isinstance(ctensor, list): # output tuple of bert pretrainhead 
                    for ct,gt in zip(ctensor,gtensor):
                        assert not ct.requires_grad
                        # assert ct.is_pinned()
                        gt.data.copy_(ct.data, non_blocking=MEMCPY_NONBLK)
                else:
                    raise ValueError("unknown tensor={}".format(tensor))
        return self.swapin_stream.record_event() # record a swapin event in this stream for compute stream to wait
        # # wait for copy stream 
        # if MEMCPY_NONBLK: self.swapin_stream.synchronize() # Wait for all the kernels in this stream to complete. 
                
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each iput'ed element
            layer_id, cuda_named_tensors, ev_compute = self.put_queue.remove() # blk
            if ev_compute is not None:
                self.swapin_stream.wait_event(ev_compute) # Stream waits for this event 
                # ev_compute.synchronize() # this CPU thread waits for this event. # Deprecated (too slow)
            # if self.nvprof: nvtx_range_push("__L{} SyncCopyIn(X)".format(layer_id)) 
            # sync
            cpu_named_tensors = self.sync_fn(layer_id) # thread safe dict
            # copyin
            ev_swapin = self._copyin(cpu_named_tensors, cuda_named_tensors)
            # ready to use
            self.get_queue.add( (cuda_named_tensors, ev_swapin) )
            # clean up reference
            del cuda_named_tensors
            # if self.nvprof: nvtx_range_pop() 
   
    def fetch(self, layer_id, named_metas):
        ''' Blocking fetch current X on GPU. Call by main thread.
            Feature: 
            0) Stateless
            1) Blocking compute stream (otherwise fetch ubatches can drift from compute ubatches) by cuda events
        '''
        # record previous compute event for swapin stream to wait
        ev_compute = self.compute_stream.record_event()
        # fetch current one
        cuda_named_tensors = self._allocate(layer_id, named_metas, None)
        self._sync_copyin(layer_id, cuda_named_tensors, ev_compute)
        # if self.nvprof: nvtx_range_push("L{} Wait(X)".format(layer_id)) 
        cuda_named_tensors, ev_swapin = self._wait()
        self.compute_stream.wait_event(ev_swapin)
        # if self.nvprof: nvtx_range_pop() 
        return cuda_named_tensors # to be delete'd by runtime

    def prefetch(self, layer_id, named_metas, is_end=False): 
        ''' Blocking wait current X and unblocking pre-swapin next X. Call by main thread. 
            Assumption: 
                      1) use double buffering for parallel compute and swapin
                      2) double buffering doesn't change in shape TODO: fix
                      3) PlanA: use cudaEvent to sync with compute stream, and event calls are still on CPU async w.r.t GPU streams
                      4) no prefetch for successor group's X
            Argument: layer_id = vLayer id of current StashX/X/dX
                      named_metas = current { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      is_end = whether ends pre-swapin after current one
            Return: recved current named_tensors. '''    
        # record previous compute event for swapin stream to wait
        ev_compute = self.compute_stream.record_event()
        # indexing double buffer
        cur_buf_idx = self.ubatch_idx % 2 # for compute
        next_buf_idx = (self.ubatch_idx+1) % 2 # for pre-swapin
        # wait current one (if no current one, have one)
        if not self.is_running:
            self.double_bufs[cur_buf_idx] = self._allocate(layer_id, named_metas, None)
            self._sync_copyin(layer_id, self.double_bufs[cur_buf_idx], ev_compute)
        # if self.nvprof: nvtx_range_push("L{} Wait(X)".format(layer_id)) 
        cur_named_tensors, ev_swapin = self._wait()
        self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        # if self.nvprof: nvtx_range_pop() 
        # pre-swapin next one if exsits
        if not is_end:
            self.double_bufs[next_buf_idx] = self._allocate(layer_id, named_metas, self.double_bufs[next_buf_idx])
            self._sync_copyin(layer_id, self.double_bufs[next_buf_idx], ev_compute)
            self.ubatch_idx += 1
        else: # clean up
            self.ubatch_idx = 0
            del self.double_bufs 
            self.double_bufs = [None, None]
        return cur_named_tensors # reference only; to be deleted by runtime
    
    def prefetch_suc(self, suc_info): 
        ''' Prefetc successor group's 1st ubatch if exists. Call by main thread. 
            Assumption: same 1) 3) above
            Argument: suc_info = None or successor group's (layer_id, named_metas)
        '''  
        if suc_info is None:
            return
        suc_layer_id, suc_named_metas = suc_info
        # if self.nvprof: nvtx_range_push("PrefetchSuc(L{}-X)".format(suc_layer_id)) 
        # must after is_end
        assert self.ubatch_idx == 0 and self.double_bufs == [None, None]
        # record previous compute event for swapin stream to wait
        ev_compute = self.compute_stream.record_event()
        # pre-swapin successor group's 1st ubatch if exsits
        self.double_bufs[0] = self._allocate(suc_layer_id, suc_named_metas, None)
        self._sync_copyin(suc_layer_id, self.double_bufs[0], ev_compute)
        # if self.nvprof: nvtx_range_pop() 

class SwapOut(object):
    """ Handle offload StashX in vPP/vDP and LocalX (X/dX) in vDP in background thread.
        
        Step:
        1) main thread: allocate pinned memory on CPU
        2) main thread: copy out X in swapout_stream
        3) main thread: optional delete
        4) swapout thread: wait for copy out
        5) swapout thread: isend to downstream ubatchconvert/msgstashx/localx

        Assumption:
        0) stateless
        1) during swap, StashX/X/dX has no grad
        2) FIFO ordering. put layer-by-layer, and swapout layer-by-layer. 
    """
    def __init__(self, output_fn, rank, swapout_stream=None, compute_stream=None, blocking=False, pin_memory=True, nvprof=False): # compute_stream2=None, 
        self.output_fn = output_fn # args: layer_id, named_tensor
        self.rank = rank
        self.swapout_stream = swapout_stream if swapout_stream is not None else torch.cuda.Stream(device=rank)
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        assert self.rank == torch.cuda.current_device() 
        self.blocking = blocking
        self.pin_memory = pin_memory
        self.nvprof = nvprof
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [ (layer_id, named_tensor, ev_compute), ...]  # between main and swapout thread
        # self.get_queue = threadsafe_data_struct.Queue() # [ ev_swapout, ...] # between swapout and main thread
        self.swapout_thread = threading.Thread(target=self._thread_func)
        self.swapout_thread.daemon = True
        self.swapout_thread.start()
        #
        # print("[SwapOut] rank{} started swapout_thread".format(self.rank))
    
    def offload(self, layer_id, cuda_named_tensors, flag=None):
        ''' Call by main thread. '''
        # record previous compute event for swapout stream to wait
        ev_compute = self.compute_stream.record_event()
        # Allocate and CopyOut and (Delete)
        self.swapout_stream.wait_event(ev_compute) # Stream waits for this event 
        # self.swapout_stream.wait_event(ev_compute2) # Stream waits for this event 
        # if self.nvprof: nvtx_range_push("L{} SwapOut(X)".format(layer_id)) 
        with torch.cuda.stream(self.swapout_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
            cpu_named_tensors = swapout(cuda_named_tensors, self.pin_memory)
        ev_swapout = self.swapout_stream.record_event() # record a swapout event in this stream for compute stream to wait
        if self.blocking: # optional blocking
            self.compute_stream.wait_event(ev_swapout)
        # if self.nvprof: nvtx_range_pop() 
        # wait in background thread
        self.put_queue.add((layer_id, cpu_named_tensors, ev_swapout, flag))
            
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each layer
            layer_id, cpu_named_tensors, ev_swapout, flag = self.put_queue.remove()
            # get ready for downstream
            # if self.nvprof: nvtx_range_push("__L{} WaitCopyOut(X)".format(layer_id)) 
            # if MEMCPY_NONBLK: self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. 
            if MEMCPY_NONBLK: ev_swapout.synchronize() # this CPU thread waits for this event. 
            self.output_fn(layer_id, cpu_named_tensors) if flag is None else \
            self.output_fn(layer_id, cpu_named_tensors, flag)
            # if self.nvprof: nvtx_range_pop() 
