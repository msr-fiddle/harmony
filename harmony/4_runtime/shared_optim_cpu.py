# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import gc
import threading
from collections import OrderedDict as ODict

import torch

from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

from task_data_struct import Medium, vTask
import threadsafe_data_struct

if os.environ.get('CUDA_LAUNCH_BLOCKING') in ['1','True', True]:
    MEMCPY_NONBLK = False
else:
    MEMCPY_NONBLK = True

def convert_to_pinned(local_model_cpu):
    ''' in-place convert a local model cpu to a pinned model (params and buffers: pinned, local, CPU, no grad) '''
    @torch.no_grad()
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # no grad
                assert param.grad is None, "convert to pinned model requires no grad in input model"
                param.detach_()
                assert not param.requires_grad
                # pin param
                param.data = param.pin_memory() # in-place update and let python do the gc 
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                assert not buf.requires_grad # buffer has no grad
                m._buffers[key] = buf.pin_memory() # in-place update and let python do the gc 
                assert not m._buffers[key].requires_grad
    local_model_cpu.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    gc.collect()

class SharedOptimCPU(object):
    """ A wrapper class of shared optimizer (referring to each shared vlayer) on CPU.
        Data structure:
        o	a multiprocess-shared vlayer and optimizer
        o	a process-local pinned .grad as input buffer
        o	a process-local pinned model (param and buffs) as output buffer
        TODO: decouple pinned_model from here
    """
    @torch.no_grad()
    def __init__(self, shared_model, optimizer, id=-1):
        """ Call this in parent process and before fork/spawn subprocess """ 
        self.id = id # vlayer_id
        if optimizer is None: # this wrapper is only for pinned model
            self.shared_model = shared_model
            self.shared_optimizer = None
            # print("[SharedOptimizer][id%d] optimizer is None; for pinned model only."%self.id)
        else:
            # confirm model and optimizer are on CPU
            for param in shared_model.parameters():
                assert not param.is_cuda and param.is_shared()
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
            # 1) create zero gradient 
            for param in shared_model.parameters():
                param.grad = torch.zeros_like(param.data) # == torch.zeros(input.size(), input.dtype, input.layout, input.device, requires_grad=False)
                # Note: "param.grad = tensor" works, but "param.data.grad = tensor" doesn't work
                # Note: id(param.grad) == id(param.data.grad)
            # 2) force initialization of optimizer states (Bert, GPT2, Adam, SGD)
            optimizer.step() 
            # 3) move optimzer.state to shared memory
            # print("[SharedOptimizer] sharing optimizer:")
            for param, state in optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        v.share_memory_(); assert v.is_shared()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))
                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage 
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64) 
                        state[k].share_memory_(); assert state[k].is_shared()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            # 4) move optimzer.hyperparams to shared memory? No. They are constant (Bert, GPT2, Adam, SGD)
            # 5) clean up gradient
            for param in shared_model.parameters():
                param.grad = None
                assert param.grad is None, "cannot None grad?"
            gc.collect()
            # optimizer.zero_grad()
            self.shared_model = shared_model
            self.shared_optimizer = optimizer
            # print("[SharedOptimizer][id%d] sharing optimizer done."%self.id)
    
    @torch.no_grad()
    def init_in_subproc(self, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this on entering subprocess """
        self.rank, self.world_size = rank, world_size
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf
        #
        if self.shared_optimizer is not None:
            # confirm model and optimizer are shared     
            for param in self.shared_model.parameters():
                assert param.data.is_shared()
                assert param.requires_grad 
                # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
            # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                for k, v in state.items():
                    assert isinstance(v, torch.Tensor) and v.is_shared()
            for k, v in self.shared_optimizer.param_groups[0].items():    
                if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
                    assert (not isinstance(v, torch.Tensor)) # or (isinstance(v, torch.Tensor) and v.is_shared())
                # ref1: https://pytorch.org/docs/1.5.0/_modules/torch/optim/optimizer.html#Optimizer.state_dict
                # ref2: https://pytorch.org/docs/1.5.0/_modules/torch/optim/adam.html#Adam
            # print("[SharedOptimizer] rank{}'s optimizer is shared".format(self.rank))
            
            # initialize local pinned .grad # Trimed
            # for param in self.shared_model.parameters():
            #     assert param.requires_grad
            #     param.grad = torch.zeros(param.shape, dtype=param.dtype, device="cpu", requires_grad=False).pin_memory()
            #     assert not param.grad.is_shared() and param.grad.is_pinned()
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned .grad"%(self.id, self.rank))

        if self.no_pin_model:
            self.pinned_model = None
        else:
            # initialize local pinned model (params and buffs)
            # with torch.no_grad():
            self.pinned_model = copy.deepcopy(self.shared_model)
            convert_to_pinned(self.pinned_model) # in-place convert a local model cpu to a pinned model
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned model"%(self.id, self.rank))

    @torch.no_grad()
    def update_buf(self):
        """ In-place copy buffer from local pinned buf to shared model """  
        if self.no_pin_grad_buf:
            return
        else:
            assert hasattr(self.shared_model, "pinned_buf")
            named_pin_buf = self.shared_model.pinned_buf
            for name, shared_buf in self.shared_model.named_buffers():
                if shared_buf is not None:    
                    shared_buf.data.copy_(named_pin_buf[name].data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local pinned buffer must match shared model"

        # for shared_buf, pinned_buf in zip(self.shared_model.buffers(), self.pinned_model.buffers()):
        #     shared_buf.data.copy_(pinned_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from pinned memory to shared memory # nonblocking useless
        #     assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
    
    @torch.no_grad()
    def step(self, zero_grad=False):
        if self.shared_optimizer is not None:
            self.shared_optimizer.step()
            if zero_grad:
                self.shared_optimizer.zero_grad()
        # confirm local .grad is still pinned
        # for param in self.shared_model.parameters():
        #     assert param.grad.is_pinned()
        # print("[SharedOptimizer] rank{} steped shared optimizer".format(self.rank))
    
    @torch.no_grad()
    def sync_pinned_model(self):
        """ In-place copy from shared model to local pinned model """  
        if self.no_pin_model:
            return
        #
        for pinned_param, shared_param in zip(self.pinned_model.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(self.pinned_model.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
        # print("[SharedOptimizer] rank{} synced pinned model".format(self.rank))


""" CPU update and sync model in background thread """
class UpdateInBkgd(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, lr_scheduler, rank, nvprof=False):
        self.swapout_stream = swapout_stream
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            self._wait_swapout()
            self._update_buf(vt) # if using local pinned model for B'
            self._step(vt)
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    def _wait_swapout(self):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. # (maybe use cuda event in future)

    @torch.no_grad()
    def _update_buf(self, vt):
        """ update B of this pack """  
        for l in vt.layers:
            self.shared_optimizer[l].update_buf()
    
    @torch.no_grad()
    def _step(self, vt):
        """ update W,K of this pack """  
        for l in vt.layers:
            assert vt.In['dW'][l].medium == "LOC"
            assert vt.In['W'][l].medium == "SHM"  
            assert vt.In['K'][l].medium == "SHM"
            assert vt.Out['W'][l].medium == "SHM"
            assert vt.Out['K'][l].medium == "SHM" 
            self.shared_optimizer[l].step() # Update shared model and optim using swap-out'ed local .grad
            if self.lr_scheduler != []: # "gpt2_huggingface"
                if self.lr_scheduler[l] is not None:
                    assert self.shared_optimizer[l].shared_optimizer is not None
                    self.lr_scheduler[l].step() 
                else:
                    assert self.shared_optimizer[l].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))

class SyncPinModelInBkgd(object):
    """ Handles synchronization to local pinned model in background thread.
        Assumption:
            0) always in FIFO ordering. put each task, and get synced task. 
        TODO: skip sync_pinned_model() if already done (FWD -> BWD)
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between sync and prefetch thread
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        # print("[SyncPinModelInBkgd] rank{} started sync_thread".format(self.rank))

    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) SyncPin(W,B)".format(vt.idx, vt.show_layers())) 
            self._sync_pinned_model(vt)
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    @torch.no_grad()
    def _sync_pinned_model(self, vt):
        """ sync W,B to local pinned model for this pack """  
        for l in vt.layers:
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                self.shared_optimizer[l].sync_pinned_model()
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                pass
            else: # P2P
                raise ValueError("Underdevelopment")
    
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
