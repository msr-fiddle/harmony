# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import OrderedDict as ODict

import torch
import torch.distributed as dist
from torch.autograd import Variable

from prof_data_struct import ConstMeta, TensorMeta, XMeta, TMeta
from profiler import realize_X, realize_dX, realize_D, realize_T

@torch.no_grad()
def pin_named_tensors(cpu_named_tensors):
    for name,tensor in cpu_named_tensors.items(): # { name: tensor/const, name: [tensors] }
        if isinstance(tensor, (torch.Tensor,Variable)):
            cpu_named_tensors[name] = tensor.pin_memory()
        elif isinstance(tensor, (float,int)):
            continue
        elif isinstance(tensor, list): # output tuple of bert pretrainhead 
            tmp = []
            for t in tensor:
                tmp.append(t.pin_memory())
            cpu_named_tensors[name] = tmp
        else:
            raise ValueError("unknown tensor={}".format(tensor))

def synthesize_data(XMETA, TMETA, ubatchszs_fwd, ubatchszs_bwd, pin_memory=True):
    # from profiler import realize_D, realize_T # overhead: https://wiki.python.org/moin/PythonSpeed/PerformanceTips#Import_Statement_Overhead
    data_ubatches = [] # [ {named_tensors}, {named_tensors}, ... ]
    target_ubatches = [] # [ {named_tensors}, {named_tensors}, ... ]

    for u in ubatchszs_fwd:
        named_data = realize_D(XMETA, u, device="cpu", use_rand=False)
        if pin_memory:
            pin_named_tensors(named_data)
        data_ubatches.append(named_data)
    
    for u in ubatchszs_bwd:
        named_target = realize_T(TMETA, u, device="cpu", use_rand=False)
        if pin_memory:
            pin_named_tensors(named_target)
        target_ubatches.append(named_target)
    
    return data_ubatches, target_ubatches

def _turn_on_grad(tensor):
    """ If tensor has no gradient (.grad=None/detached/not requires_grad), then just turn on its gradient.
    Else, keep its gradient buff (with zero out), detach from graph, then turns on its gradient. """
    assert isinstance(tensor, (torch.Tensor,Variable))
    if tensor.grad is None:
        assert tensor.is_leaf and not tensor.requires_grad
        tensor.requires_grad_(True)
        tensor.retain_grad()
    else: # double buffer of P2PIn
        tensor.grad.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
        tensor.grad.zero_()
        tensor.detach_()
        assert not tensor.requires_grad
        tensor.requires_grad_(True)
        tensor.retain_grad()
        assert tensor.grad is not None
            
def turn_on_X_grad(X_named_tensors):
    for name, tensor in X_named_tensors.items():
        if name in ["input0","input1","input2","input_ids"]: # TODO: add identity chain to input (separate identity removal func to outside)
            continue
        if isinstance(tensor, (torch.Tensor,Variable)):
            _turn_on_grad(tensor)
        elif isinstance(tensor, (int,float)):
            continue
        elif isinstance(tensor, list): # output tuple of bert pretrainhead 
            [_turn_on_grad(t) for t in tensor]
        else:
            raise ValueError("unknown tensor={}".format(tensor))

def make_tensors_named(names, tensors):
    assert isinstance(tensors,list)
    named_tensors = ODict()
    if len(names) == 1 and len(tensors) > 1: # output tuple of bert pretrainhead
        named_tensors[names[0]] = tensors # tuple(tensors)
    else:
        for name, tensor in zip(names, tensors):
            named_tensors[name] = tensor
    return named_tensors

@torch.no_grad()
def make_dX_from_X(X_named_tensors):
    """ Make reference to .grad of named_tensors for P2P/SWP sending 
        Assumption: 
        1) Not data tensors 
        2) TODO: input identiy chain removed before this call
        3) Except input identiy chain and input const, all tensors has .grad to be sent (to match metas of receiver) 
    """
    dX_named_tensors = ODict()
    for name, X in X_named_tensors.items():
        if isinstance(X,(torch.Tensor, Variable)):
            assert X.requires_grad and X.grad is not None
            dX_named_tensors[name] = X.grad.data
            # assert not X.grad.data.requires_grad
        elif isinstance(X, list): # output tuple of bert pretrainheader
            dX_named_tensors[name] = [ x.grad.data for x in X ]
            assert len(dX_named_tensors[name]) != 0
    return dX_named_tensors

def make_dY_named_metas(XMETA, ubatchsize, layerid): 
    """ Make dY named metas based on profiled XMETA for P2P receiving
        Args: layerid = layer id of dY
        Assumption: The same as 'make_dX_from_X'.
    """
    # remove const, set float32
    named_metas = ODict() # { name:TensorMeta, name:[TensorMeta,TensorMeta] }
    for name, meta in XMETA.get(ubatchsize,layerid+1).items(): # named metas
        if isinstance(meta, TensorMeta):
            named_metas[name] = TensorMeta(name, meta.shape, dtype=torch.float32)
        elif isinstance(meta, list): # output tuple of bert pretrainheader
            named_metas[name] = [TensorMeta(name, m.shape, dtype=torch.float32) for m in meta]
    return named_metas


class SucInfoForPrefetch(object):
    """ Wrapper of finding successor info for all prefetches 
        NOTE: 
        1) don't use cache, which slows down 10x
        2) don't inline _suc_info_* function, which slows down 2x
    """
    def __init__(self, rank, rTASKS, XMETA):
        self.rank_vtasks = rTASKS[rank]
        self.XMETA = XMETA
    
    def set(self, vt, rank_vt_idx):
        self.vt = vt
        self.rank_vt_idx = rank_vt_idx
    
    def _find_suc_vt(self, rank_vt_idx, rank_vtasks):
        """ Assumption: max search distance is 2 tasks
            Return: None or found successor vt  """
        if rank_vt_idx + 1 >= len(rank_vtasks):
            return None
        if rank_vtasks[rank_vt_idx+1].type in ['FWD','BWD']:
            return rank_vtasks[rank_vt_idx+1]
        else:
            if rank_vt_idx+2 >= len(rank_vtasks):
                return None
            elif rank_vtasks[rank_vt_idx+2].type not in ['FWD','BWD']:
                return None
            else:
                return rank_vtasks[rank_vt_idx+2]
        
    def _find_case(self, vt, suc_vt):
        """" 
        case-1: current FWD (non-criterion), successor FWD (non-criterion)
        case-2: current FWD (non-criterion), successor BWD (criterion)
        case-3: current FWD (non-criterion), successor BWD (non-criterion) 
        case-4: current BWD (criterion),     successor BWD (non-criterion)
        case-5: current BWD (non-criterion), successor BWD (non-criterion) """
        if vt.type == 'FWD' and suc_vt.type == 'FWD':
            return 1
        elif vt.type == 'FWD' and suc_vt.type == 'BWD' and suc_vt.has_criterion:
            return 2
        elif vt.type == 'FWD' and suc_vt.type == 'BWD' and (not suc_vt.has_criterion):
            return 3
        elif vt.type == 'BWD' and vt.has_criterion and suc_vt.type == 'BWD' and (not suc_vt.has_criterion):
            return 4
        elif vt.type == 'BWD' and (not vt.has_criterion) and suc_vt.type == 'BWD' and (not suc_vt.has_criterion):
            return 5
        else:
            raise ValueError

    def model(self):
        return self._find_suc_vt(self.rank_vt_idx, self.rank_vtasks)

    def _suc_info_X_for_p2p_prerecv(self, suc_vt, XMETA):
        ### In {X} 
        l, m = suc_vt.layers[0], suc_vt.In['X'][suc_vt.layers[0]]
        if m.medium != "P2P":
            return None
        else: # indeed P2PIn
            return XMETA.get(suc_vt.ubatchszs[0],l), m.rank

    def _suc_info_dY_for_p2p_prerecv(self, suc_vt, XMETA):
        ### In {dY}
        l, m = suc_vt.layers[-1], suc_vt.In['dY'][suc_vt.layers[-1]]
        if m.medium != "P2P":
            return None
        else: # indeed P2PIn
            return make_dY_named_metas(XMETA, suc_vt.ubatchszs[0], l), m.rank

    def p2pin(self):
        """ 
        case-1: -> P2PIn(X) @ FWD's ULast
        case-2: -> P2PIn(X) @ FWD's ULast
        case-3: -> P2PIn(dY) @ FWD's ULast
        case-4: -> P2PIn(dY) @ Recompute (criterion)'s ULast
        case-5: -> P2PIn(dY) @ BWD's ULast
        Return: None or 
                ( suc_named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }, suc_src = 0~3 ) """
        suc_vt = self._find_suc_vt(self.rank_vt_idx, self.rank_vtasks)
        if suc_vt is None:
            return None
        case = self._find_case(self.vt, suc_vt)
        if case in [1,2]:
            return self._suc_info_X_for_p2p_prerecv(suc_vt, self.XMETA)
        else:
            return self._suc_info_dY_for_p2p_prerecv(suc_vt, self.XMETA)

    def _suc_info_X_for_prefetch_msgx(self, suc_vt, XMETA):
        ### In {MSGX}
        l, m = suc_vt.layers[0], suc_vt.In['X'][suc_vt.layers[0]]
        if m.medium != "MSG": 
            return None
        else: # last FWD convert to first BWD
            return l, XMETA.get(suc_vt.ubatchszs[0],l)

    def msgx(self):
        """ 
        case-2: -> MSGIn(X) @ FWD's ULast
        
        Return: None or 
                ( suc_layer_id, suc_named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] } ) """
        suc_vt = self._find_suc_vt(self.rank_vt_idx, self.rank_vtasks)
        if suc_vt is None:
            return None
        case = self._find_case(self.vt, suc_vt)
        if case in [2]:
            return self._suc_info_X_for_prefetch_msgx(suc_vt, self.XMETA)
        else:
            return None

    def _suc_info_stashx_for_prefetch_stashx(self, suc_vt, XMETA):
        ### In {StashX}
        l, m = suc_vt.layers[0], suc_vt.In['X'][suc_vt.layers[0]]
        if m.medium != "MSG":
            return None
        else:
            return l, XMETA.get(suc_vt.ubatchszs[0],l) 

    def stashx(self):
        """ 
        For StashX in vPP:
        case-3: -> SwapIn(StashX) @ FWD's ULast
        case-4: -> SwapIn(StashX) @ Recompute (criterion)'s ULast
        case-5: -> SwapIn(StashX) @ Recompute (non-criterion)'s ULast
        For StashX in vDP:
        case-3: -> doesn't exist
        case-4: -> same action
        case-5: -> same action
        Return: None or 
                ( suc_layer_id, suc_named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] ) """
        suc_vt = self._find_suc_vt(self.rank_vt_idx, self.rank_vtasks)
        if suc_vt is None:
            return None
        case = self._find_case(self.vt, suc_vt)
        if case in [3,4,5]:
            return self._suc_info_stashx_for_prefetch_stashx(suc_vt, self.XMETA)
        else:
            return None

    def _suc_info_X_for_prefetch_localx(self, suc_vt, XMETA):
        ### In {LocalX}
        l, m = suc_vt.layers[0], suc_vt.In['X'][suc_vt.layers[0]]
        if m.medium != "SWP": # swap locally for vDP
            return None
        else:
            return l, XMETA.get(suc_vt.ubatchszs[0],l)

    def _suc_info_dY_for_prefetch_localx(self, suc_vt, XMETA):
        ### In {dY}
        l, m = suc_vt.layers[-1], suc_vt.In['dY'][suc_vt.layers[-1]]
        if m.medium != "SWP": # swap locally for vDP
            return None
        else:   
            return l+1, make_dY_named_metas(XMETA, suc_vt.ubatchszs[0], l)

    def localx(self):
        """ 
        case-1: -> SwapIn(X) @ FWD's ULast
        case-2: -> SwapIn(X) @ FWD's ULast
        case-4: -> SwapIn(dY) @ BWD's ULast
        case-5: -> SwapIn(dY) @ BWD's ULast
        Return: None or 
                ( suc_layer_id, suc_named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] ) """
        suc_vt = self._find_suc_vt(self.rank_vt_idx, self.rank_vtasks)
        if suc_vt is None:
            return None
        case = self._find_case(self.vt, suc_vt)
        if case in [1,2]:
            return self._suc_info_X_for_prefetch_localx(suc_vt, self.XMETA)
        elif case in [4,5]:
            return self._suc_info_dY_for_prefetch_localx(suc_vt, self.XMETA)
        else:
            return None


# def make_dX_from_X(X_named_tensors): # for send
#     dX_named_tensors = ODict()
#     for name, X in X_named_tensors.items(): # only tensor & required_grad can run autograd
#         if isinstance(X,(torch.Tensor, Variable)) and (X.requires_grad):
#             if X.grad is not None:
#                 dX_named_tensors[name] = X.grad.data
#                 assert not X.grad.data.requires_grad
#         elif isinstance(X, list): # output tuple of bert pretrainheader
#             multi_grad = []
#             for x in X:
#                 if isinstance(x,(torch.Tensor, Variable)) and (x.requires_grad):
#                     if x.grad is not None:
#                         multi_grad.append(x.grad.data)
#                         assert not x.grad.data.requires_grad
#             if len(multi_grad) != 0:
#                 dX_named_tensors[name] = multi_grad
#     # NOTE: alreay considered gradient of identiy chain to input (separate identity removal func to outside)
#     return dX_named_tensors # ready to use for swp & msg & p2p send

# def make_dX_meta_from_X(X_named_tensors): # for receive
#     dX_named_metas = ODict() # { name:TensorMeta, name:[TensorMeta,TensorMeta] }
#     for name, X in X_named_tensors.items(): 
#         if isinstance(X,(torch.Tensor, Variable)): # and (X.requires_grad): # only tensor & required_grad can run autograd
#             dX_named_metas[name] = TensorMeta(name, X.shape, dtype=torch.float32)
#         elif isinstance(X, list): # output tuple of bert pretrainheader
#             multi_grad = []
#             for x in X:
#                 if isinstance(x,(torch.Tensor, Variable)): # and (x.requires_grad):
#                     multi_grad.append(TensorMeta(name, x.shape, dtype=torch.float32))
#             if len(multi_grad) != 0:
#                 dX_named_metas[name] = multi_grad
#     # NOTE: when grad of identity chain to input exsits, need to ignore X.requires_grad, i.e., always make dX meta
#     # TODO: when removing grad. of identity chain to input, only X.requires_grad can make dX. (separate identity removal func to outside)
#     return dX_named_metas # ready to use for p2p-recv
