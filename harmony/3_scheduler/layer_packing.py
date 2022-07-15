# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from math import ceil, floor

def convert_to_layer_packs(packed_memories): 
    # packed_memories: [array([100, 50, 110]), array([100, 50, 110])] # bytes
    if packed_memories == []:
        return []
    assert isinstance(packed_memories, list)
    assert isinstance(packed_memories[0], np.ndarray) 
    
    # convert to layer packs
    cnt = 0
    layer_packs = [] # [ [0,1,2], [3,4,5] ]
    per_pack_memories = [] # [ 260, 260 ]
    for pack in packed_memories:
        num_layers = len(pack)
        assert num_layers > 0, "no empty pack allowed"
        layer_packs.append( list(range(cnt, cnt+num_layers)) )
        cnt += num_layers
        per_pack_memories.append(pack.sum())
    # assert cnt == len(per_layer_memories)
    
    return layer_packs, per_pack_memories

def print_memory_packing(layer_packs, per_pack_memories, title="", tab="\t"):
    assert isinstance(layer_packs, list) and isinstance(per_pack_memories, list)
    memories = np.array(per_pack_memories)
    print("%s-------%s-------"%(tab,title))
    for i, (layers, mem) in enumerate(zip(layer_packs, memories)):
        print("%s#%04d: L%04d-%04d: %6.0f MB"%
              (tab, i, layers[0], layers[-1], mem/1024./1024.))
    print("%sper_pack_memories: mean %.0f, std %.0f, max %.0f, min %.0f MB"%
        (tab, memories.mean()/1024./1024., memories.std()/1024./1024., memories.max()/1024./1024., memories.min()/1024./1024.))


def greedy_memory_packing(per_layer_memories, capacity, 
                        reverse=False, per_layer_x=None,
                        verbose=False, title="greedy_memory_packing", tab="\t"):
    """ 
    Arguments:
        per_layer_memories: read-only

        reverse: if False, greedy packing from 1st to last layer, leave fraction in the last criterion pack.
                 if True, greedy packing from last to 1st layer, leave fraction in the 1st pack. 
                 packed layer id is always ascending. 
        
        per_layer_x: if not None, add back x size in each pack; otherwise, more pack will have less memory (x stripped off) when summing up per_layer_memories; ready-only
    """
    
    assert isinstance(per_layer_memories, list) # [ 100, 50, 110, 100, 50, 110 ] # bytes
    # if verbose: print("\tper_layer_memories (MB): {}".format(memories/1024./1024.)) 
    memories = np.array(per_layer_memories) # a numpy object with a new memory
    if per_layer_x is None:
        xmems = np.zeros(len(memories))
    else:
        xmems = np.array(per_layer_x)
        assert len(xmems) == len(memories)
    
    # pack memories by accumulating to capacity
    if reverse is False:
        packed_memories = [] # [array([100, 50, 110]), array([100, 50, 110])] # bytes
        new_pack_sum = memories[0] + xmems[0]
        # assert new_pack_sum < capacity, "capacity cannot hold even one layer"  
        if new_pack_sum >= capacity:
            if verbose: print("capacity cannot hold even one layer; return None")
            return None
        new_pack = [ new_pack_sum ]
        for mem, x in zip(memories[1:], xmems[1:]): # until the last layer
            new_pack_sum += mem
            if new_pack_sum < capacity:
                new_pack.append(mem)
            else:
                packed_memories.append(np.array(new_pack))
                new_pack_sum = mem + x
                # assert new_pack_sum < capacity, "capacity cannot hold even one layer"  
                if new_pack_sum >= capacity:
                    if verbose: print("capacity cannot hold even one layer; return None")
                    return None
                new_pack = [ new_pack_sum ]
        packed_memories.append(np.array(new_pack))
    else:
        # reverse layer order
        memories = np.flip(memories) # return a view of m with the entries reversed (underlying memory shared)
        xmems = np.flip(xmems)
        # pack
        packed_memories = [] # [array([100, 50, 110]), array([100, 50, 110])] # bytes
        new_pack_sum = memories[0] # the last layer
        # assert new_pack_sum + xmems[0] < capacity, "capacity cannot hold even one layer"
        if new_pack_sum + xmems[0] >= capacity:
            if verbose: print("capacity cannot hold even one layer; return None")
            return None
        new_pack = [ new_pack_sum ]
        prev_x = xmems[0]
        for mem, x in zip(memories[1:], xmems[1:]): # until the first layer
            new_pack_sum += mem
            if new_pack_sum + x < capacity:
                new_pack.append(mem)
            else:
                new_pack[-1] += prev_x
                packed_memories.append(np.array(new_pack))
                new_pack_sum = mem
                # assert new_pack_sum + x < capacity, "capacity cannot hold even one layer"
                if new_pack_sum + x >= capacity:
                    if verbose: print("capacity cannot hold even one layer; return None")
                    return None
                new_pack = [ new_pack_sum ]
            prev_x = x
        new_pack[-1] += prev_x
        packed_memories.append(np.array(new_pack))
        # # confirm add x into each pack
        # cnt = 0
        # for pack in packed_memories: 
        #     cnt += len(pack)
        #     idx = cnt - 1
        #     assert pack[-1] == memories[idx] + xmems[idx]
        # restore layer order
        packed_memories.reverse() # inplace
        for i in range(len(packed_memories)):
            packed_memories[i] = np.flip(packed_memories[i])
    
    # confirm correctness
    assert len(memories) == sum(len(pack) for pack in packed_memories)
    if per_layer_x is None:
        assert memories.sum() == sum(pack.sum() for pack in packed_memories)
    assert max(pack.sum() for pack in packed_memories) < capacity
    # if verbose: print("\tpacked_memories (MB): {}".format(
    #                    [pack/1024./1024. for pack in packed_memories]))
    # convert to layer packs
    layer_packs, per_pack_memories = convert_to_layer_packs(packed_memories)
    
    if verbose: print_memory_packing(layer_packs, per_pack_memories, title=title, tab=tab)
    return layer_packs

def _balanced_sum_split(A, S):
    """ split 'A' into 'S' 'packs' such that 'packs' have approximately equal sum """
    
    assert isinstance(A, np.ndarray) and A.dtype in (np.int64, np.float64), "A = {} ({})".format(A, A.dtype)
    # get prefix sum
    prefix_sum = A.cumsum()
    # approximate per-pack sum
    pack_sum = prefix_sum[-1] // S if A.dtype == np.int64 else prefix_sum[-1] / S
    # get prefix sum of per-pack sum
    prefix_pack_sum = np.array(range(1, S)) * pack_sum 
    # binary search the indices such that the prefix per-pack sums are inserted
    indices = np.searchsorted(prefix_sum, prefix_pack_sum, side='left')
    # split into approximately equal-sum packs
    packs = np.split(A, indices) # a list of sub-arrays as views into A (underlying memory is shared between sub-arrays as A)
    
    return packs # [array([0, 1, 2]), array([3, 4, 5])] (memory shared with input A)
 
def balanced_memory_packing(per_layer_memories, capacity, 
                            per_layer_x=None, 
                            verbose=False, title="balanced_memory_packing", tab="\t"):
    """ 
    Argument:
        per_layer_memories: read-only
        capacity: a forced constraint during packing
        per_layer_x: if not None, add back x size in each pack; otherwise, more pack will have less memory (x stripped off) when summing up per_layer_memories; ready-only
    """
    
    assert isinstance(per_layer_memories, list) # [ 100, 50, 110, 100, 50, 110 ] # bytes
    # if verbose: print("\tper_layer_memories (MB): {}".format(memories/1024./1024.)) 
    memories = np.array(per_layer_memories) # a numpy object with a new memory (int)
    if per_layer_x is not None:
        assert len(per_layer_x) == len(memories)
    
    # find num of packs (initial guess)
    num_packs = ceil(memories.sum()/float(capacity))
    
    # parition into num of packs (under capacity constraint)
    packed_memories = None
    for S in range(num_packs, len(memories)+1):
        # balance the memory per pack
        packed = _balanced_sum_split(memories, S) # [array([100, 50, 110]), array([100, 50, 110])] # bytes
        # check empty pack
        if sum([ int(len(pack)==0) for pack in packed ]) != 0:
            if verbose: print("\tbalanced %d packs has empty pack; try more packs"%S) 
            continue
        # check memory of each pack
        if per_layer_x is None:
            if max(pack.sum() for pack in packed) < capacity:
                packed_memories = packed # found
                break
        else:
            is_exceed = False
            idx = 0
            for pack in packed: 
                if pack.sum() + per_layer_x[idx] > capacity:
                    is_exceed = True
                    break
                idx += len(pack)
            if not is_exceed:
                packed_memories = packed # found 
                idx = 0
                for pack in packed_memories: # add x in results (don't add x during compare, because memories and packed_memories share the memory)
                    pack[0] += per_layer_x[idx]
                    idx += len(pack)
                break
        # continue packing       
        if verbose: print("\tbalanced %d packs exceed capacity; try more packs"%S)  
    
    # check results
    if packed_memories is None:
        return None
    else: # confirm conrrectness
        assert len(memories) == sum(len(pack) for pack in packed_memories)
        if per_layer_x is None:
            assert memories.sum() == sum(pack.sum() for pack in packed_memories)
        # if verbose: print("\tpacked_memories (MB): {}".format(
        #                    [pack/1024./1024. for pack in packed_memories]))  
        # convert to layer packs
        layer_packs, per_pack_memories = convert_to_layer_packs(packed_memories)
        
        if verbose: print_memory_packing(layer_packs, per_pack_memories, title=title, tab=tab)
        return layer_packs

def reuse_memory_packing(packs, per_layer_memories, per_layer_x, capacity, 
                         verbose=False, tab="\t"):
    """ Reuse input packing (e.g. FWD reuses BWD's).
        If packs' memory is within capacity, then return it.
        Else, return None. """
    
    assert packs is not None
    if packs == []:
        return []
    assert isinstance(per_layer_memories, list) # [ 100, 50, 110, 100, 50, 110 ] # bytes
    memories = np.array(per_layer_memories) # a numpy object with a new memory
    if per_layer_x is None:
        xmems = np.zeros(len(memories))
    else:
        xmems = np.array(per_layer_x)
        assert len(xmems) == len(memories)
    
    # check if memory < capacity
    if max([ memories[p].sum() + xmems[p[0]] for p in packs ]) < capacity:
        if verbose: print("{}reuse packs: {}".format(tab,packs)) 
        return packs
    else:
        # if verbose: print("{}not reusable packs: {}".format(tab,packs)) 
        return None

def balanced_time_packing(per_layer_times, 
                          per_layer_memories, per_layer_x, capacity, 
                          verbose=False, title="balanced_time_packing", tab="\t"):
    """ 
    Argument: per_layer_times: Compute time of each layer (no Swap nor P2P)
    """
    
    assert isinstance(per_layer_times, list)
    assert isinstance(per_layer_memories, list)
    assert isinstance(per_layer_x, list)
    assert len(per_layer_times) == len(per_layer_memories) and \
           len(per_layer_memories) == len(per_layer_x)
    times = np.array(per_layer_times) # a numpy object with a new memory
    memories = np.array(per_layer_memories) # a numpy object with a new memory
    
    # find num of packs (initial guess)
    num_packs = ceil(memories.sum()/float(capacity))
    
    # parition into num of packs (under capacity constraint)
    packed_memories = None
    for S in range(num_packs, len(memories)+1):
        # balance the time per pack
        packed_times = _balanced_sum_split(times, S)
        if verbose: print("\tS: {}, packed_times: {}".format(S, packed_times))
        # check empty pack
        if sum([ int(len(pack)==0) for pack in packed_times ]) != 0:
            if verbose: print("\tbalanced %d packs has empty pack; try more packs"%S) 
            continue
        # check memory of each pack
        is_exceed = False
        idx = 0
        for pack in packed_times: 
            if memories[idx: idx+len(pack)].sum() + per_layer_x[idx] > capacity:
                is_exceed = True
                break
            idx += len(pack)
        if not is_exceed:
            packed_memories = [] # found
            idx = 0
            for pack in packed_times: 
                pack_mem = memories[idx: idx+len(pack)]
                pack_mem[0] += per_layer_x[idx]
                packed_memories.append(pack_mem)
                idx += len(pack)
            break
        # continue packing       
        if verbose: print("\tbalanced %d packs exceed capacity; try more packs"%S)  
    
    # check results
    if packed_memories is None:
        return None
    else: # confirm conrrectness
        assert len(memories) == sum(len(pack) for pack in packed_memories)
        if verbose: 
            print("\tpacked_times (sec) sum: {}".format(
                    [pack.sum() for pack in packed_times] ))
            # print("\tpacked_memories (MB): {}".format(
            #             [pack/1024./1024. for pack in packed_memories]))                
        # convert to layer packs
        layer_packs, per_pack_memories = convert_to_layer_packs(packed_memories)
        
        if verbose: print_memory_packing(layer_packs, per_pack_memories, title=title, tab=tab)
        
        return layer_packs

def convert_splits_to_packs(splits, inclusive=True, verbose=False):
    """ 
    Assumption: splits elements are always increasing

    If input "splits" is inclusive:
        split [x,y,z] will be converted to [ [0,...,x], [x+1,...,y], [y+1,...,z] ] 
    else input "splits" is exclusive:
        split [x,y,z] will be converted to [ [0,...,x-1], [x,...,y-1], [y,...,z-1] ] 
        
    Return: converted packs for vt.layers
    """
    packs = []
    
    if inclusive:
        prev_s = -1
        for s in splits:
            assert prev_s < s
            packs.append( list(range(prev_s+1,s+1)) )
            prev_s = s
    else:
        prev_s = 0
        for s in splits:
            assert prev_s < s
            packs.append( list(range(prev_s,s)) )
            prev_s = s
    if verbose: print("convert_splits_to_packs: {}".format(packs))
    
    return packs

def manual_pack(R, pack_size, reverse=True, verbose=False):
    """ a.k.a constant size packing.
        pack 'R' layers by constant 'pack_size'
        
        when R % pack_size !=0, 
            if reverse: first pack is less than 'pack_size'
            otherwise:  last pack is less than 'pack_size'  """
    assert R != -1 and pack_size != -1
    if reverse:
        splits = list(range(R-1, -1, -pack_size))[::-1]
    else:
        splits = list(range(pack_size-1, R, pack_size))
        if splits[-1] != R-1:
            splits.append(R-1)
    
    pack_bwd = convert_splits_to_packs(splits, verbose=verbose)
    pack_fwd = convert_splits_to_packs(splits[:-1], verbose=verbose)
    
    return pack_fwd, pack_bwd

# # # Old
# def constant_memory_packing(per_layer_memories, capacity, verbose=False, title="constant_memory_packing"):
#     """ Assume uniform layers, i.e., each layer has the same 'constant' memory """
#     """ Deprecated: 1) no capacity constraint 2) optimal might be not constant """
#     assert isinstance(per_layer_memories, list) # [ 100, 50, 110, 100, 50, 110 ] # bytes
#     memories = np.array(per_layer_memories)
#     R = len(memories)
#     avg_memory_per_layer = memories.sum() / R
#     constant_packsize = int(floor(capacity/avg_memory_per_layer))
#     if verbose: print("\tconstant_packsize = %d"%constant_packsize)
#     pack_sizes = [ R%constant_packsize ] + \
#                  [ constant_packsize ] * int(R/constant_packsize)
#     if pack_sizes[0] == 0:
#         pack_sizes.pop(0)
#     assert sum(pack_sizes) == R
#     # packed_memories: [array([100]), array([100, 50, 110]), array([100, 50, 110])]
#     packed_memories = []
#     cnt = 0
#     for num_layers in pack_sizes:
#         packed_memories.append( np.array(memories[cnt:cnt+num_layers]) )       
#         cnt += num_layers
#     # confirm correctness
#     assert len(memories) == sum(len(pack) for pack in packed_memories)
#     assert memories.sum() == sum(pack.sum() for pack in packed_memories)
#     if verbose: print("\tpacked_memories (MB): {}".format(
#                        [pack/1024./1024. for pack in packed_memories]))
#     # convert to layer packs
#     layer_packs, per_pack_memories = convert_to_layer_packs(packed_memories)
#     
#     if verbose: print_memory_packing(layer_packs, per_pack_memories, title=title)
#     return layer_packs
