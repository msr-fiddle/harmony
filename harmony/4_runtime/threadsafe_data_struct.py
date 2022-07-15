# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading
from collections import OrderedDict as ODict

"""
Implementation of a thread-safe queue with one producer and one consumer.
"""
class Queue:
    def __init__(self):
        self.queue = []
        self.cv = threading.Condition()

    def add(self, tensor):
        self.cv.acquire()
        self.queue.append(tensor)
        self.cv.notify()
        self.cv.release()

    def remove(self):
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        tensor = self.queue.pop(0)
        self.cv.release()
        return tensor
    
"""
Implementation of a thread-safe dictionary with one producer and one consumer.
"""
class OrderedDictionary:
    def __init__(self):
        self.odict = ODict() # { layer_id: [u1's {"name1": tensor1, "name2": [tensor2]}, u2's {}, ... ] }
        self.cv = threading.Condition()

    def __repr__(self, title="thread safe dict"): #.format( "-" if src_rank is None else "(src_rank%d)"%(src_rank) )
        
        def show_named_tensors(named_tensors):
            sss = []
            for name, tensor in named_tensors.items():
                sss.append("{}:{}".format(name, type(tensor)))
            return "{ %s }" % (", ".join(sss))
        
        s = "----- %s -----\n"%(title)
        for layer_id, named_tensors_list in self.odict.items():
            ss = ", ".join([show_named_tensors(named_tensors) for named_tensors in named_tensors_list])
            s += "L{}:[{}]\n".format(layer_id, ss)
        s += "-------------------------------"
        return s

    def init_layer_ids(self, layer_ids): # always ascending
        assert isinstance(layer_ids,list)
        for id in sorted(layer_ids): 
            self.odict[id] = []
        self.layer_ids = list(self.odict.keys())
    
    def add(self, layer_id, named_tensors):
        self.cv.acquire()
        # if layer_id not in self.odict:
        #     self.odict[layer_id] = []
        self.odict[layer_id].append(named_tensors)
        self.cv.notify()
        self.cv.release()

    def remove(self, layer_id):
        self.cv.acquire()
        # if layer_id not in self.odict:
        #     self.cv.release()
        #     return None
        while len(self.odict[layer_id]) == 0:
            self.cv.wait()
        named_tensors = self.odict[layer_id].pop(0)
        self.cv.release()
        return named_tensors

    
