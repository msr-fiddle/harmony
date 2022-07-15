# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import numpy as np
import torch
import os

def seed(seed, cudnn=False):
    """
    - random.seed is a must
    - seeding must be for each child process
    - seeding cudnn is not necessary and has performance penalty
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # == torch.random.maunal_seed()
    torch.cuda.manual_seed_all(seed) # for multi-GPU.
    torch.cuda.manual_seed(seed)
    if cudnn: # seed cudnn
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    """ for pytorch data loader subprocess """
    worker_seed = torch.initial_seed()
    assert worker_seed < 2**32
    seeding(worker_seed)

class RandState(object):
    """
    # python: https://docs.python.org/3/library/random.html
    random.getstate()
    random.setstate()
    # numpy: https://numpy.org/doc/stable/reference/random/generated/numpy.random.get_state.html
    np.random.get_state()
    np.random.set_state()
    # torch: https://pytorch.org/docs/1.5.0/random.html?highlight=random#module-torch.random
    torch.random.get_rng_state()
    torch.random.set_rng_state()
    # torch.cuda: https://pytorch.org/docs/1.5.0/cuda.html?highlight=random
    torch.cuda.get_rng_state_all()
    torch.cuda.set_rng_state_all()
    # cudnn?
    # pythonhash seed?
    """
    def __init__(self):
        self.get()
    def get(self):
        self.states = []
        self.states.append(random.getstate())
        self.states.append(np.random.get_state())
        self.states.append(torch.random.get_rng_state())
        self.states.append(torch.cuda.get_rng_state_all())
    def set(self):
        random.setstate(self.states[0])
        np.random.set_state(self.states[1])
        torch.random.set_rng_state(self.states[2])
        torch.cuda.set_rng_state_all(self.states[3])
