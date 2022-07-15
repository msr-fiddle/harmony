import torch

def is_skip_minibatch(minibatch, defined_minibatch_size, defined_sample_len, verbose=False):
    # images, target = minibatch
    # print("\n -- images, target -- ")
    # for t in minibatch:
    #     print(t.shape)
    # > torch.Size([-1, 3, 224, 224])
    # > torch.Size([-1])
    
    for t in minibatch:
        if t.shape[0] != defined_minibatch_size: # last minibatch can be fractional
            if verbose:
                print("[INFO] minibatch's tensor is not defined size: {}".format(t.shape))
            return True
    return False

def preprocess_minibatch(minibatch):
    """ Data Processing for harmony"""
    return minibatch
