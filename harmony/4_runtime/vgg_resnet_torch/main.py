# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import json
from collections import OrderedDict as ODict

import torch
from torch.utils.data import DataLoader, RandomSampler

import sys
sys.path.append("../../../model_lib")
sys.path.append("../../../util_lib")

sys.path.append("../../4_runtime")
sys.path.append("../../3_scheduler")
sys.path.append("../../2_profiler")

def add_args(parser):
    ### VGG, ResNet
    parser.add_argument('--imagenet_dir', type=str, default="/data/imagenet",
                        help="directory to imagenet dataset")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="The initial learning rate.")

    return parser

import seeding
def seed_worker(worker_id):
    """ for pytorch data loader subprocess (when num_workers > 0)
        NOTE: must be outside of real_dataset to be viewable by DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32 
    seeding.seed(worker_seed, False) # should be fine without seeding cuDNN, i.e., DataLoader is CPU only

def real_dataset(args, minibatch_size, data_workers=0):

    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from vgg_resnet_torch.data_processing import is_skip_minibatch, preprocess_minibatch
    ### Dataset
    dataset = datasets.ImageFolder(
        os.path.join(args.imagenet_dir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ]))
    examples = None
    ### Loader
    data_loader = DataLoader(dataset, sampler=None, batch_size=minibatch_size, num_workers=data_workers, worker_init_fn=seed_worker, shuffle=True)
    ### Batch Names
    """ batch = images, target """
    bnames = ODict()
    bnames["is_data"] = [True, False]
    bnames["name"]    = ["input0", "target"]
    ### Feature Dimension
    fdim = (3,224,224)
    ### Copy MiniBatch
    is_copy_minibatch = False

    return data_loader, examples, is_skip_minibatch, preprocess_minibatch, bnames, fdim, is_copy_minibatch

def create_model(args):
    sys.path.append(args.module_dir)
    import importlib; module = importlib.import_module(args.module_name + ".code")

    criterion = torch.nn.CrossEntropyLoss()
    model = module.model(criterion)

    return model

def create_optimizer(args, model):
    optimizer = []
    from torch.optim import SGD
    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) == 0:
            optimizer.append(None)
        else:
            optim = SGD(vlayer.parameters(), 
                        lr=args.learning_rate, 
                        momentum=0.9, 
                        weight_decay=1e-4)
            optimizer.append(optim)
    return optimizer

if __name__ == "__main__":    
    import runt_args
    args = runt_args.initialize_args(custom_args_provider=add_args)

    import runtime
    runtime.run(args, real_dataset, create_model, create_optimizer)
    
