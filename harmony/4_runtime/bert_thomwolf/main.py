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
    ### BERT
    parser.add_argument("--bert_task_name", type=str, default="MRPC",
                        help="The name of the task to train.")
    parser.add_argument("--bert_data_dir", type=str, default="/data/glue/MRPC",
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_seq_length", type=int, default=512, 
                        help="data sample length")
    parser.add_argument("--bert_config_path", type=str, 
                        default="../../../model_lib/bert_configs/bert-large-uncased.json",
                        help="always use this to config bert")
    parser.add_argument("--bert_model", type=str,
                        default="/workspace/.pretrained_models/BERT-Large-Uncased",
                        help="If non-empty, use this directory to load: pytorch_model.bin. Otherwise, train from scratch")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="The initial learning rate.")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, 
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    return parser

import seeding
def seed_worker(worker_id):
    """ for pytorch data loader subprocess (when num_workers > 0)
        NOTE: must be outside of real_dataset to be viewable by DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32 
    seeding.seed(worker_seed, False) # should be fine without seeding cuDNN, i.e., DataLoader is CPU only

def real_dataset(args, minibatch_size, data_workers=0):

    from bert_thomwolf.tokenization import BertTokenizer
    from bert_thomwolf.data_processing import make_dataset, is_skip_minibatch, preprocess_minibatch
    ### Dataset
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True) 
    # NOTE: we download bert tokenizer for simplicity, because bert tokenizer is the same for download or local, bert-base or bert-large.
    dataset, examples = make_dataset(args.bert_task_name.lower(), args.bert_data_dir, args.bert_seq_length, tokenizer, kind='train') 
    ### Loader
    data_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=minibatch_size, num_workers=data_workers, worker_init_fn=seed_worker)
    ### Batch Names
    """ batch = input_ids, input_mask, segment_ids, label_ids """
    bnames = ODict() 
    bnames["is_data"] = [True, True, True, False]
    bnames["name"]    = ["input0", "input2", "input1", "labels"]
    ### Feature Dimension
    fdim = args.bert_seq_length
    ### Copy MiniBatch
    is_copy_minibatch = False

    return data_loader, examples, is_skip_minibatch, preprocess_minibatch, bnames, fdim, is_copy_minibatch

def create_model(args):
    sys.path.append(args.module_dir)
    import importlib; module = importlib.import_module(args.module_name + ".code")

    from bert_thomwolf.modeling2 import BertConfig
    config = BertConfig.from_json_file(args.bert_config_path)
    criterion = torch.nn.CrossEntropyLoss()
    model = module.model(config, criterion)
    if args.bert_model != "":
        import utils
        assert os.path.exists(args.bert_model)
        utils.load_model(torch.load(os.path.join(args.bert_model, "pytorch_model.bin")), model, verbose=True)

    return model

def create_optimizer(args, model):
    optimizer = []
    from bert_thomwolf.optimization2 import BertAdam
    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) == 0:
            optimizer.append(None)
        else:
            param_optimizer = list(vlayer.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
            optim = BertAdam(optimizer_grouped_parameters, 
                            lr=args.learning_rate, 
                            warmup=args.warmup_proportion, 
                            t_total=args.num_train_steps)
            optimizer.append(optim)
    return optimizer

def get_train_steps(args, examples, minibatch_size):
    return int(len(examples) / minibatch_size * args.num_epochs)

def compute_loss(last_vlayer, Y_named_tensors, Y_names, T_named_tensors):
    logits = Y_named_tensors[Y_names[0]]
    loss = last_vlayer(logits.view(-1, 2), T_named_tensors["labels"].view(-1))
    return [loss]

def save_model(args, shared_model, update_cnt):
    if args.bert_model != "": # need a reference model to save
        import utils
        state_dict = utils.copy_model(shared_model, torch.load(os.path.join(args.bert_model, "pytorch_model.bin")))
        utils.save_model(state_dict, args.bert_model, args.output_dir, verbose=True)
        
        with open(os.path.join(args.output_dir, "hyper_params.json"), "w") as f:
            json.dump( { 'update_cnt': update_cnt }, f )

if __name__ == "__main__":    
    import runt_args
    args = runt_args.initialize_args(custom_args_provider=add_args)

    import runtime
    runtime.run(args, real_dataset, create_model, create_optimizer, get_train_steps=get_train_steps, compute_loss=compute_loss, save_model=save_model)
    
