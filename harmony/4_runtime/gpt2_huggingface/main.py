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
    ### GPT2
    parser.add_argument("--gpt2_train_file", type=str, default="/data/gpt2/wikitext-2-tokens/wiki.train.tokens")
    parser.add_argument("--gpt2_config_path", type=str, default="../../../model_lib/gpt2_configs/gpt2-xl-config.json",
                        help="always use this to config gpt2")
    parser.add_argument("--gpt2_model", type=str, default="",
                        help="If non-empty (e.g., /workspace/.pretrained_models/GPT2-Medium_DGX02_Seed42), use this directory to load: pytorch_model.bin. Otherwise, train from scratch.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="The initial learning rate.")
    parser.add_argument("--warmup_steps", default=0, type=int, 
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, 
                        help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float, 
    #                     help="Max gradient norm.")

    return parser

import seeding
def seed_worker(worker_id):
    """ for pytorch data loader subprocess (when num_workers > 0)
        NOTE: must be outside of real_dataset to be viewable by DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32 
    seeding.seed(worker_seed, False) # should be fine without seeding cuDNN, i.e., DataLoader is CPU only

def real_dataset(args, minibatch_size, data_workers=0):

    from gpt2_huggingface.tokenization_gpt2 import GPT2Tokenizer
    from gpt2_huggingface.data_processing import load_and_cache_examples, is_skip_minibatch, preprocess_minibatch
    ### Dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # NOTE: we download gpt2 tokenizer for simplicity, because gpt2 tokenizer is the same for download or local, "gpt2" or "gpt2-medium" or "gpt2-xl".
    block_size = tokenizer.max_len
    dataset = load_and_cache_examples(args.gpt2_train_file, tokenizer, block_size, line_by_line=False)
    examples = None
    ### Loader
    from typing import List
    from torch.nn.utils.rnn import pad_sequence
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    data_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=minibatch_size, collate_fn=collate, num_workers=data_workers, worker_init_fn=seed_worker)
    ### Batch Names
    """ batch = input_ids, labels """
    bnames = ODict()
    bnames["is_data"] = [True, False]
    bnames["name"]    = ["input0", "labels"]
    ### Feature Dimension
    fdim = block_size
    ### Copy MiniBatch
    is_copy_minibatch = True

    return data_loader, examples, is_skip_minibatch, preprocess_minibatch, bnames, fdim, is_copy_minibatch

def create_model(args):
    sys.path.append(args.module_dir)
    import importlib; module = importlib.import_module(args.module_name + ".code")

    from gpt2_huggingface.configuration_gpt2 import GPT2Config
    config = GPT2Config.from_json_file(args.gpt2_config_path)
    # print("gpt2config({}): num_hidden_layers={}, hidden_size={}, num_attention_heads={}, max_seq_length={}".format(
    #    os.path.basename(args.gpt2_config_path), config.num_hidden_layers, config.hidden_size, config.num_attention_heads, config.n_positions))
    criterion = torch.nn.CrossEntropyLoss()
    model = module.model(config, criterion)
    if args.gpt2_model != "":
        import utils
        assert os.path.exists(args.gpt2_model)
        utils.load_model(torch.load(os.path.join(args.gpt2_model, "pytorch_model.bin")), model, verbose=True)

    return model

def create_optimizer(args, model):
    optimizer = []
    from gpt2_huggingface.optimization2 import AdamW
    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) == 0:
            optimizer.append(None)
        else:
            param_optimizer = list(vlayer.named_parameters())
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.0},
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            optim = AdamW(optimizer_grouped_parameters, 
                        lr=args.learning_rate, 
                        eps=args.adam_epsilon) 
            optimizer.append(optim)
    return optimizer

from gpt2_huggingface.optimization2 import get_linear_schedule_with_warmup
def get_lr_sched(args, optim):
    return get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_train_steps)

def compute_loss(last_vlayer, Y_named_tensors, Y_names, T_named_tensors):
    output = Y_named_tensors[Y_names[0]][..., :-1, :].contiguous()
    output = output.view(-1, output.size(-1))
    shift_labels = T_named_tensors["labels"][..., 1:].contiguous()
    loss = last_vlayer(output, shift_labels.view(-1))
    return [loss]

def save_model(args, shared_model, update_cnt):
    if args.gpt2_model != "": # need a reference model to save
        import utils
        state_dict = utils.copy_model(shared_model, torch.load(os.path.join(args.gpt2_model, "pytorch_model.bin")))
        utils.save_model(state_dict, args.gpt2_model, args.output_dir, verbose=True)

if __name__ == "__main__":    
    import runt_args
    args = runt_args.initialize_args(custom_args_provider=add_args)

    import runtime
    runtime.run(args, real_dataset, create_model, create_optimizer, get_lr_sched=get_lr_sched, compute_loss=compute_loss, save_model=save_model)
    
