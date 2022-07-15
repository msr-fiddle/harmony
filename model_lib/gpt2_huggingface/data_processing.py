""" copied from run_language_modeling.py """
import logging
import os
import pickle
from typing import Dict, List, Tuple
import numpy as np
import gc
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from .tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "gpt2_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file): # and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

def load_and_cache_examples(file_path, tokenizer, block_size, line_by_line=False):
    # file_path = args.eval_data_file if evaluate else args.train_data_file
    if line_by_line:
        return LineByLineTextDataset(tokenizer, file_path=file_path, block_size=block_size)
    else:
        return TextDataset(tokenizer, file_path=file_path, block_size=block_size)

def is_skip_minibatch(minibatch, defined_minibatch_size, defined_seq_len, verbose=False):
    assert isinstance(minibatch, tuple)
    # batch = minibatch
    # print("\n -- minibatch.shape: {} -- ".format(minibatch.shape))
    for t in minibatch:
        if t.shape[0] != defined_minibatch_size: # last minibatch can be fractional
            if verbose:
                print("[INFO] minibatch's tensor is not defined size: {}".format(t.shape))
            return True
        if len(t.shape) > 1 and t.shape[1] != defined_seq_len:
            if verbose:
                print("[INFO] minibatch's tensor is not defined seq length: {}".format(t.shape))
            return True
    return False

def preprocess_minibatch(minibatch):
    """ Data Processing for model2_gpt2.py and harmony"""
    return minibatch
