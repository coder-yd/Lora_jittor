#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os, sys
import glob
import random
from collections import Counter, OrderedDict
import numpy as np

import json


import jittor as jt
from jittor.dataset import Dataset,DataLoader


def padding_tokens(tokens, max_seq_length, pad_token, direct, max_context_length=0):

    if max_context_length == 0:
        max_context_length = max_seq_length

    if len(tokens) > max_context_length:
        if direct > 0:
            pad_tokens = tokens[:max_context_length]
        else:
            pad_tokens = tokens[-max_context_length:]
    else:
        pad_tokens = tokens
    token_len = len(pad_tokens)
    pad_tokens = pad_tokens + [pad_token for _ in range(max_seq_length - token_len)]
    return pad_tokens, token_len


class FT_Dataset(Dataset):
    def __init__(self, ft_file, batch_size, max_seq_length,
                 max_eval_length=0, joint_lm=False, prefix_len=0, infix_len=0,
                 prefix_cursor=1000000, infix_cursor=2000000):
        super().__init__()
        self.ft_file = ft_file
        self.ft_samples = self.read_ft_file(ft_file)
        self.batch_size = batch_size
        self.num_examples = len(self.ft_samples)
        self.max_seq_length = max_seq_length
        self.max_eval_length = max_eval_length
        self.rng = random.Random(911)
        self.joint_lm = joint_lm

        self.num_batches = int((self.num_examples + self.batch_size - 1) / self.batch_size)

        self.prefix_len = prefix_len
        self.infix_len = infix_len
        self.prefix_cursor = prefix_cursor
        self.infix_cursor = infix_cursor

    def __len__(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, item):
        if item >= self.num_examples:
            item = self.rng.randint(0, self.num_examples - 1)
        context, completion = self.ft_samples[item]

        pretokens = [self.prefix_cursor + i for i in range(self.prefix_len)]
        intokens = [self.infix_cursor + i for i in range(self.infix_len)]
        conditions = pretokens + context + intokens

        # 输入处理
        input_seq = conditions + completion
        _input, _input_len = padding_tokens(input_seq, self.max_seq_length, 0, 1)

        # 目标处理
        pad_targets = [0] * self.prefix_len + context + [0] * self.infix_len + completion
        _target, _ = padding_tokens(pad_targets[1:], self.max_seq_length, 0, 1)

        # 掩码处理
        if not self.joint_lm:
            mask = [0.0] * (len(conditions) - 1) + [1.0] * (_input_len - len(conditions))
        else:
            mask = [1.0] * (_input_len - 1)
        mask, _ = padding_tokens(mask, self.max_seq_length, 0.0, 1)

        # 查询处理
        _query, _query_len = padding_tokens(conditions, self.max_seq_length, 0, -1,
                                            self.max_seq_length - self.max_eval_length)

        # 转换为Jittor张量
        output = {
            "id": jt.array(item),
            "query": jt.array(_query),
            "query_len": jt.array(_query_len),
            "input": jt.array(_input),
            "target": jt.array(_target),
            "mask": jt.array(mask)
        }

        return output



    def read_ft_file(self, ft_file):
        ft_samples = []
        with open(ft_file, 'r') as reader:
            for line in reader:
                items = json.loads(line.strip())
                context = items['context']
                completion = items['completion']
                ft_samples.append([context, completion])
        return ft_samples


if __name__=="__main__":
    train_data="./data/e2e/train.jsonl"
    train_batch_size=8
    seq_len=256
    valid_data="./data/e2e/valid.jsonl"
    valid_batch_size=4

    train_data = FT_Dataset(
        train_data, train_batch_size, seq_len,
    )

    valid_data = FT_Dataset(
        valid_data, valid_batch_size, seq_len,
    )


    train_loader = DataLoader(
        train_data, batch_size=train_batch_size, num_workers=0,
        shuffle=False, drop_last=True,
    )

    valid_loader = DataLoader(
        valid_data, batch_size=valid_batch_size, num_workers=0,
        shuffle=False, drop_last=False,
    )

    for batch in train_loader:
        print(batch)


