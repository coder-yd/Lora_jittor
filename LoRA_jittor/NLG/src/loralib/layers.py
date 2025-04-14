#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import math
from typing import Optional, List

import jittor as jt
from jittor import nn, Module

cnt = 5


class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = jt.nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(jt.nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            r: int = 0,
            lora_alpha: int = 1,
            merge_weights: bool = True,
            **kwargs
    ):
        jt.nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = self.weight.new_zeros((r, num_embeddings))
            self.lora_B = self.weight.new_zeros((embedding_dim, r))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):

        jt.init.gauss_(self.weight)
        if self.padding_idx is not None:
            self.weight[self.padding_idx] = 0

        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            jt.init.zero_(self.lora_A)
            jt.init.gauss_(self.lora_B)

    def train(self, mode: bool = True):
        jt.nn.Embedding.train(self)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True

    def execute(self, x: jt.Var):
        if self.r > 0 and not self.merged:
            result = jt.nn.Embedding.execute(self, x)
            after_A = jt.nn.embedding(
                x, self.lora_A.transpose(0, 1)
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return jt.nn.Embedding.forward(self, x)


class Linear(jt.nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs
    ):
        jt.nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = self.weight.new_zeros((r, in_features))
            self.lora_B = self.weight.new_zeros((out_features, r))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        jt.nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            jt.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            jt.nn.init.zero_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        jt.nn.Linear.train(self)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def execute(self, x: jt.Var):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = jt.nn.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return jt.nn.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(jt.nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        jt.nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = self.weight.new_zeros((r * sum(enable_lora), in_features))
            self.lora_B = self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)
                                                )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.stop_grad()
            # Compute the indices
            self.lora_ind = jt.zeros((out_features,), dtype="bool").view(len(enable_lora), -1)
            self.lora_ind[jt.array(enable_lora), :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight = self.weight.transpose(0, 1)  #这里直接transpose

    def reset_parameters(self):
        def _calculate_fan_in_and_fan_out(weight):
            dimensions = weight.dim()
            if dimensions < 2:
                raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

            num_input_fmaps = weight.size(1)
            num_output_fmaps = weight.size(0)
            receptive_field_size = 1
            if weight.dim() > 2:
                receptive_field_size = weight[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

            return fan_in, fan_out

        # jt.nn.Linear.reset_parameters(self)
        jt.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            jt.init.uniform_(self.bias, -bound, bound)

        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            jt.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            jt.nn.init.zero_(self.lora_B)

    def zero_pad(self, x):
        # x: (2048,1024 )
        # result: (3072,1024 )
        # lora_ind:(3072,)
        result = jt.zeros((len(self.lora_ind), *x.shape[1:]), dtype=x.dtype)
        # print(f"result shape before write: {result.shape}")
        # print(f"lora_ind shape: {self.lora_ind.shape}")

        result[jt.array(self.lora_ind), :] = x
        global cnt
        cnt += 1
        # print("cnt :",cnt)
        # result[jt.where(self.lora_ind)[0], :] = x

        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # conv1d = jt.nn.Conv1d(
        #     in_channels=self.lora_A.shape[0],
        #     out_channels=self.lora_B.shape[0],
        #     kernel_size=1,
        #     groups=sum(self.enable_lora)
        # )
        # conv1d.weight = self.lora_B.unsqueeze(-1)
        # # (2048,1024)
        # delta_w = conv1d(self.lora_A.unsqueeze(0)).squeeze(0)

        con1d = jt.nn.Conv1d(
            in_channels=self.lora_A.unsqueeze(0).shape[1],
            out_channels=self.lora_B.unsqueeze(-1).shape[0],
            kernel_size=self.lora_B.unsqueeze(-1).shape[2],
            groups=sum(self.enable_lora),
            bias=False
        )

        con1d.weight = self.lora_B.unsqueeze(-1)
        delta_w = con1d(self.lora_A.unsqueeze(0)).squeeze(0)

        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        jt.nn.Linear.train(self)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight += self.merge_AB() * self.scaling
                self.merged = True

    def execute(self, x):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return jt.nn.linear(x, T(self.weight), bias=self.bias)
        else:
            result = jt.nn.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += jt.matmul(self.lora_dropout(x), T(self.merge_AB().transpose(0, 1)) * self.scaling)
            return result


class ConvLoRA(jt.nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0.,
                 merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        for name, param in self.conv.named_parameters():
            self.register_parameter(name, param)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            self.lora_B = self.conv.weight.new_zeros((out_channels // self.conv.groups * kernel_size, r * kernel_size))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            jt.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            jt.nn.init.zero_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train()
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def execute(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(jt.nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(jt.nn.Conv1d, *args, **kwargs)


# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(jt.nn.Conv3d, *args, **kwargs)
