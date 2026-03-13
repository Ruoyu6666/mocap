"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
from __future__ import division

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numbers


# when each instance is
class StartConv(nn.Module):
    def __init__(self, d_in: int, d_out: int, kernel_size: int=3, stride: int=1, padding: int=0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_in,     # feature dims
            out_channels=d_out,   # new feature channels
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.d_out = d_out

    def forward(self, x):
        B, N, T, D = x.shape                        # (batch_size, num_instance, len_per_patch, dimension)
        x = x.reshape(B * N, T, D).permute(0, 2, 1) # (B * N, D, T)
        x = torch.relu(self.conv(x))                # (B*N, hidden, T)
        x = x.mean(dim=-1)                          # Global average pooling over time: (B*N, hidden)
        return x.reshape(B, N, self.d_out)          # reshape back

    


class ConvBlock(nn.Module):
    """
    Convolutional module: Conv1D + BatchNorm + (optional) ReLU.
    """
    def __init__(
        self, 
        n_in_channels: int,  n_out_channels: int, kernel_size: int, 
        padding_mode: str = "replicate", include_relu: bool = True,
    ) :
        super().__init__()
        
        layers = [
            nn.Conv1d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=n_out_channels),
        ]
        
        if include_relu:
            layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        return out





def manual_pad(x: torch.Tensor, min_length: int) -> torch.Tensor:
    """
    Manual padding function that pads x to a minimum length with replicate padding.
    PyTorch padding complains if x is too short relative to the desired pad size, hence this function.

    :param x: Input tensor to be padded.
    :param min_length: Length to which the tensor will be padded.
    :return: Padded tensor of length min_length.
    """
    # Calculate amount of padding required
    pad_amount = min_length - x.shape[-1]
    # Split either side
    pad_left = pad_amount // 2          # Pad left (replicate first value)
    pad_right = pad_amount - pad_left   # Pad right (replicate last value)
    # pad_x = F.pad(x, [pad_left, 0], mode="constant", value=x[:, :, 0].item())
    pad_x = F.pad(x, [pad_left, 0], mode="constant", value=0.)
    # pad_x = F.pad(pad_x, [0, pad_right], mode="constant", value=x[:, :, -1].item())
    pad_x = F.pad(pad_x, [0, pad_right], mode="constant", value=0.)
    return pad_x




class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)