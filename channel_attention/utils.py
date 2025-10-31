import torch
from torch import nn


def create_conv_layer(
    n_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    groups: int = 1,
) -> nn.Module:
    """Create a convolutional layer based on the number of dimensions."""
    if n_dims == 1:
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
    elif n_dims == 2:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
    else:
        raise ValueError("Only 1D and 2D convolutional layers are supported.")
