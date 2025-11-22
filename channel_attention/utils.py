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
    """
    Create a convolutional layer based on the number of dimensions.
    
    :param n_dims: The number of dimensions (1 or 2).
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel_size: Size of the convolutional kernel.
    :param stride: Stride of the convolution. Default is 1.
    :param padding: Padding added to both sides of the input. Default is 0.
    :param bias: If True, adds a learnable bias to the output. Default is True.
    :param groups: Number of blocked connections from input channels to output channels. Default is 1.
    
    :return: A convolutional layer (nn.Conv1d or nn.Conv2d).
    """
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
