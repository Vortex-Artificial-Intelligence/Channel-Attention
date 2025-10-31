from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def create_conv_layer(
    n_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
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
        )
    elif n_dims == 2:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    else:
        raise ValueError("Only 1D and 2D convolutional layers are supported.")


class SoftPooling(torch.nn.Module):
    def __init__(
        self,
        n_dims: int,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: Optional[int] = 0,
    ) -> None:
        super().__init__()

        self.n_dims = n_dims

        self.avg_pool = (
            nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
            if self.n_dims == 2
            else nn.AvgPool1d(kernel_size, stride, padding, count_include_pad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = torch.exp(x)
        x_exp_pool = self.avg_pool(x_exp)
        x = self.avg_pool(x_exp * x)
        return x / x_exp_pool


class LocalModule(nn.Module):
    def __init__(self, n_dims: int, n_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()

        self.n_dims = n_dims
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels

        # Create point-wise convolutional layer
        self.point_wise = create_conv_layer(
            n_dims=n_dims,
            in_channels=n_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )

        # Create soft pooling layer
        self.soft_pooling = SoftPooling(n_dims=n_dims, kernel_size=7, stride=3)

        # Create convolutional layers for local attention
        self.conv = nn.Sequential(
            create_conv_layer(
                n_dims=n_dims,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            create_conv_layer(
                n_dims=n_dims,
                in_channels=hidden_channels,
                out_channels=n_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # Gate activation
            nn.Sigmoid(),
        )

        self.mode = "linear" if n_dims == 1 else "bilinear"

    def get_data_size(self, data: torch.Tensor) -> Union[Tuple[int], Tuple[int, int]]:
        """Get the spatial dimensions of the input data."""
        if self.n_dims == 2:
            return data.size(2), data.size(3)
        elif self.n_dims == 1:
            return (data.size(2),)
        else:
            raise ValueError("Invalid number of dimensions.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the local attention module.

        :param x: (torch.Tensor) Input tensor of shape (batch_size, n_channels, height, width) for 2D.

        :return: (torch.Tensor) Output tensor of shape (batch_size, n_channels, height, width).
        """
        # Forward pass through the local attention squeeze-and-excitation
        y = self.point_wise(x)
        y = self.soft_pooling(y)
        y = self.conv(y)

        # interpolate the heat map
        w = F.interpolate(
            input=y, size=self.get_data_size(x), mode=self.mode, align_corners=False
        )

        return w


class LocalModuleSpeed(nn.Module):
    def __init__(self, n_dims: int, n_channels: int) -> None:
        super().__init__()

        self.n_dims = n_dims
        self.n_channels = n_channels

        self.conv = nn.Sequential(
            create_conv_layer(
                n_dims=n_dims,
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward"""
        # interpolate the heat map
        w = self.conv(x)

        return w


class LocalAttention(nn.Module):
    """attention based on local importance"""

    def __init__(
        self, n_dims: int, n_channels: int, hidden_channels: int = 16, speed: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.n_dims = n_dims
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.speed = speed

        self.body = (
            LocalModule(
                n_dims=n_dims, n_channels=n_channels, hidden_channels=hidden_channels
            )
            if not speed
            else LocalModuleSpeed(n_dims=n_dims, n_channels=n_channels)
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):
        """forward"""
        # interpolate the heat map
        g = self.gate(x[:, :1])
        w = self.body(x=x)

        return x * w * g  # (w + g) #self.gate(x, w)
