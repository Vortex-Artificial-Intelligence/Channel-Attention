from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from channel_attention.utils import create_conv_layer


class SoftPooling(torch.nn.Module):
    """
    The Soft Pooling Layer for Time Series (1D) or Image (2D) Analysis.
    This layer performs soft pooling by computing a weighted average of the input features,
    where the weights are determined by the exponential of the input features.
    """

    def __init__(
        self,
        n_dims: int,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: Optional[int] = 0,
    ) -> None:
        """
        Initialize the SoftPooling layer.

        :param n_dims: (int) The dimension of input data, either 1 (time series) or 2 (image).
        :param kernel_size: (int) The size of the pooling kernel.
        :param stride: (int) The stride of the pooling operation. If None, it defaults to kernel_size.
        :param padding: (int) The amount of zero-padding added to both sides of the input.
        """
        super().__init__()
        # Store parameters
        self.n_dims = n_dims
        self.kernel_size = kernel_size

        # Create average pooling layer based on input dimension
        self.avg_pool = (
            nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
            if self.n_dims == 2
            else nn.AvgPool1d(kernel_size, stride, padding, count_include_pad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the soft pooling layer.

        :param x: (torch.Tensor) Input tensor of shape (batch_size, channels, height, width) for 2D
                      or (batch_size, channels, seq_len) for 1D.

        :return: (torch.Tensor) Output tensor after soft pooling.
        """
        # Apply soft pooling operation
        x_exp = torch.exp(x)
        x_exp_pool = self.avg_pool(x_exp)

        # Get the weighted average
        x = self.avg_pool(x_exp * x)

        return x / x_exp_pool


class LocalModule(nn.Module):
    """
    Local Module for Channel Attention based on Local Importance.
    This module applies point-wise convolution, soft pooling, and additional convolutional layers
    to compute local attention weights for refining feature representations.

    Reference: "PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution"
    URL: https://arxiv.org/abs/2409.13435
    """

    def __init__(self, n_dims: int, n_channels: int, hidden_channels: int = 16) -> None:
        """
        Initialize the LocalModule.

        :param n_dims: (int) The dimension of input data, either 1 (time series) or 2 (image).
        :param n_channels: (int) The number of input channels of time series or image
        :param hidden_channels: (int) The number of hidden channels in the local attention module.
        """
        super().__init__()
        # Validate the input dimension
        assert n_dims in [1, 2], "The dimension of input data must be either 1 or 2."
        self.n_dims = n_dims

        # Store parameters
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
        """
        Get the spatial dimensions of the input data.

        :param data: (torch.Tensor) Input tensor of shape (batch_size, n_channels, height, width) for 2D.

        :return: (Union[Tuple[int], Tuple[int, int]]) Spatial dimensions of the input data.
        """
        if self.n_dims == 2:
            return data.size(2), data.size(3)
        elif self.n_dims == 1:
            return (data.size(2),)
        else:
            raise ValueError("Invalid number of dimensions.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the local attention module.

        :param x: (torch.Tensor) Input tensor of shape (batch_size, n_channels, height, width) for 2D or (batch_size, n_channels, seq_len) for 1D.

        :return: (torch.Tensor) Output tensor of shape (batch_size, n_channels, height, width) for 2D or (batch_size, n_channels, seq_len) for 1D.
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
    """
    Local Module for Channel Attention based on Local Importance (Speed Version).
    This module applies convolutional layers to compute local attention weights for refining feature representations.
    This version is optimized for speed.
    """

    def __init__(self, n_dims: int, n_channels: int) -> None:
        """
        Initialize the LocalModuleSpeed.

        :param n_dims: (int) The dimension of input data, either 1 (time series) or 2 (image).
        :param n_channels: (int) The number of input channels of time series or image.
        """
        super().__init__()

        # Store parameters
        self.n_dims = n_dims
        self.n_channels = n_channels

        # Create convolutional layers for local attention
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
        """
        Forward pass through the local attention module.

        :param x: (torch.Tensor) Input tensor of shape (batch_size, n_channels, height, width) for 2D or (batch_size, n_channels, seq_len) for 1D.

        :return: (torch.Tensor) Output tensor of shape (batch_size, n_channels, height, width) for 2D or (batch_size, n_channels, seq_len) for 1D.
        """
        # interpolate the heat map
        w = self.conv(x)

        return w


class LocalAttention(nn.Module):
    """
    Attention based on local importance for Time Series (1D) or Image (2D) Analysis.
    This module applies local attention weights to refine feature representations.

    Reference: "PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution"
    URL: https://arxiv.org/abs/2409.13435
    """

    def __init__(
        self,
        n_dims: int,
        n_channels: int,
        hidden_channels: int = 16,
        speed: Optional[bool] = False,
    ) -> None:
        """
        Initialize the LocalAttention module.

        :param n_dims: (int) The dimension of input data, either 1 (time series) or 2 (image).
        :param n_channels: (int) The number of input channels of time series or image
        :param hidden_channels: (int) The number of hidden channels in the local attention module.
        :param speed: (bool) Whether to use the speed-optimized version of the local attention module.
        """
        super().__init__()

        # Validate the input dimension
        assert n_dims in [1, 2], "The dimension of input data must be either 1 or 2."

        # Store parameters
        self.n_dims = n_dims
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.speed = speed

        # Initialize the local attention body
        self.body = (
            LocalModule(
                n_dims=n_dims, n_channels=n_channels, hidden_channels=hidden_channels
            )
            if not speed
            else LocalModuleSpeed(n_dims=n_dims, n_channels=n_channels)
        )

        # Gate activation
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the local attention module.
        """
        # interpolate the heat map
        g = self.gate(x[:, :1])
        w = self.body(x=x)

        return x * w * g  # (w + g) #self.gate(x, w)
