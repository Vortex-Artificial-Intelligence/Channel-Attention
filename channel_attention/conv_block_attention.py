from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from channel_attention.spatial_attention import SpatialAttention


class ChannelAttention(nn.Module):
    """
    The Channel Attention Module for Time Series (1D) or Image (2D) Analysis.
    This module focuses on 'what' is an informative feature by applying attention weights across channel dimensions.

    `Attention Here`: The linear layer is shared for both `max-pooled` and `avg-pooled` features.

    Reference: "CBAM: Convolutional Block Attention Module" by Sanghyun Woo, et al.

    URL: https://arxiv.org/abs/1807.06521
    """

    def __init__(
        self, n_dims: int, n_channels: int, reduction: Optional[int] = 4
    ) -> None:
        """
        Initialize the Channel Attention Module.

        :param n_dims: (int) The dimension of input data, either 1 (time series) or 2 (image).
        :param n_channels: (int) The number of input channels of time series or image
        :param reduction: (int) The reduction ratio for the intermediate layer in the channel attention block.
        """
        super().__init__()

        # Validate the input dimension
        assert n_dims in [1, 2], "The dimension of input data must be either 1 or 2."

        # Store parameters
        self.n_channels = n_channels
        self.reduction = reduction

        # Fully connected layers for the excitation operation
        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.n_channels,
                out_features=self.n_channels // self.reduction,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.n_channels // self.reduction,
                out_features=self.n_channels,
                bias=True,
            ),
        )

        # View shape for reshaping the excitation output
        self.view_shape = (1, 1) if n_dims == 2 else (1,)

        # The adaptive pooling layers for squeezing operation
        self.adaptive_max_pool = (
            nn.AdaptiveMaxPool2d(1) if n_dims == 2 else nn.AdaptiveMaxPool1d(1)
        )
        self.adaptive_avg_pool = (
            nn.AdaptiveAvgPool2d(1) if n_dims == 2 else nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Channel Attention Module.

        :param x: (torch.Tensor)
                  1D Time Series: Input tensor of shape (batch_size, channels, seq_len);
                  2D Image: Input tensor of shape (batch_size, channels, height, width).

        :return: (torch.Tensor) Output tensor of the same shape as input.
        """
        # Compute max and average pooling along the channel dimension
        max = self.adaptive_max_pool(x)
        avg = self.adaptive_avg_pool(x)

        # Apply the fully connected layers for excitation operation
        batch_size, n_channels = x.size()[:2]
        linear_max = self.linear(max.view(batch_size, n_channels)).view(
            batch_size, n_channels, *self.view_shape
        )
        linear_avg = self.linear(avg.view(batch_size, n_channels)).view(
            batch_size, n_channels, *self.view_shape
        )

        # Combine and apply sigmoid activation to get attention map
        output = linear_max + linear_avg

        # Apply sigmoid activation and scale the input feature map
        output = F.sigmoid(output) * x

        return output


class ConvBlockAttention(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) for Time Series (1D) or Image (2D) Analysis.
    This module sequentially applies Channel Attention and Spatial Attention to refine feature representations.

    Reference: "CBAM: Convolutional Block Attention Module" by Sanghyun Woo, et al.

    URL: https://arxiv.org/abs/1807.06521

    Also see: `ChannelAttention` and `SpatialAttention` classes.
    """

    def __init__(
        self,
        n_dims: int,
        n_channels: int,
        reduction: Optional[int] = 4,
        kernel_size: Optional[int] = 7,
    ) -> None:
        """
        Initialize the Convolutional Block Attention Module.

        :param n_dims: (int) The dimension of input data, either 1 (time series) or 2 (image).
        :param n_channels: (int) The number of input channels of time series or image.
        :param reduction: (int) The reduction ratio for the intermediate layer in the channel attention block.
        :param kernel_size: (int) The size of the convolutional kernel in the spatial attention block. Must be odd to maintain spatial dimensions.
        """
        super().__init__()

        # Initialize Channel Attention and Spatial Attention modules
        self.channel_attention = ChannelAttention(
            n_dims=n_dims, n_channels=n_channels, reduction=reduction
        )
        self.spatial_attention = SpatialAttention(
            n_dims=n_dims, kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor, with_residual: bool = True) -> torch.Tensor:
        """
        Forward pass for the Convolutional Block Attention Module.

        :param x: (torch.Tensor)
                  1D Time Series: Input tensor of shape (batch_size, channels, seq_len);
                  2D Image: Input tensor of shape (batch_size, channels, height, width).
        :param with_residual: (bool) Whether to include a residual connection from input to output.

        :return: (torch.Tensor) Output tensor of the same shape as input.
        """
        output = self.channel_attention(x)
        output = self.spatial_attention(output)

        if with_residual:
            return output + x
        return output


class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x
