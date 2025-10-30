from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    The Spatial Attention Module for Time Series (1D) or Image (2D) Analysis.
    This module focuses on 'where' is an informative part by applying attention weights across spatial dimensions.

    Reference: "CBAM: Convolutional Block Attention Module" by Sanghyun Woo, et al.

    URL: https://arxiv.org/abs/1807.06521
    """

    def __init__(
        self, n_dims: int, kernel_size: Optional[int] = 7, bias: Optional[bool] = False
    ) -> None:
        """
        Initialize the Spatial Attention Module.

        :param n_dims: (int) The dimension of input data, either 1 (time series) or 2 (image).
        :param kernel_size: (int) The size of the convolutional kernel. Must be
                                odd to maintain the spatial dimensions.
        :param bias: (bool) Whether to include bias term in the convolutional layer.
        """
        super().__init__()

        # Validate the input dimension and kernel size
        assert n_dims in [1, 2], "The dimension of input data must be either 1 or 2."
        assert kernel_size % 2 == 1, "The kernel size must be odd."

        # Store parameters
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        self.bias = bias

        # parameters for 1D conv
        parameters = {
            "in_channels": 2,
            "out_channels": 1,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": 1,
            "bias": self.bias,
        }

        # Define convolutional layer based on input dimension
        self.conv = nn.Conv1d(**parameters) if n_dims == 1 else nn.Conv2d(**parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Spatial Attention Module.

        :param x: (torch.Tensor)
                  1D Time Series: Input tensor of shape (batch_size, channels, seq_len);
                  2D Image: Input tensor of shape (batch_size, channels, height, width).

        :return: (torch.Tensor) Output tensor of the same shape as input.
        """
        # Compute max and average pooling along the channel dimension
        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)

        # Concatenate max and average pooled features
        concat = torch.cat((max, avg), dim=1)

        # Apply convolution and sigmoid activation to get attention map
        output = self.conv(concat)
        output = F.sigmoid(output) * x

        return output
