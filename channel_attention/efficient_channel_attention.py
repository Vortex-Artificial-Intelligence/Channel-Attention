from typing import Optional

import torch
from torch import nn


def _transform_1d(data: torch.Tensor) -> torch.Tensor:
    """Transform 1D tensor by transposing the last two dimensions."""
    return data.transpose(-1, -2)


def _transform_2d(data: torch.Tensor) -> torch.Tensor:
    """Transform 2D tensor by squeezing the last dimension and transposing."""
    return data.squeeze(-1).transpose(-1, -2)


def _invert_transform_1d(data: torch.Tensor) -> torch.Tensor:
    """Invert transform for 1D tensor by transposing the last two dimensions."""
    return data.transpose(-1, -2)


def _invert_transform_2d(data: torch.Tensor) -> torch.Tensor:
    """Invert transform for 2D tensor by transposing and unsqueezing the last dimension."""
    return data.transpose(-1, -2).unsqueeze(-1)


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    This module captures cross-channel interaction efficiently without dimensionality reduction.

    Reference: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks

    URL: https://arxiv.org/abs/1910.03151
    """

    def __init__(
        self, n_dims: int, kernel_size: Optional[int] = 3, bias: Optional[bool] = False
    ) -> None:
        """
        Initialize EfficientChannelAttention module.

        :param n_dims: Number of dimensions of the input data (1 for time series, 2 for images).
        :param kernel_size: Size of the convolutional kernel (must be odd).
        :param bias: Whether to include a bias term in the convolutional layer.
        """
        super().__init__()
        # Check n_dims is valid for time series or image data
        assert n_dims in [
            1,
            2,
        ], "Only 1D and 2D `EfficientChannelAttention` are supported."
        self.n_dims = n_dims

        # Global average pooling layer to squeeze the spatial dimensions
        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1) if n_dims == 2 else nn.AdaptiveAvgPool1d(1)
        )

        # Convolutional layer for capturing cross-channel interaction
        assert kernel_size % 2 == 1, "The kernel size must be odd."
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

        # Create the transformation functions
        if n_dims == 1:
            self.transform = _transform_1d
            self.invert_transform = _invert_transform_1d
        else:
            self.transform = _transform_2d
            self.invert_transform = _invert_transform_2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EfficientChannelAttention module.

        :param x: (torch.Tensor) Input tensor of shape (batch_size, n_channels, length) for 1D
                  or (batch_size, n_channels, height, width) for 2D.

        :return: (torch.Tensor) Output tensor of the same shape as input.
        """
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.transform(y)
        y = self.conv(y)
        y = self.invert_transform(y)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
