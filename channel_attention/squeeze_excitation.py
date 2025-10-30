from typing import Optional

import torch
from torch import nn


class SEAttention(nn.Module):
    """
    The Squeeze-and-Excitation Attention for Time Series (1D) or Image (2D) Analysis.
    This module adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.

    Reference: "Squeeze-and-Excitation Networks" by Jie Hu, Li Shen, et al.

    URL: https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        n_dims: int,
        n_channels: int,
        reduction: Optional[int] = 4,
        bias: bool = False,
    ) -> None:
        """
        1D Squeeze-and-Excitation Attention for Time Series Analysis or
        2D Squeeze-and-Excitation Attention for Image Analysis.

        :param n_dims: (int) The dimension of input data, either 1 (time series) or 2 (image).
        :param n_channels: (int) The number of input channels of time series data.
        :param reduction: (int) The reduction ratio for the intermediate layer in the SE block.
        :param bias: (bool) Whether to include bias terms in the linear layers.
        """
        super().__init__()

        # Validate the input dimension
        assert n_dims in [1, 2], "The dimension of input data must be either 1 or 2."

        # The dimension of inputs data
        self.n_dims = n_dims

        # Global average pooling layer to squeeze the spatial dimensions
        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1) if n_dims == 2 else nn.AdaptiveAvgPool1d(1)
        )

        # Fully connected layers for the excitation operation
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=bias),
            nn.Sigmoid(),
        )

        # View shape for reshaping the excitation output
        self.view_shape = (1, 1) if n_dims == 2 else (1,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SEAttention module.

        :param x: (torch.Tensor)
                  1D Time Series: Input tensor of shape (batch_size, channels, seq_len);
                  2D Image: Input tensor of shape (batch_size, channels, height, width).

        :return: (torch.Tensor) Output tensor of the same shape as input
        """
        # Get the batch size, number of channels
        batch_size, channels = x.size()[:2]

        # Perform the Squeeze operation
        y = self.avg_pool(x).view(batch_size, channels)

        # Perform the Excitation operation
        y = self.fc(y).view(batch_size, channels, *self.view_shape)

        # Scale the input tensor with the recalibrated weights
        return x * y.expand_as(x)
