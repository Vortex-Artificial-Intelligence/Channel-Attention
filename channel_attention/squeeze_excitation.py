from typing import Optional

import torch
from torch import nn

from channel_attention.utils import create_conv_layer


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


class MultiSEAttention(nn.Module):
    """
    Multi-Branch Squeeze-and-Excitation Attention Module for Time Series (1D) or Image (2D) Analysis.
    This module enhances the representational power of the standard SE block by incorporating multiple branches and adaptive style assignment.
    """

    def __init__(self, n_dims: int, n_channels: int, reduction: int = 4, n_branches: int = 3) -> None:
        """
        Multi-Branch Squeeze-and-Excitation Attention Module for Time Series (1D) or Image (2D) Analysis.

        :param n_dims: (int) The dimension of input data, either 1 (time series) or 2 (image).
        :param n_channels: (int) The number of input channels of time series data.
        :param reduction: (int) The reduction ratio for the intermediate layer in the SE block.
        :param n_branches: (int) The number of branches in the multi-branch SE module.
        """
        super(MultiSEAttention, self).__init__()

        # Dimension assertion
        assert n_dims in [1, 2], "The dimension of input data must be either 1 or 2."
        self.n_dims = n_dims

        # Create the average pooling layer and activation function
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if n_dims == 2 else nn.AdaptiveAvgPool1d(1)
        self.activation = nn.Sigmoid()

        # Store the reduction ratio, number of branches, and number of channels
        self.reduction = reduction
        self.n_branches = n_branches
        self.n_channels = n_channels
        new_channels = n_channels * n_branches

        # Layers for multi-branch excitation
        self.fc = nn.Sequential(create_conv_layer(n_dims=n_dims, in_channels=new_channels, out_channels=new_channels // self.reduction, kernel_size=1, bias=True, groups=n_branches),
                                 nn.ReLU(inplace=True),
                                    create_conv_layer(n_dims=n_dims, in_channels=new_channels // self.reduction, out_channels=new_channels, kernel_size=1, bias=True, groups=n_branches))

        # Style assignment layer
        self.style_assigner = nn.Linear(n_channels, n_branches, bias=False)

        # Repeat size for reshaping the output
        self.repeat_size = (1, 1) if n_dims == 2 else (1,)
    
    def _style_assignment(self, channel_mean: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Assign styles to each channel based on the channel mean.

        :param channel_mean: (torch.Tensor) The mean values of each channel, shape (batch_size, n_channels, 1, 1).
        :param batch_size: (int) The batch size of the input tensor.

        :return: (torch.Tensor) Style assignment probabilities for each branch, shape (batch_size, n_branches).
        """
        style_assignment = self.style_assigner(channel_mean.view(batch_size, -1))
        style_assignment = nn.functional.softmax(style_assignment, dim=1)
        return style_assignment

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiSEAttention module.

        :param x: (torch.Tensor)
                  1D Time Series: Input tensor of shape (batch_size, channels, seq_len);
                  2D Image: Input tensor of shape (batch_size, channels, height, width).

        :return: (torch.Tensor) Output tensor of the same shape as input.
        """
        # Apply global average pooling
        avg_y = self.avg_pool(x)
        batch_size, n_channels = avg_y.shape[:2]

        # Perform style assignment
        style_assignment = self._style_assignment(avg_y, batch_size=batch_size) # B x N

        # Multi-branch excitation
        avg_y = avg_y.repeat(1, self.n_branches, *self.repeat_size)

        # [batch_size, n_branches * n_channels, 1, 1]
        z = self.fc(avg_y)

        # Apply style assignment
        style_assignment = style_assignment.repeat_interleave(n_channels, dim=1)
        if self.n_dims == 1:
            z = z * style_assignment[:, :, None]
        else:
            z = z * style_assignment[:, :, None, None]

        # [batch_size, n_channels, 1, 1]
        z = torch.sum(z.view(batch_size, self.n_branches, n_channels, *self.repeat_size), dim=1) # B x C x 1 x 1
        z = self.activation(z)

        return x * z