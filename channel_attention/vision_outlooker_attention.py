import torch
from torch import nn


class VisionOutlookerAttention(nn.Module):
    """
    The Vision Outlooker Attention Module for Visual Recognition Tasks.
    This module implements the attention mechanism for Time Series (1D) and Image (2D) data.

    References: "VOLO: Vision Outlooker for Visual Recognition" by Li Yuan, Qibin Hou, et al.

    URL: https://arxiv.org/abs/2106.13112
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
