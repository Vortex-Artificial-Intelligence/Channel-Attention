from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# 论文：FECAM: Frequency Enhanced Channel Attention Mechanism for Time Series Forecasting
# 论文地址：https://arxiv.org/abs/2212.01209

try:
    from torch.fft import rfft, irfft
except ImportError:

    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r

    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = rfft(v, 1)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


class FreqEnhancedAttention(nn.Module):
    def __init__(
        self,
        n_dims: int,
        n_channels: int,
        reduction: Optional[int] = 2,
        dropout: Optional[float] = 0.1,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid(),
        )

        self.dct_norm = nn.LayerNorm(n_channels, eps=1e-6)  # for lstm on length-wise

    def forward(self, x):
        b, c, l = x.size()  # (B,C,L) (32,96,512)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)

        lr_weight = F.normalize()
        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(lr_weight)
        lr_weight = self.dct_norm(lr_weight)

        return x * lr_weight


if __name__ == "__main__":
    input = torch.rand(8, 7, 96)
    block = FreqEnhancedAttention(96)
    result = block(input)
    print("input_tensor.shape:", input.shape)
    print("result.shape:", result.shape)
