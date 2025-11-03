# Channel-Attention <img width="20%" align="right" src="https://github.com/wwhenxuan/S2Generator/blob/main/docs/source/_static/S2Generator_logo.png?raw=true">

A plug-and-play channel attention mechanism module implemented in PyTorch.

<div align="center">

[Installation](#Installation) | [Usage](#Usage) | [Modules](#Modules) | [Blog](https://mp.weixin.qq.com/s/D6O5SBl2RYHdkiinV6UM8w) | [Experiments](#Experiments)
</div>

<div align="center">

[![PyPI version](https://badge.fury.io/py/channel-attention.svg)](https://pypi.org/project/channel-attention/)  ![License](https://img.shields.io/github/license/wwhenxuan/PySDKit) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-blue)](https://pytorch.org/) [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>


## Installation <a id="Installation"></a>

You can install the package via pip:

```bash
pip install channel-attention==0.0.1
```

We only develop and test with PyTorch. Please make sure to install it from [PyTorch official website](https://pytorch.org/get-started/locally/) based on your system configuration.

## Usage <a id="Usage"></a>

**The core of the channel attention mechanism lies in its invariance between input and output.** Therefore, we can easily embed this module into a certain location in a neural network to further improve the model's performance. 

~~~python
import torch
from channel_attention import SEAttention

# 1D Time Series Data with (batch_size, channels, seq_len)
inputs = torch.rand(8, 16, 128)
attn = SEAttention(n_dims=1, n_channels=16, reduction=4)
print(attn(inputs).shape)

# 2D Image Data with (batch_size, channels, height, width)
inputs_2d = torch.rand(8, 16, 64, 64)
attn_2d = SEAttention(n_dims=2, n_channels=16, reduction=4)
print(attn_2d(inputs_2d).shape)
~~~

When the number of input channels is small, the channel attention mechanism is very lightweight and does not significantly increase computational complexity.

## Modules <a id="Modules"></a>

#### 1. [`SEAttention`](https://github.com/wwhenxuan/Channel-Attention/blob/main/channel_attention/squeeze_excitation.py): [[paper]](https://arxiv.org/abs/1709.01507) The Squeeze-and-Excitation Attention with Global Average Pooling and Feed Forward Network.

<div align="center">
    <img width="80%" src="images/SEAttention.png">
</div>

#### 2. [`ChannelAttention`](https://github.com/wwhenxuan/Channel-Attention/blob/main/channel_attention/channel_attention.py): [[paper]](https://arxiv.org/abs/1807.06521) The Channel Attention with Global Average Pooling and Global Max Pooling.

<div align="center">
    <img width="80%" src="images/ChannelAttention.png">
</div>

#### 3. [`SpatialAttention`](https://github.com/wwhenxuan/Channel-Attention/blob/main/channel_attention/spatial_attention.py): [[paper]](https://arxiv.org/abs/1807.06521) The Spatial Attention with Global Average Pooling and Global Max Pooling.

<div align="center">
    <img width="80%" src="images/SpatialAttention.png">
</div>

#### 4. [`ConvBlockAttention`](https://github.com/wwhenxuan/Channel-Attention/blob/main/channel_attention/spatial_attention.py): [[paper]](https://arxiv.org/abs/1807.06521) The Convolutional Block Attention Module (CBAM) combining Channel Attention and Spatial Attention.

<div align="center">
    <img width="80%" src="images/ConvBlockAttention.png">
</div>

## Experiments <a id="Experiments"></a>

