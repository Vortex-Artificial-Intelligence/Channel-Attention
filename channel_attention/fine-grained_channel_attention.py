import math
import torch
from torch import nn

# 论文：Unsupervised Bidirectional Contrastive Reconstruction and Adaptive Fine-Grained Channel Attention Networks for image dehazing
# 论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0893608024002387


class Mix(nn.Module):
    def __init__(self, m: float = -0.80) -> None:
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        mix_factor = self.mix_block(self.w)
        out = feature1 * mix_factor.expand_as(feature1) + feature2 * (
            1 - mix_factor.expand_as(feature2)
        )
        return out


#
class FineGrainedCAttention(nn.Module):
    """
    Adaptive Fine-Grained Channel Attention (FCA) module.
    This module captures fine-grained cross-channel interaction adaptively.

    Reference: Unsupervised Bidirectional Contrastive Reconstruction and Adaptive Fine-Grained Channel Attention Networks for image dehazing

    URL: https://www.sciencedirect.com/science/article/abs/pii/S0893608024002387
    """

    def __init__(self, n_channels: int, b: float = 1.0, gamma: float = 2.0) -> None:
        super(FineGrainedCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        # 一维卷积
        t = int(abs((math.log(n_channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(n_channels, n_channels, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)  # (1,1,64)
        out1 = (
            torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        )  # (1,64,1,1)
        out1 = self.sigmoid(out1)
        out2 = (
            torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = (
            self.conv1(out.squeeze(-1).transpose(-1, -2))
            .transpose(-1, -2)
            .unsqueeze(-1)
        )
        out = self.sigmoid(out)

        return input * out


if __name__ == "__main__":
    input = torch.rand(1, 64, 256, 256)
    block = FineGrainedCAttention(n_channels=64)
    output = block(input)
    print(output.size())
