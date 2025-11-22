from typing import Optional

import torch
import torch.nn as nn


class SimpleAttentionModule(torch.nn.Module):
    """
    A Simple, Parameter-Free Attention Module for Convolutional
    Module for Time Series (1D) and Image (2D) Data.

    Refernces: "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks" by Lingxiao Yang, Ru-Yuan Zhang, et al.

    URL: https://proceedings.mlr.press/v139/yang21o.html
    """

    def __init__(
        self, n_dims: int, in_channels: int = None, e_lambda: Optional[float] = 1e-4
    ) -> None:
        """ """
        super().__init__()

        n_dims = n_dims

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "lambda=%f)" % self.e_lambda
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )

        return x * self.activaton(y)
