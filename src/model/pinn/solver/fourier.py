import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, num_features: int, sigma: float = 1.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.sigma = sigma
        B = torch.randn(num_features, input_dim) * sigma
        self.register_buffer('B', B)

    @property
    def out_dim(self) -> int:
        return 2 * self.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def extra_repr(self) -> str:
        return f'in_features={self.input_dim}, out_features={self.out_dim}, sigma={self.sigma}'
