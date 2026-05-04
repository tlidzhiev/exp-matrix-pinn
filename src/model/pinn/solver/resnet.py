from typing import Literal

import torch
import torch.nn as nn

from ...utils import get_activation, initialize_weights
from .fourier import FourierFeatures


class ResBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        activation: Literal['relu', 'tanh', 'silu', 'gelu'] = 'relu',
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.block(h)


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        activation: Literal['relu', 'tanh', 'silu', 'gelu'] = 'tanh',
        init_mode: Literal['normal', 'uniform'] | None = 'normal',
        fourier_features: int | None = None,
        fourier_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        if fourier_features is not None:
            self.fourier = FourierFeatures(1, fourier_features, fourier_sigma)
            t_dim = self.fourier.out_dim
        else:
            self.fourier = None
            t_dim = 1

        self.input_proj = nn.Linear(t_dim + (input_dim - 1), hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, activation) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(
            get_activation(activation),
            nn.Linear(hidden_dim, output_dim),
        )

        if init_mode is not None:
            initialize_weights(self, activation, init_mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        t = inputs[..., :1]
        cond = inputs[..., 1:]
        if self.fourier is not None:
            t = self.fourier(t)
        h = self.input_proj(torch.column_stack([t, cond]))
        h = self.blocks(h)
        return self.output_proj(h)


class SplitResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        activation: Literal['relu', 'tanh', 'silu', 'gelu'] = 'tanh',
        init_mode: Literal['normal', 'uniform'] | None = 'normal',
        fourier_features: int | None = None,
        fourier_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        if fourier_features is not None:
            self.fourier = FourierFeatures(1, fourier_features, fourier_sigma)
            t_dim = self.fourier.out_dim
        else:
            self.fourier = None
            t_dim = 1

        self.t_encoder = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            get_activation(activation),
        )
        self.cond_encoder = nn.Sequential(
            nn.Linear(input_dim - 1, hidden_dim),
            get_activation(activation),
        )
        self.input_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, activation) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(
            get_activation(activation),
            nn.Linear(hidden_dim, output_dim),
        )

        if init_mode is not None:
            initialize_weights(self, activation, init_mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        t = inputs[..., :1]
        cond = inputs[..., 1:]
        if self.fourier is not None:
            t = self.fourier(t)
        h = self.input_proj(torch.column_stack([self.t_encoder(t), self.cond_encoder(cond)]))
        h = self.blocks(h)
        return self.output_proj(h)
