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
            effective_input = self.fourier.out_dim + (input_dim - 1)
        else:
            self.fourier = None
            effective_input = input_dim

        self.input_proj = nn.Linear(effective_input, hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, activation) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(
            get_activation(activation),
            nn.Linear(hidden_dim, output_dim),
        )

        if init_mode is not None:
            initialize_weights(self, activation, init_mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            h = torch.column_stack([self.fourier(inputs[..., :1]), inputs[..., 1:]])
        else:
            h = inputs
        h = self.input_proj(h)
        h = self.blocks(h)
        h = self.output_proj(h)
        return h
