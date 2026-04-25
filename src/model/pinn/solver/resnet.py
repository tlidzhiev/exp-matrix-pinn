from typing import Literal

import torch
import torch.nn as nn

from ...utils import get_activation, initialize_weights


class ResBlock(nn.Module):
    def __init__(self, hidden_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
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
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(activation),
        )
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, get_activation(activation)) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        if init_mode is not None:
            initialize_weights(self, activation, init_mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(inputs)
        h = self.blocks(h)
        h = self.output_proj(h)
        return h
