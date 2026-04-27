from typing import Literal

import torch
import torch.nn as nn

from ...utils import get_activation, initialize_weights
from .fourier import FourierFeatures


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        activation: Literal['relu', 'tanh', 'silu', 'gelu'] = 'relu',
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

        self.input_proj = nn.Sequential(
            nn.Linear(t_dim + (input_dim - 1), hidden_dim),
            get_activation(activation),
        )
        hiddens = []
        for _ in range(num_blocks):
            hiddens.extend([nn.Linear(hidden_dim, hidden_dim), get_activation(activation)])
        self.hidden_layers = nn.Sequential(*hiddens)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        if init_mode is not None:
            initialize_weights(self, activation, init_mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        t = inputs[..., :1]
        cond = inputs[..., 1:]
        if self.fourier is not None:
            t = self.fourier(t)
        h = self.input_proj(torch.column_stack([t, cond]))
        h = self.hidden_layers(h)
        return self.output_proj(h)


class SplitMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        activation: Literal['relu', 'tanh', 'silu', 'gelu'] = 'relu',
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
        self.input_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            get_activation(activation),
        )
        hiddens = []
        for _ in range(num_blocks):
            hiddens.extend([nn.Linear(hidden_dim, hidden_dim), get_activation(activation)])
        self.hidden_layers = nn.Sequential(*hiddens)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        if init_mode is not None:
            initialize_weights(self, activation, init_mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        t = inputs[..., :1]
        cond = inputs[..., 1:]
        if self.fourier is not None:
            t = self.fourier(t)
        h = self.input_proj(torch.column_stack([self.t_encoder(t), self.cond_encoder(cond)]))
        h = self.hidden_layers(h)
        return self.output_proj(h)
