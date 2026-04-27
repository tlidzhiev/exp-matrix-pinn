from typing import Literal

import torch
import torch.nn as nn

from ...utils import get_activation, initialize_weights
from .fourier import FourierFeatures


class FiLMLayer(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, hidden_dim)
        self.beta_proj = nn.Linear(cond_dim, hidden_dim)

        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.gamma_proj(cond) * h + self.beta_proj(cond)


class FiLMMLP(nn.Module):
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
            effective_input = self.fourier.out_dim
        else:
            self.fourier = None
            effective_input = 1

        self.act = get_activation(activation)
        self.input_proj = nn.Sequential(
            nn.Linear(effective_input, hidden_dim), get_activation(activation)
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(input_dim - 1, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
        )
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.film_layers = nn.ModuleList(
            [FiLMLayer(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        if init_mode is not None:
            initialize_weights(self.input_proj, activation, init_mode)
            initialize_weights(self.cond_mlp, activation, init_mode)
            initialize_weights(self.blocks, activation, init_mode)
            initialize_weights(self.output_proj, activation, init_mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        t = inputs[..., :1]
        raw_cond = inputs[..., 1:]
        cond_embedding = self.cond_mlp(raw_cond)

        h = self.fourier(t) if self.fourier is not None else t
        h = self.input_proj(h)
        for block, film in zip(self.blocks, self.film_layers):
            h = self.act(film(block(h), cond_embedding))
        return self.output_proj(h)
