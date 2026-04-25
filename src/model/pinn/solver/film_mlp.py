from typing import Literal

import torch
import torch.nn as nn

from ...utils import get_activation, initialize_weights


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
        main_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        activation: Literal['relu', 'tanh', 'silu', 'gelu'] = 'tanh',
        init_mode: Literal['normal', 'uniform'] | None = 'normal',
    ) -> None:
        super().__init__()
        self.main_dim = main_dim
        cond_dim = input_dim - main_dim

        self.act = get_activation(activation)
        self.input_proj = nn.Sequential(nn.Linear(main_dim, hidden_dim), self.act)

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
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
        t = inputs[..., : self.main_dim]
        raw_cond = inputs[..., self.main_dim :]
        cond_embedding = self.cond_mlp(raw_cond)

        h = self.input_proj(t)
        for block, film in zip(self.blocks, self.film_layers):
            h = self.act(film(block(h), cond_embedding))
        return self.output_proj(h)
