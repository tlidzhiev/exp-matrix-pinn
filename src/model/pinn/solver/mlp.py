from typing import Literal

import torch
import torch.nn as nn

from ...utils import get_activation, initialize_weights


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        activation: Literal['relu', 'tanh', 'silu', 'gelu'] = 'relu',
        init_mode: Literal['normal', 'uniform'] | None = 'normal',
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim,
            ),
            get_activation(activation),
        )

        hiddens = []
        for _ in range(num_blocks):
            hiddens.extend(
                [
                    nn.Linear(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                    ),
                    get_activation(activation),
                ]
            )
        self.hidden_layers = nn.Sequential(*hiddens)
        self.output_proj = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
        )

        if init_mode is not None:
            initialize_weights(self, activation, init_mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(inputs)
        h = self.hidden_layers(h)
        h = self.output_proj(h)
        return h
