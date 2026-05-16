from typing import Literal

import torch
import torch.nn as nn

from ...utils import get_activation, initialize_weights
from .fourier import FourierFeatures


class Baseline(nn.Module):
    def __init__(
        self,
        n: int,
        k: int,
        hidden_dim: int,
        num_xu_blocks: int,
        num_fusion_blocks: int,
        activation: Literal['relu', 'tanh', 'silu', 'gelu'] = 'silu',
        fourier_features: int | None = None,
        fourier_sigma: float = 1.0,
        kernel_size: int = 3,
        init_mode: Literal['normal', 'uniform'] | None = 'normal',
    ) -> None:
        super().__init__()
        self.n = n

        if fourier_features is not None:
            self.fourier = FourierFeatures(1, fourier_features, fourier_sigma)
            t_input_dim = self.fourier.out_dim
        else:
            self.fourier = None
            t_input_dim = 1

        self.t_encoder = nn.Sequential(
            nn.Linear(t_input_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        )

        xu_layers = []
        in_ch = k + 1
        for _ in range(num_xu_blocks):
            xu_layers.extend(
                [
                    nn.Conv1d(in_ch, hidden_dim, kernel_size, padding='same'),
                    get_activation(activation),
                ]
            )
            in_ch = hidden_dim
        self.xu_encoder = nn.Sequential(*xu_layers)

        fusion_layers = []
        in_ch = hidden_dim + hidden_dim
        for _ in range(num_fusion_blocks):
            fusion_layers += [
                nn.Conv1d(in_ch, hidden_dim, kernel_size, padding='same'),
                get_activation(activation),
            ]
            in_ch = hidden_dim
        self.fusion = nn.Sequential(*fusion_layers)
        self.output_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        if init_mode is not None:
            initialize_weights(self, activation, init_mode)

    def forward(self, t: torch.Tensor, x_matrix: torch.Tensor) -> torch.Tensor:
        t_enc = self.fourier(t) if self.fourier is not None else t
        t_h = self.t_encoder(t_enc)
        t_h_exp = t_h.unsqueeze(-1).expand(-1, -1, self.n)
        xu_h = self.xu_encoder(x_matrix)
        h = torch.cat([xu_h, t_h_exp], dim=1)
        h = self.fusion(h)
        return self.output_proj(h).squeeze(1)
