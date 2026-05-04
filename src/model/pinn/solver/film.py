from typing import Literal

import torch
import torch.nn as nn

from ...utils import get_activation
from .fourier import FourierFeatures
from .residual import FiLMConditionedResBlock, ResBlock


class FiLM(nn.Module):
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
            get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.xu_encoder = nn.Sequential(
            nn.Conv1d(k + 1, hidden_dim, kernel_size=1),
            *[ResBlock(hidden_dim, kernel_size, activation) for _ in range(num_xu_blocks)],
        )
        self.fusion = nn.ModuleList(
            [
                FiLMConditionedResBlock(hidden_dim, kernel_size, activation)
                for _ in range(num_fusion_blocks)
            ]
        )
        self.output_proj = nn.Sequential(
            get_activation(activation),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, t: torch.Tensor, x_matrix: torch.Tensor) -> torch.Tensor:
        t_enc = self.fourier(t) if self.fourier is not None else t
        t_h = self.t_encoder(t_enc)
        h = self.xu_encoder(x_matrix)
        for block in self.fusion:
            h = block(h, t_h)
        return self.output_proj(h).squeeze(1)
