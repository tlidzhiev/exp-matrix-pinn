from typing import Literal

import torch
import torch.nn as nn

from ...utils import get_activation
from .fourier import FourierFeatures


class Conv1DSolver(nn.Module):
    """
    Conv1D encoder for (X, u0) matrix input with time conditioning.

    Input layout: x_matrix is (B, K+1, N) where the first K channels are the
    padded superdiagonals of X and the last channel is u0. Time t is encoded
    separately and concatenated with the pooled conv features before the output MLP.
    """

    def __init__(
        self,
        in_channels: int,
        n: int,
        hidden_channels: int,
        num_conv_layers: int,
        hidden_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        activation: Literal['relu', 'tanh', 'silu', 'gelu'] = 'relu',
        fourier_features: int | None = None,
        fourier_sigma: float = 1.0,
        init_mode: Literal['normal', 'uniform'] | None = 'normal',
    ) -> None:
        super().__init__()

        if fourier_features is not None:
            self.fourier = FourierFeatures(1, fourier_features, fourier_sigma)
            t_dim = self.fourier.out_dim
        else:
            self.fourier = None
            t_dim = 1

        conv_layers: list[nn.Module] = []
        in_ch = in_channels
        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv1d(in_ch, hidden_channels, kernel_size, padding='same'))
            conv_layers.append(get_activation(activation))
            in_ch = hidden_channels
        self.conv_encoder = nn.Sequential(*conv_layers)

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels + t_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, output_dim),
        )

        if init_mode is not None:
            self._initialize(activation, init_mode)

    def _initialize(self, activation: str, mode: str) -> None:
        nonlin = 'leaky_relu' if 'leaky' in activation else 'relu'
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                if activation == 'tanh':
                    gain = nn.init.calculate_gain('tanh')
                    init_fn = (
                        nn.init.xavier_normal_ if mode == 'normal' else nn.init.xavier_uniform_
                    )
                    init_fn(m.weight, gain=gain)
                else:
                    init_fn = (
                        nn.init.kaiming_normal_ if mode == 'normal' else nn.init.kaiming_uniform_
                    )
                    init_fn(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, x_matrix: torch.Tensor) -> torch.Tensor:
        # t: (B, 1), x_matrix: (B, K+1, N)
        t_enc = self.fourier(t) if self.fourier is not None else t  # (B, t_dim)
        h = self.conv_encoder(x_matrix)  # (B, hidden_channels, N)
        h = h.mean(dim=-1)  # global avg pool → (B, hidden_channels)
        h = torch.cat([h, t_enc], dim=-1)  # (B, hidden_channels + t_dim)
        return self.output_mlp(h)  # (B, output_dim)
