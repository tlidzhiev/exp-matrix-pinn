import torch
import torch.nn as nn

from ...utils import get_activation


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            get_activation(activation),
            nn.Conv1d(channels, channels, kernel_size, padding='same'),
            get_activation(activation),
            nn.Conv1d(channels, channels, kernel_size, padding='same'),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.block(h)


class TimeConditionedResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.act = get_activation(activation)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.time_proj = nn.Sequential(get_activation(activation), nn.Linear(channels, channels))

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h_out = self.conv1(self.act(h))
        h_out = h_out + self.time_proj(t_emb).unsqueeze(-1)
        h_out = self.conv2(self.act(h_out))
        return h + h_out


class FiLMConditionedResBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, activation: str) -> None:
        super().__init__()
        self.act = get_activation(activation)

        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same')

        self.film_proj = nn.Sequential(
            get_activation(activation), nn.Linear(hidden_dim, hidden_dim * 2)
        )

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h_out = self.conv1(self.act(h))
        film_params = self.film_proj(t_emb)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        h_out = h_out * (1.0 + gamma) + beta
        h_out = self.conv2(self.act(h_out))
        return h + h_out
