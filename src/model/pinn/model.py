from typing import Any, Literal

import torch

from ..base import BaseModel
from .solver import get_solver

_SolverName = Literal['baseline', 'resnet', 'resnet_tc', 'film']


class PINN(BaseModel):
    def __init__(
        self,
        n: int,
        k: int,
        t_domain: tuple[float, float],
        solver_params: dict[str, Any],
        solver_name: _SolverName = 'baseline',
    ) -> None:
        super().__init__()
        self.solver = get_solver(solver_name, {'k': k, 'n': n, **solver_params})
        self.register_buffer('t_min', torch.tensor(t_domain[0], dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(t_domain[1], dtype=torch.float32))

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u0: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0  # ty:ignore[unsupported-operator]
        x_matrix = torch.cat([x, u0.unsqueeze(1)], dim=1)  # (B, k+1, n)
        ut = self.solver(t_norm, x_matrix)
        return {'ut': ut}


class HardPINN(BaseModel):
    def __init__(
        self,
        n: int,
        k: int,
        t_domain: tuple[float, float],
        solver_params: dict[str, Any],
        solver_name: _SolverName = 'baseline',
        norm_preserving: bool = False,
    ) -> None:
        super().__init__()
        self.norm_preserving = norm_preserving
        self.solver = get_solver(solver_name, {'k': k, 'n': n, **solver_params})
        self.register_buffer('t_min', torch.tensor(t_domain[0], dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(t_domain[1], dtype=torch.float32))

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u0: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0  # ty:ignore[unsupported-operator]
        t_offset = t - self.t_min  # ty:ignore[unsupported-operator]

        x_matrix = torch.cat([x, u0.unsqueeze(1)], dim=1)
        delta = self.solver(t_norm, x_matrix)
        ut = u0 + t_offset * delta

        if self.norm_preserving:
            u0_norm = u0.norm(dim=-1, keepdim=True)
            ut = ut / ut.norm(dim=-1, keepdim=True) * u0_norm

        return {'ut': ut}
