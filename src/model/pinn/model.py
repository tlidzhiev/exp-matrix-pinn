from typing import Any, Literal

import torch

from ..base import BaseModel
from .solver import get_solver

_SolverName = Literal['baseline', 'resnet', 'film']


class PINN(BaseModel):
    def __init__(
        self,
        n: int,
        k: int,
        t_domain: tuple[float, float],
        solver_params: dict[str, Any],
        solver_name: _SolverName = 'baseline',
        ansatz_type: Literal['linear', 'exp'] = 'linear',
    ) -> None:
        super().__init__()
        self.ansatz_type = ansatz_type
        if self.ansatz_type == 'exp':
            self.tau = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
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
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        x_matrix = torch.cat([x, u0.unsqueeze(1)], dim=1)
        t_offset = t - self.t_min
        delta = self.solver(t_norm, x_matrix)
        if self.ansatz_type == 'linear':
            ut_raw = u0 + t_offset * delta
        else:
            ut_raw = u0 + (1 - torch.exp(-t_offset / (self.tau.abs() + 1e-6))) * delta
        u0_norm = u0.norm(dim=-1, keepdim=True)
        ut_norm = ut_raw.norm(dim=-1, keepdim=True) + 1e-8
        ut = (ut_raw / ut_norm) * u0_norm
        return {'ut': ut}
