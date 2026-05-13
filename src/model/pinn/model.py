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
        hard_constrained: bool = False,
    ) -> None:
        super().__init__()
        self.hard_constrained = hard_constrained
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
        x_matrix = torch.cat([x, u0.unsqueeze(1)], dim=1)
        if self.hard_constrained:
            t_offset = t - self.t_min  # ty:ignore[unsupported-operator]
            delta = self.solver(t_norm, x_matrix)
            ut = u0 + t_offset * delta
            u0_norm = u0.norm(dim=-1, keepdim=True)
            ut = ut / ut.norm(dim=-1, keepdim=True) * u0_norm
            return {'ut': ut}

        ut = self.solver(t_norm, x_matrix)
        return {'ut': ut}
