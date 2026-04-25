from typing import Any, Literal

import torch

from ..base import BaseModel
from .solver import get_solver


class PINN(BaseModel):
    def __init__(
        self,
        n: int,
        k: int,
        t_domain: tuple[float, float],
        solver_name: Literal['mlp', 'resnet', 'film_mlp'],
        solver_params: dict[str, Any],
    ) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self.t_domain = t_domain

        n_nnz = sum(n - d for d in range(1, k + 1))
        solver_params['input_dim'] = 1 + n_nnz + n  # t, x (vals), u0
        solver_params['output_dim'] = n
        self.solver = get_solver(solver_name, solver_params)

        self.register_buffer(
            't_min',
            torch.tensor(t_domain[0], requires_grad=False, dtype=torch.float32),
        )
        self.register_buffer(
            't_max',
            torch.tensor(t_domain[1], requires_grad=False, dtype=torch.float32),
        )

    def __call__(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u0: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0  # ty:ignore[unsupported-operator]
        z = torch.column_stack([t_norm, x, u0])
        ut = self.solver(z)
        return {'ut': ut}
