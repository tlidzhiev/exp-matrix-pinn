from typing import Any, Literal

import torch

from ..base import BaseModel
from .solver import get_solver
from .solver.conv1d import Conv1DSolver


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


class Conv1DPINN(BaseModel):
    """
    PINN variant that encodes (X, u0) as a 2-D matrix via Conv1D.

    X is represented as (K, N): each of the k superdiagonals is zero-padded
    to length N and stacked as a channel. u0 is appended as an extra channel,
    yielding a (K+1, N) matrix fed to Conv1DSolver.
    """

    def __init__(
        self,
        n: int,
        k: int,
        t_domain: tuple[float, float],
        solver_params: dict[str, Any],
    ) -> None:
        super().__init__()
        self.n = n
        self.k = k

        rows, cols = torch.triu_indices(n, n, offset=1)
        mask = (cols - rows) <= k
        self.register_buffer('rows', rows[mask])
        self.register_buffer('cols', cols[mask])

        self.solver = Conv1DSolver(
            in_channels=k + 1,
            n=n,
            output_dim=n,
            **solver_params,
        )

        self.register_buffer('t_min', torch.tensor(t_domain[0], dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(t_domain[1], dtype=torch.float32))

    def _vals_to_matrix(self, vals: torch.Tensor) -> torch.Tensor:
        """Convert sparse vals (B, num_vals) → padded diagonal matrix (B, k, n)."""
        B = vals.shape[0]
        x_matrix = torch.zeros(B, self.k, self.n, dtype=vals.dtype, device=vals.device)
        diag_idx = self.cols - self.rows - 1  # ty:ignore[operator]
        pos_idx = self.rows
        x_matrix[:, diag_idx, pos_idx] = vals
        return x_matrix

    def __call__(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u0: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0  # ty:ignore[unsupported-operator]
        x_matrix = self._vals_to_matrix(x)  # (B, k, n)
        x_with_u0 = torch.cat([x_matrix, u0.unsqueeze(1)], dim=1)  # (B, k+1, n)
        ut = self.solver(t_norm, x_with_u0)
        return {'ut': ut}
