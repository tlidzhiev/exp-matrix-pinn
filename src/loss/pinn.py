import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss


class PointBatchPINNLoss(BaseLoss):
    def __init__(
        self,
        n: int,
        k: int,
        lambda_pde: float = 1.0,
        lambda_ic: float = 1.0,
    ) -> None:
        super().__init__()
        self.n, self.k = n, k
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.loss_names: list[str] = ['loss', 'loss_ic', 'loss_pde']
        self.loss_weight_names: list[str] = ['lambda_ic', 'lambda_pde']

        rows, cols = torch.triu_indices(n, n, offset=1)
        mask = (cols - rows) <= k
        self.register_buffer('rows', rows[mask])
        self.register_buffer('cols', cols[mask])

    def forward(
        self,
        model: nn.Module,
        t0: torch.Tensor,
        u0: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor | float]:
        loss_pde = self._loss_pde(t=t, u0=u0, x=x, model=model)
        loss_ic = self._loss_ic(t0=t0, u0=u0, x=x, model=model)

        loss = self.lambda_ic * loss_ic + self.lambda_pde * loss_pde
        output = {
            'loss': loss,
            'loss_pde': loss_pde,
            'loss_ic': loss_ic,
            'lambda_ic': self.lambda_ic,
            'lambda_pde': self.lambda_pde,
        }
        return output

    def _loss_pde(self, t: torch.Tensor, u0: torch.Tensor, x: torch.Tensor, model: nn.Module):
        with fwAD.dual_level():
            dual_t = fwAD.make_dual(t, torch.ones_like(t))
            out_dual = model(t=dual_t, x=x, u0=u0)['ut']
            ut_pred, dudt = fwAD.unpack_dual(out_dual)

        xu = self._apply_matrix(x, ut_pred)
        pde_term = dudt + xu
        loss_pde = (pde_term**2).mean()
        return loss_pde

    def _loss_ic(
        self,
        t0: torch.Tensor,
        u0: torch.Tensor,
        x: torch.Tensor,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        u0_pred = model(t=t0, x=x, u0=u0)['ut']
        loss_ic = F.mse_loss(u0_pred, u0)
        return loss_ic

    def _apply_matrix(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        B = u.shape[0]
        result = torch.zeros_like(u)
        rows = self.rows.expand(B, -1)  # ty:ignore[call-non-callable]
        cols = self.cols.expand(B, -1)  # ty:ignore[call-non-callable]
        result.scatter_add_(1, rows, x * u[:, self.cols])  # ty:ignore[invalid-argument-type]
        result.scatter_add_(1, cols, -x * u[:, self.rows])  # ty:ignore[invalid-argument-type]
        return result

    def extra_repr(self) -> str:
        return f'n={self.n}, k={self.k}, lambda_pde={self.lambda_pde}, lambda_ic={self.lambda_ic})'
