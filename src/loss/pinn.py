import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss


class BasePINNLoss(BaseLoss):
    def __init__(
        self,
        n: int,
        k: int,
        lambda_pde: float = 1.0,
        lambda_ic: float = 1.0,
        hard: bool = False,
    ) -> None:
        super().__init__()
        self.n, self.k = n, k
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.hard = hard
        if hard:
            self.loss_names = ['loss', 'loss_pde']
            self.loss_weight_names = ['lambda_pde']
        else:
            self.loss_names = ['loss', 'loss_ic', 'loss_pde']
            self.loss_weight_names = ['lambda_ic', 'lambda_pde']

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

        if self.hard:
            return {'loss': loss_pde, 'loss_pde': loss_pde, 'lambda_pde': self.lambda_pde}

        loss_ic = self._loss_ic(t0=t0, u0=u0, x=x, model=model)
        loss = self.lambda_ic * loss_ic + self.lambda_pde * loss_pde
        return {
            'loss': loss,
            'loss_pde': loss_pde,
            'loss_ic': loss_ic,
            'lambda_ic': self.lambda_ic,
            'lambda_pde': self.lambda_pde,
        }

    def _loss_pde(
        self, t: torch.Tensor, u0: torch.Tensor, x: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        raise NotImplementedError(f'{type(self).__name__} must implemenent _loss_pde method.')

    def _loss_ic(
        self, t0: torch.Tensor, u0: torch.Tensor, x: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        u0_pred = model(t=t0, x=x, u0=u0)['ut']
        return F.mse_loss(u0_pred, u0)

    def _pde_residual(self, t, u0, x, model):
        eps = 1e-3
        ut_plus = model(t=t + eps, x=x, u0=u0)['ut']
        ut_minus = model(t=t - eps, x=x, u0=u0)['ut']
        dudt = (ut_plus - ut_minus) / (2 * eps)
        ut_pred = model(t=t, x=x, u0=u0)['ut']
        return dudt + self._apply_matrix(x, ut_pred)

    def _apply_matrix(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        B, k, n = x.shape
        u_wins = F.pad(u, (0, k)).unfold(1, k, 1)[:, 1 : n + 1, :]
        result = (x.permute(0, 2, 1) * u_wins).sum(-1)
        xu_padded = F.pad(x * u.unsqueeze(1), (k, 0))
        result -= torch.diagonal(xu_padded.flip(1).unfold(2, n, 1), dim1=1, dim2=2).sum(-1)
        return result

    def extra_repr(self) -> str:
        s = f'n={self.n}, k={self.k}, lambda_pde={self.lambda_pde}'
        if not self.hard:
            s += f', lambda_ic={self.lambda_ic}'
        return s


class PointBatchPINNLoss(BasePINNLoss):
    def _loss_pde(
        self, t: torch.Tensor, u0: torch.Tensor, x: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        return (self._pde_residual(t, u0, x, model) ** 2).mean()
