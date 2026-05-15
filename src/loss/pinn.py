import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss


class BasePINNLoss(BaseLoss):
    def __init__(
        self,
        n: int,
        k: int,
        lambda_ode: float = 1.0,
        num_chunks: int | None = None,
        eps_causal: float = 1.0,
    ) -> None:
        super().__init__()
        self.n, self.k = n, k
        self.lambda_ode = lambda_ode
        self.loss_names = ['loss', 'loss_ode']
        self.loss_weight_names = ['lambda_ode']

        self.num_chunks = num_chunks
        self.eps_causal = eps_causal

    def forward(
        self,
        model: nn.Module,
        t: torch.Tensor,
        x: torch.Tensor,
        u0: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor | float]:
        loss_pde = self.lambda_ode * self._loss_ode(t=t, u0=u0, x=x, model=model)
        return {'loss': loss_pde, 'loss_ode': loss_pde, 'lambda_ode': self.lambda_ode}

    def _loss_ode(
        self, t: torch.Tensor, u0: torch.Tensor, x: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        raise NotImplementedError(f'{type(self).__name__} must implemenent _loss_ode method.')

    def _ode_residual(
        self, t: torch.Tensor, u0: torch.Tensor, x: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
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
        s = f'n={self.n}, k={self.k}, lambda_ode={self.lambda_ode}'
        if self.num_chunks:
            s += f', num_chunks={self.num_chunks}, eps_causal={self.eps_causal}'
        return s


class PointBatchPINNLoss(BasePINNLoss):
    def _loss_ode(
        self, t: torch.Tensor, u0: torch.Tensor, x: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        B = x.shape[0]
        pde_term = self._ode_residual(t, u0, x, model)

        if self.num_chunks is None:
            return (pde_term**2).mean()

        C = self.num_chunks
        sort_idx = t.view(B).argsort()
        pde_term = pde_term[sort_idx].reshape(C, B // C, -1)
        L_bc = (pde_term**2).mean(dim=[1, 2])
        cumsum_prev = torch.cat([t.new_zeros(1), L_bc[:-1].cumsum(dim=0)], dim=0).detach()
        weights = torch.exp(-self.eps_causal * cumsum_prev)
        return (weights * L_bc).mean()


class TrajectoryBatchPINNLoss(BasePINNLoss):
    def _loss_ode(
        self, t: torch.Tensor, u0: torch.Tensor, x: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        B, T = x.shape[0], t.shape[0]

        t_sorted = t.squeeze(1).sort().values.unsqueeze(1)
        t_exp = t_sorted.unsqueeze(0).expand(B, T, 1).reshape(B * T, 1)
        x_exp = x.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, *x.shape[1:])
        u0_exp = u0.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)

        pde_term = self._ode_residual(t_exp, u0_exp, x_exp, model).reshape(B, T, -1)

        if self.num_chunks is None:
            return (pde_term**2).mean()

        C = self.num_chunks
        pde_term = pde_term.reshape(B, C, T // C, -1)
        L_bc = (pde_term**2).mean(dim=[2, 3])
        cumsum_prev = torch.cat([t.new_zeros(B, 1), L_bc[:, :-1].cumsum(dim=1)], dim=1).detach()
        weights = torch.exp(-self.eps_causal * cumsum_prev)
        return (weights * L_bc).mean()
