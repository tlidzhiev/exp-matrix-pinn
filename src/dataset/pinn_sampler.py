from typing import Iterator

import torch


def _sparsity_pattern(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (rows, cols) index arrays for the upper-triangle diagonals 1..k."""
    rows, cols = torch.triu_indices(n, n, offset=1)
    mask = (cols - rows) <= k
    return rows[mask], cols[mask]


def _sample_x_u0(
    batch_size: int,
    n: int,
    num_vals: int,
    trunc_bounds: tuple[float, float],
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    trunc_lower, trunc_upper = trunc_bounds
    vals = torch.nn.init.trunc_normal_(
        torch.empty(batch_size, num_vals), a=trunc_lower, b=trunc_upper, generator=rng
    )
    u0 = torch.nn.init.trunc_normal_(
        torch.empty(batch_size, n), a=trunc_lower, b=trunc_upper, generator=rng
    )
    return vals, u0


class _BasePINNSampler(torch.utils.data.IterableDataset):
    def __init__(
        self,
        n: int,
        k: int,
        batch_size: int,
        t_domain: tuple[float, float] = (0.0, 1.0),
        trunc_bounds: tuple[float, float] = (-2.0, 2.0),
        resample_step: int | None = None,
        seed: int = 42,
    ) -> None:
        self.n = n
        self.k = k
        self.batch_size = batch_size
        self.t_domain = t_domain
        self.trunc_bounds = trunc_bounds
        self.resample_step = resample_step
        self.seed = seed
        self.rows, self.cols = _sparsity_pattern(n, k)
        self.rng = torch.Generator().manual_seed(seed)

    def _sample_batch(self, rng: torch.Generator) -> dict[str, torch.Tensor]:
        raise NotImplementedError(f'{type(self).__name__} must implement _sample_batch method')

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        cached_batch: dict[str, torch.Tensor] | None = None
        step = 0
        while True:
            need_resample = (
                self.resample_step is None or cached_batch is None or step % self.resample_step == 0
            )
            if need_resample:
                cached_batch = self._sample_batch(self.rng)
            yield cached_batch  # ty:ignore[invalid-yield]
            step += 1


class PointBatchSampler(_BasePINNSampler):
    """
    Infinite pointwise sampler for PINN training on du(t)/dt = -Xu(t).

    Each batch contains batch_size independent (t_i, X_i, u0_i) points.
    Use with DataLoader(batch_size=None, num_workers=0).

    Batch tensors
    -------------
    t0  : (batch_size, 1)        — t_min for each point
    t   : (batch_size, 1)        — random collocation time
    x   : (batch_size, num_vals)
    u0  : (batch_size, n)
    """

    def _sample_batch(self, rng: torch.Generator) -> dict[str, torch.Tensor]:
        t_min, t_max = self.t_domain
        num_vals = self.rows.shape[0]

        t0 = torch.full((self.batch_size, 1), t_min)
        t = torch.empty(self.batch_size, 1).uniform_(t_min, t_max, generator=rng)
        vals, u0 = _sample_x_u0(self.batch_size, self.n, num_vals, self.trunc_bounds, rng)
        return {'t0': t0, 'u0': u0, 't': t, 'x': vals}


class TrajectoryBatchSampler(_BasePINNSampler):
    """
    Infinite trajectory-wise sampler for PINN training on du(t)/dt = -Xu(t).

    Each batch contains batch_size trajectories sharing a random time grid.
    Use with DataLoader(batch_size=None, num_workers=0).

    Batch tensors
    -------------
    t0  : (batch_size, 1)          — t_min for each trajectory
    t   : (num_time_points, 1)     — uniform random time points in [t_min, t_max]
    x   : (batch_size, num_vals)
    u0  : (batch_size, n)
    """

    def __init__(self, *args, num_time_points: int = 100, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_time_points = num_time_points

    def _sample_batch(self, rng: torch.Generator) -> dict[str, torch.Tensor]:
        t_min, t_max = self.t_domain
        num_vals = self.rows.shape[0]

        t0 = torch.full((self.batch_size, 1), t_min)
        t = torch.empty(self.num_time_points, 1).uniform_(t_min, t_max, generator=rng)
        vals, u0 = _sample_x_u0(self.batch_size, self.n, num_vals, self.trunc_bounds, rng)
        return {'t0': t0, 'u0': u0, 't': t, 'x': vals}
