from typing import Iterator

import torch


def _sparsity_pattern(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (rows, cols) index arrays for the upper-triangle diagonals 1..k."""
    rows, cols = torch.triu_indices(n, n, offset=1)
    mask = (cols - rows) <= k
    return rows[mask], cols[mask]


class PINNBatchSampler(torch.utils.data.IterableDataset):
    """
    Infinite batched sampler for PINN training on du(t)/dt = -Xu(t).

    Yields pre-formed batches, optionally reusing (resampling) each batch for
    resample_step consecutive steps before drawing a new one.
    Use with DataLoader(batch_size=None, num_workers=0).
    """

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
        """
        Parameters
        ----------
        n : int
            Dimension of the ODE system (matrix X is n×n).
        k : int
            Number of super-diagonals with non-zero entries (bandwidth).
            X has non-zero entries on diagonals ±1, ±2, ..., ±k.
        batch_size : int
            Number of collocation points per batch.
        t_domain : tuple[float, float], optional
            Time bounds (t_min, t_max), by default (0.0, 1.0).
        trunc_bounds : tuple[float, float], optional
            Truncation bounds (lower, upper) in standard deviations, by default (-2.0, 2.0).
        resample_step : int or None, optional
            Resample the batch every this many steps. None means resample every step.
        seed : int, optional
            RNG seed for reproducibility, by default 42.
        """
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
        t_min, t_max = self.t_domain
        trunc_lower, trunc_upper = self.trunc_bounds
        num_vals = self.rows.shape[0]

        t0 = torch.full((self.batch_size, 1), t_min)
        t = torch.empty(self.batch_size, 1).uniform_(t_min, t_max, generator=rng)
        vals = torch.nn.init.trunc_normal_(
            torch.empty(self.batch_size, num_vals), a=trunc_lower, b=trunc_upper, generator=rng
        )
        u0 = torch.nn.init.trunc_normal_(
            torch.empty(self.batch_size, self.n),
            a=trunc_lower,
            b=trunc_upper,
            generator=rng,
        )

        return {'t0': t0, 'u0': u0, 't': t, 'x': vals}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Yield an endless stream of random training batches.

        Yields
        ------
        dict[str, torch.Tensor]
            Batch containing t0, u0, t, and x tensors.
        """
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
