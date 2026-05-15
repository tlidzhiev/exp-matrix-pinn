from typing import Iterator

import torch


def _sparsity_pattern(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (rows, cols) index arrays for the upper-triangle diagonals 1..k."""
    rows, cols = torch.triu_indices(n, n, offset=1)
    mask = (cols - rows) <= k
    return rows[mask], cols[mask]


def _vals_to_matrix(
    vals: torch.Tensor,
    n: int,
    k: int,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    """
    Convert sparse vals (B, num_vals) to padded diagonal matrix (B, k, n).

    Each of the k superdiagonals occupies one channel, zero-padded to length n.
    """
    B = vals.shape[0]
    x_matrix = torch.zeros(B, k, n, dtype=vals.dtype, device=vals.device)
    diag_idx = cols - rows - 1
    pos_idx = rows
    x_matrix[:, diag_idx, pos_idx] = vals
    return x_matrix


def _sample_x_u0(
    batch_size: int,
    n: int,
    num_vals: int,
    trunc_bounds: tuple[float, float],
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    trunc_lower, trunc_upper = trunc_bounds
    vals = torch.nn.init.trunc_normal_(
        torch.empty(batch_size, num_vals, dtype=torch.float32),
        a=trunc_lower,
        b=trunc_upper,
        generator=rng,
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
    def _sample_batch(self, rng: torch.Generator) -> dict[str, torch.Tensor]:
        t_min, t_max = self.t_domain
        num_vals = self.rows.shape[0]

        t = torch.empty(self.batch_size, 1, dtype=torch.float32).uniform_(
            t_min, t_max, generator=rng
        )
        vals, u0 = _sample_x_u0(self.batch_size, self.n, num_vals, self.trunc_bounds, rng)
        x = _vals_to_matrix(vals, self.n, self.k, self.rows, self.cols)
        return {'u0': u0, 't': t, 'x': x}
