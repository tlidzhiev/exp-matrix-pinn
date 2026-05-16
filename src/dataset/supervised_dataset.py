from pathlib import Path
from typing import Any, Callable, Literal

import safetensors.torch
import torch
from tqdm.auto import tqdm

from src.dataset.base import BaseDataset
from src.dataset.pinn_sampler import _sparsity_pattern, _vals_to_matrix
from src.utils.io import get_root, read_json, write_json


class SparseExpMatrixDataset(BaseDataset):
    """
    Supervised dataset for du(t)/dt = -Xu(t) with k-diagonal skew-symmetric X.
    """

    def __init__(
        self,
        n: int,
        k: int,
        root: Path | str | None = None,
        split: Literal['train', 'test'] = 'train',
        t_domain: tuple[float, float] = (0.0, 1.0),
        trunc_bounds: tuple[float, float] = (-2.0, 2.0),
        num_samples: int = 10_000,
        num_time_points: int = 100,
        seed: int = 42,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: dict[str, Callable] | None = None,
        force_reindex: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        n : int
            Dimension of the ODE system (matrix X is n×n).
        k : int
            Number of super-diagonals with non-zero entries (bandwidth).
            X has non-zero entries on diagonals ±1, ±2, ..., ±k.
        root : Path or str or None, optional
            Root directory for the dataset, by default None.
            If None, uses 'data/ode/{split}' from project root.
        split : {'train', 'test'}, optional
            Dataset split name, by default 'train'.
        t_domain : tuple[float, float], optional
            Time bounds (t_min, t_max), by default (0.0, 1.0).
        num_samples : int, optional
            Number of (X, u0) pairs to generate, by default 10_000.
        num_time_points : int, optional
            Number of time points in the uniform grid, by default 100.
        trunc_bounds : tuple[float, float], optional
            Truncation bounds (lower, upper) in standard deviations, by default (-2.0, 2.0).
        seed : int, optional
            RNG seed for reproducibility, by default 42.
        limit : int or None, optional
            Limit number of samples, by default None.
        shuffle_index : bool, optional
            Whether to shuffle index, by default False.
        instance_transforms : dict[str, Callable] or None, optional
            Instance transforms to apply, by default None.
        force_reindex : bool, optional
            Force recreation of index file, by default False.
        """
        self.n = n
        self.k = k
        self.t_domain = t_domain
        self.trunc_bounds = trunc_bounds
        self.num_samples = num_samples
        self.num_time_points = num_time_points
        self.seed = seed

        if root is None:
            root = get_root() / 'data' / f'ode-n={n}-k={k}-t={t_domain[0]},{t_domain[1]}' / split
        else:
            root = Path(root)

        index_path = root / 'index.json'
        if index_path.exists() and not force_reindex:
            index: list[dict[str, Any]] = read_json(str(index_path))  # ty:ignore[invalid-assignment]
        else:
            index = self._create_index(root)

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _create_index(self, data_path: Path) -> list[dict[str, Any]]:
        data_path.mkdir(parents=True, exist_ok=True)

        rng = torch.Generator().manual_seed(self.seed)
        t_min, t_max = self.t_domain
        trunc_lower, trunc_upper = self.trunc_bounds
        t = torch.linspace(t_min, t_max, self.num_time_points)
        rows, cols = _sparsity_pattern(self.n, self.k)

        index = []
        print(
            f'Generating {self.num_samples} ODE samples (n={self.n}, k={self.k}, t_domain={self.t_domain}, num_time_points={self.num_time_points})...'
        )
        for i in tqdm(range(self.num_samples)):
            vals = torch.nn.init.trunc_normal_(
                torch.empty(rows.shape[0]), a=trunc_lower, b=trunc_upper, generator=rng
            )
            u0 = torch.nn.init.trunc_normal_(
                torch.empty(self.n), a=trunc_lower, b=trunc_upper, generator=rng
            )

            X_dense = self._build_dense(vals, rows, cols, self.n)
            ut = torch.linalg.matrix_exp(-t.reshape(-1, 1, 1) * X_dense) @ u0

            x = _vals_to_matrix(vals.unsqueeze(0), self.n, self.k, rows, cols).squeeze(0)  # (k, n)

            save_path = data_path / f'{i:06d}.safetensors'
            safetensors.torch.save_file({'u0': u0, 'ut': ut, 't': t, 'x': x}, save_path)
            index.append({'path': str(save_path)})

        write_json(index, str(data_path / 'index.json'))
        print(f'Successfully generated {len(index)} samples.')
        return index

    @staticmethod
    def _build_dense(
        vals: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Reconstruct a dense skew-symmetric (n, n) matrix from upper-triangle values."""
        X = torch.zeros(n, n)
        X[rows, cols] = vals
        X[cols, rows] = -vals
        return X
