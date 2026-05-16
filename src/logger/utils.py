import io

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch


def plot_trajectories(
    ut: torch.Tensor,
    ut_pred: torch.Tensor,
    t: torch.Tensor,
    n_dims: int = 3,
) -> np.ndarray:
    """
    Plot target vs predicted trajectories for fixed samples and first/last n_dims coordinates.

    Parameters
    ----------
    ut : torch.Tensor
        Ground truth trajectories, shape (B, T, n).
    ut_pred : torch.Tensor
        Predicted trajectories, shape (B, T, n).
    t : torch.Tensor
        Time grid, shape (B, T).
    n_dims : int
        Number of first and last dimensions to plot.

    Returns
    -------
    np.ndarray
        HWC uint8 RGB image of the figure.
    """
    B, T, n = ut.shape
    n_dims = min(n_dims, n)
    first = list(range(n_dims))
    last = list(range(n - n_dims, n))
    seen: set[int] = set()
    dims = []
    for d in first + last:
        if d not in seen:
            seen.add(d)
            dims.append(d)
    K = min(8, B)
    ut_np = ut[:K, :].cpu().float().numpy()
    ut_pred_np = ut_pred[:K, :].cpu().float().numpy()
    t_np = t[:K].cpu().float().numpy()

    fig, axes = plt.subplots(K, len(dims), figsize=(4 * len(dims), 3 * K), squeeze=False)

    for i in range(K):
        for col, j in enumerate(dims):
            ax = axes[i, col]
            ax.plot(t_np[i], ut_np[i, :, j], label='target', linewidth=1.5)
            ax.plot(t_np[i], ut_pred_np[i, :, j], label='pred', linewidth=1.5, linestyle='--')
            ax.set_title(f'sample {i}, coord {j}')
            if i == 0 and col == 0:
                ax.legend(fontsize=8)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    plt.show()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)

    image = np.array(PIL.Image.open(buf).convert('RGB'))
    return image
