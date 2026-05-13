import io

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch


def plot_trajectories(
    ut: torch.Tensor,
    ut_pred: torch.Tensor,
    n_dims: int = 3,
) -> np.ndarray:
    """
    Plot target vs predicted trajectories for fixed samples and first n_dims coordinates.

    Parameters
    ----------
    ut : torch.Tensor
        Ground truth trajectories, shape (B, T, n).
    ut_pred : torch.Tensor
        Predicted trajectories, shape (B, T, n).
    n_dims : int
        Number of dimensions to plot.

    Returns
    -------
    np.ndarray
        HWC uint8 RGB image of the figure.
    """
    B, T, n = ut.shape
    n_dims = min(n_dims, n)
    K = min(8, B)
    ut_np = ut[:K, :].cpu().float().numpy()
    ut_pred_np = ut_pred[:K, :].cpu().float().numpy()

    fig, axes = plt.subplots(K, n_dims, figsize=(4 * n_dims, 3 * K), squeeze=False)

    for i in range(K):
        for j in range(n_dims):
            ax = axes[i, j]
            ax.plot(ut_np[i, :, j], label='target', linewidth=1.5)
            ax.plot(ut_pred_np[i, :, j], label='pred', linewidth=1.5, linestyle='--')
            ax.set_title(f'sample {i}, coord {j}')
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
            ax.tick_params(labelsize=7)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)

    image = np.array(PIL.Image.open(buf).convert('RGB'))
    return image
