from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.logger.base import BaseWriter
from src.utils.ema import EMA


@torch.no_grad()
def get_grad_norm(model: nn.Module, norm_type: float | str | None = 2) -> float:
    """
    Calculates the gradient norm for logging.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    norm_type : float or str or None, optional
        The order of the norm, by default 2.

    Returns
    -------
    float
        The calculated norm.
    """
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]

    if len(gradients) == 0:
        return 0.0

    total_norm = torch.norm(
        torch.stack([torch.norm(grad, norm_type) for grad in gradients]),
        norm_type,
    )
    return total_norm.item()


def clip_grad_norm(model: nn.Module, max_grad_norm: float | None = None) -> None:
    """
    Clips the gradient norm by the value.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    max_grad_norm : float or None, optional
        Maximum gradient norm value. If None, no clipping is performed,
        by default None.
    """
    if max_grad_norm is not None:
        clip_grad_norm_(model.parameters(), max_grad_norm)


def save_checkpoint(
    cfg: DictConfig,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    mnt_best: float,
    filename: Path | str,
    epoch: int,
    writer: BaseWriter | None = None,
    ema: EMA | None = None,
) -> None:
    """
    Save checkpoint with model, optimizer, and lr_scheduler states.

    Parameters
    ----------
    cfg : DictConfig
        Experiment config containing training config.
    model : nn.Module
        PyTorch model.
    optimizer : Optimizer
        Optimizer for the model.
    lr_scheduler : LRScheduler
        Learning rate scheduler for the optimizer.
    mnt_best : float
        Best value of the monitored metric achieved so far.
    filename : Path or str
        Path to save the checkpoint file.
    epoch : int
        Current epoch number.
    writer : BaseWriter or None, optional
        Writer instance for logging checkpoint, by default None.
    ema : EMA or None, optional
        EMA instance to save, by default None.
    """
    filename = Path(filename)
    checkpoint_dir = filename.parent
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'monitor_best': mnt_best,
        'cfg': cfg,
    }

    if ema is not None:
        state['ema'] = ema.state_dict()

    torch.save(state, str(filename))
    if writer is not None:
        writer.add_checkpoint(str(filename), str(checkpoint_dir.parent))


def load_checkpoint(path: Path | str, device: str = 'cpu') -> dict[str, Any]:
    """
    Load checkpoint from disk.

    Parameters
    ----------
    path : Path or str
        Path to the checkpoint file.
    device : str, optional
        Device to map the checkpoint tensors to, by default 'cpu'.

    Returns
    -------
    dict[str, Any]
        Dictionary containing checkpoint data including model state,
        optimizer state, epoch, and configuration.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


def resume_checkpoint(
    cfg: DictConfig,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    path: Path | str,
    device: str = 'cpu',
    ema: EMA | None = None,
) -> tuple[nn.Module, Optimizer, LRScheduler, int, float, EMA | None]:
    """
    Resume from a saved checkpoint (in case of server crash, etc.).

    The function loads state dicts for everything, including model,
    optimizers, etc.

    Parameters
    ----------
    cfg : DictConfig
        Experiment config containing training config.
    model : nn.Module
        PyTorch model to load the state into.
    optimizer : Optimizer
        Optimizer to load the state into.
    lr_scheduler : LRScheduler
        Learning rate scheduler to load the state into.
    path : Path or str
        Path to the checkpoint to be resumed.
    device : str, optional
        Device to map the checkpoint tensors to, by default 'cpu'.
    ema : EMA or None, optional
        EMA instance to load the state into, by default None.

    Returns
    -------
    model : nn.Module
        The model with restored state dictionary.
    optimizer : Optimizer
        The optimizer with restored state dictionary.
    lr_scheduler : LRScheduler
        The learning rate scheduler with restored state dictionary.
    start_epoch : int
        The epoch number to start training from.
    mnt_best : float
        The best value of the monitored metric recorded in the checkpoint.
    ema : EMA or None
        The EMA instance with restored state dictionary, if provided.
    """
    path = str(path)
    checkpoint = load_checkpoint(path, device)
    start_epoch = checkpoint['epoch'] + 1
    mnt_best = cast(float, checkpoint.get('monitor_best'))

    # load architecture params from checkpoint.
    if checkpoint['cfg']['model'] != cfg['model']:
        raise RuntimeError(
            'Architecture configuration given in the config file is different from that of the checkpoint'
        )
    else:
        model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if (
        checkpoint['cfg']['optimizer'] != cfg['optimizer']
        or checkpoint['cfg']['lr_scheduler'] != cfg['lr_scheduler']
    ):
        raise RuntimeError(
            'Optimizer or lr_scheduler given in the config file is different '
            'from that of the checkpoint. Optimizer and scheduler parameters '
            'are not resumed.'
        )
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # load EMA state if an EMA object is passed and the state exists in the checkpoint
    if ema is not None and 'ema' in checkpoint:
        ema.load_state_dict(checkpoint['ema'])

    return model, optimizer, lr_scheduler, start_epoch, mnt_best, ema


def from_pretrained(
    model: nn.Module,
    path: Path | str,
    device: str = 'cpu',
) -> nn.Module:
    """
    Init model with weights from pretrained pth file.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    path : Path or str
        Path to the model state dict.
    device : str, optional
        Device to map the checkpoint tensors to, by default 'cpu'.

    Returns
    -------
    nn.Module
        Initialized PyTorch model.
    """
    path = str(path)
    checkpoint = load_checkpoint(path=path, device=device)

    if checkpoint.get('state_dict') is not None:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model
