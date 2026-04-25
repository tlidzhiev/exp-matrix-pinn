from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def get_lr_scheduler(
    cfg: DictConfig,
    optimizer: Optimizer,
    epoch_len: int,
) -> LRScheduler:
    """
    Get learning rate scheduler from config.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    optimizer : Optimizer
        PyTorch optimizer.
    epoch_len : int
        Number of steps in each epoch.

    Returns
    -------
    LRScheduler
        Learning rate scheduler.
    """
    if cfg.lr_scheduler.get('scheduler') is not None:
        if cfg.lr_scheduler.scheduler.name == 'constant':
            num_training_steps, num_warmup_steps = None, None
        else:
            num_training_steps = cfg.trainer.num_epochs * epoch_len
            num_warmup_steps = int(
                round(num_training_steps * cfg.lr_scheduler.get('warmup_ratio', 0.03))
            )
        return instantiate(
            cfg.lr_scheduler.scheduler,
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )
    else:
        return instantiate(cfg.lr_scheduler, optimizer=optimizer)
