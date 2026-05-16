from typing import Any, Literal

import torch

from src.logger.utils import plot_trajectories
from src.metrics.tracker import MetricTracker

from .base import BaseTrainer


class PINNTrainer(BaseTrainer):
    """
    PINN Trainer class.

    Defines the logic of batch logging and processing for PINN models.
    """

    def process_batch(
        self,
        batch_idx: int,
        batch: dict[str, Any],
        metric_tracker: MetricTracker,
        part: Literal['train', 'val', 'test'] | str = 'train',
    ) -> dict[str, Any]:
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Parameters
        ----------
        batch_idx : int
            Batch index.
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader.
        metric_tracker : MetricTracker
            MetricTracker object that computes and aggregates the metrics.
            The metrics depend on the type of the partition (train or inference).
        part : {'train', 'val', 'test'} or str, optional
            Partition type, by default 'train'.

        Returns
        -------
        dict[str, Any]
            Dict-based batch containing the data from the dataloader
            (possibly transformed via batch transform), model outputs, and losses.
        """
        batch = self._to_device(batch)
        batch = self._transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics['train' if part == 'train' else 'inference']
        if part == 'train':
            self.optimizer.zero_grad()

        if part == 'train':
            all_losses = self.criterion(model=self.model, **batch)
            batch.update(all_losses)
            batch['loss'].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()

            # update metrics for each loss (in case of multiple losses)
            for loss_name in self.criterion.loss_names:
                metric_tracker.update(loss_name, batch[loss_name].item())
            for loss_weight_name in self.criterion.loss_weight_names:
                metric_tracker.update(loss_weight_name, batch[loss_weight_name])
        else:
            B, num_t, n = batch['ut'].shape
            t_flat = batch['t'].reshape(B * num_t, 1)
            x_flat = batch['x'].repeat_interleave(num_t, dim=0)
            u0_flat = batch['u0'].repeat_interleave(num_t, dim=0)
            ut_pred = self.model(t=t_flat, x=x_flat, u0=u0_flat)['ut'].reshape(B, num_t, n)
            batch['ut_pred'] = ut_pred

        for met in metric_funcs:
            metric_tracker.update(met.name, met(**batch))

        return batch

    @torch.no_grad()
    def _log_batch(
        self,
        batch_idx: int,
        batch: dict[str, Any],
        epoch: int,
    ) -> None:
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch.
        batch : dict[str, Any]
            Dict-based batch after going through the 'process_batch' function.
        epoch : int
            Current epoch number.
        """
        if not self.is_train:
            image = plot_trajectories(batch['ut'], batch['ut_pred'], batch['t'])
            self.writer.add_image('trajectories', image)
