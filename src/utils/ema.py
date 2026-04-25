from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters that are updated
    with exponential moving average during training. The EMA weights
    typically provide better generalization than the final trained weights.

    Parameters
    ----------
    model : nn.Module
        The model to track.
    decay : float, optional
        The decay rate for EMA. Higher values mean slower updates.
        Common values: 0.999, 0.9999. By default 0.999.
    warmup_steps : int, optional
        Number of steps before EMA starts. During warmup, decay is
        ramped up from 0 to the target decay. By default 0.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 0,
    ) -> None:
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0

        # Create shadow model (deep copy of parameters)
        self.shadow = deepcopy(model)
        self.shadow.eval()
        self.shadow.requires_grad_(False)

        # Store original model reference for context manager
        self._model: nn.Module | None = None

    def get_decay(self) -> float:
        """
        Get the current decay value, accounting for warmup.

        Returns
        -------
        float
            Current decay value.
        """
        if self.warmup_steps > 0 and self.step < self.warmup_steps:
            # Linear warmup of decay
            return self.decay * (self.step / self.warmup_steps)
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update EMA weights with current model weights.

        Should be called after each optimizer step.

        Parameters
        ----------
        model : nn.Module
            The model with updated weights.
        """
        decay = self.get_decay()
        self.step += 1

        for ema_param, model_param in zip(
            self.shadow.parameters(),
            model.parameters(),
            strict=True,
        ):
            # EMA update: shadow = decay * shadow + (1 - decay) * model
            ema_param.lerp_(model_param, 1 - decay)

        # Also update buffers (e.g., BatchNorm running stats)
        for ema_buf, model_buf in zip(
            self.shadow.buffers(),
            model.buffers(),
            strict=True,
        ):
            ema_buf.copy_(model_buf)

    def apply_shadow(self, model: nn.Module) -> None:
        """
        Copy EMA weights to the model.

        Parameters
        ----------
        model : nn.Module
            The model to copy EMA weights to.
        """
        model.load_state_dict(self.shadow.state_dict())

    def state_dict(self) -> dict:
        """
        Return EMA state for checkpointing.

        Returns
        -------
        dict
            Dictionary containing EMA state.
        """
        return {
            'shadow': self.shadow.state_dict(),
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
            'step': self.step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load EMA state from checkpoint.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing EMA state.
        """
        self.shadow.load_state_dict(state_dict['shadow'])
        self.decay = state_dict['decay']
        self.warmup_steps = state_dict['warmup_steps']
        self.step = state_dict['step']

    def to(self, device: str | torch.device) -> 'EMA':
        """
        Move EMA shadow model to device.

        Parameters
        ----------
        device : str or torch.device
            Target device.

        Returns
        -------
        EMA
            Self for chaining.
        """
        self.shadow = self.shadow.to(device)
        return self


class ema_scope:
    """
    Context manager to temporarily apply EMA weights to a model.

    Usage
    -----
    >>> ema = EMA(model)
    >>> # During training, update EMA after each step
    >>> ema.update(model)
    >>> # For evaluation, use EMA weights temporarily
    >>> with ema_scope(ema, model):
    ...     output = model(input)  # Uses EMA weights
    >>> # After context, model has original weights back
    """

    def __init__(self, ema: EMA, model: nn.Module) -> None:
        """
        Parameters
        ----------
        ema : EMA
            The EMA instance.
        model : nn.Module
            The model to temporarily apply EMA weights to.
        """
        self.ema = ema
        self.model = model
        self._backup: dict | None = None

    def __enter__(self) -> nn.Module:
        """Apply EMA weights and backup original weights."""
        self._backup = deepcopy(self.model.state_dict())
        self.ema.apply_shadow(self.model)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore original weights."""
        if self._backup is not None:
            self.model.load_state_dict(self._backup)
            self._backup = None
