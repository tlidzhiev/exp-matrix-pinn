from typing import Literal

import torch.nn as nn


def _parse_act(activation: str) -> tuple[str, float | None]:
    """
    Parse activation string to extract name and optional parameter.

    Parameters
    ----------
    activation : str
        Activation string, optionally with parameter (e.g., "leaky_relu:0.2").

    Returns
    -------
    act_name : str
        Activation name in lowercase.
    param : float or None
        Activation parameter if provided, None otherwise.

    Raises
    ------
    ValueError
        If parameter string cannot be converted to float.
    """
    if ':' in activation:
        act_name, param_str = activation.split(':', 1)
        try:
            param = float(param_str)
        except ValueError:
            raise ValueError(
                f"Invalid activation parameter '{param_str}' in '{activation}'. "
                f'Parameter must be a number.'
            )
        return act_name.lower(), param
    return activation.lower(), None


def get_activation(activation: str) -> nn.Module:
    """
    Get activation module by name.

    Parameters
    ----------
    activation : str
        Activation name, optionally with parameter (e.g., "leaky_relu:0.2").
        Supported: "relu", "leaky_relu", "tanh", "silu", "gelu".

    Returns
    -------
    nn.Module
        PyTorch activation module.

    Raises
    ------
    ValueError
        If activation type is not supported.
    """
    act_name, param = _parse_act(activation)
    match act_name:
        case 'relu':
            return nn.ReLU()
        case 'leaky_relu':
            param = param if param is not None else 0.01
            return nn.LeakyReLU(param)
        case 'tanh':
            return nn.Tanh()
        case 'silu':
            return nn.SiLU()
        case 'gelu':
            return nn.GELU()
        case _:
            raise ValueError(
                f"Unknown activation type: '{act_name}'. Supported types: 'relu', 'leaky_relu', 'tanh', 'silu', 'gelu'."
            )


def initialize_weights(
    module: nn.Module,
    activation: str,
    mode: Literal['normal', 'uniform'],
) -> None:
    """
    Initialize weights for neural network module using He/Kaiming or Xavier/Glorot initialization.

    Parameters
    ----------
    module : nn.Module
        PyTorch module to initialize.
    activation : str
        Activation function name.
        Supported: "relu", "leaky_relu", "tanh", "silu", "gelu".
        For tanh, Xavier initialization with gain is used.
    mode : {'normal', 'uniform'}
        Initialization mode.

    Raises
    ------
    ValueError
        If mode or activation type is not supported.
    """
    if mode not in ['normal', 'uniform']:
        raise ValueError(
            f"Unknown initialization mode: '{mode}'. Supported modes: 'normal', 'uniform'."
        )

    act_name, param = _parse_act(activation)
    param = param if param is not None else 0.0

    if act_name not in ['relu', 'leaky_relu', 'tanh', 'silu', 'gelu']:
        raise ValueError(
            f"Unknown activation type for initialization: '{act_name}'. "
            f"Supported types: 'relu', 'leaky_relu', 'tanh', 'silu', 'gelu'."
        )

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if act_name == 'tanh':
                gain = nn.init.calculate_gain('tanh')
                init_fn = nn.init.xavier_normal_ if mode == 'normal' else nn.init.xavier_uniform_
                init_fn(m.weight, gain=gain)
            else:
                nonlin = 'leaky_relu' if act_name == 'leaky_relu' else 'relu'
                init_fn = nn.init.kaiming_normal_ if mode == 'normal' else nn.init.kaiming_uniform_
                init_fn(m.weight, a=param, nonlinearity=nonlin)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
