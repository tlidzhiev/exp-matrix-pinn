from typing import Any, Literal

import torch.nn as nn

from .film_mlp import FiLMMLP
from .mlp import MLP, SplitMLP
from .resnet import ResNet, SplitResNet


def get_solver(
    name: Literal['mlp', 'split_mlp', 'resnet', 'split_resnet', 'film_mlp'],
    params: dict[str, Any],
) -> nn.Module:
    match name.lower():
        case 'mlp':
            return MLP(**params)
        case 'split_mlp':
            return SplitMLP(**params)
        case 'resnet':
            return ResNet(**params)
        case 'split_resnet':
            return SplitResNet(**params)
        case 'film_mlp':
            return FiLMMLP(**params)
        case _:
            raise ValueError(
                f'Unknown model name: {name}. '
                f"Supported models: 'mlp', 'split_mlp', 'resnet', 'split_resnet', 'film_mlp'."
            )
