from typing import Any, Literal

import torch.nn as nn

from .baseline import Baseline
from .film import FiLM
from .resnet import ResNet, ResNetTC


def get_solver(
    name: Literal['baseline', 'resnet', 'resnet_tc', 'film'],
    params: dict[str, Any],
) -> nn.Module:
    match name.lower():
        case 'baseline':
            return Baseline(**params)
        case 'resnet':
            return ResNet(**params)
        case 'resnet_tc':
            return ResNetTC(**params)
        case 'film':
            return FiLM(**params)
        case _:
            raise ValueError(
                f'Unknown solver name: {name}. '
                f"Supported: 'baseline', 'resnet', 'resnet_tc', 'film'."
            )
