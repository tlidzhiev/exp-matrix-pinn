from typing import Any, Literal

import torch.nn as nn

from .mlp import MLP
from .resnet import ResNet


def get_solver(name: Literal['mlp'], params: dict[str, Any]) -> nn.Module:
    match name.lower():
        case 'mlp':
            return MLP(**params)
        case 'resnet':
            return ResNet(**params)
        case _:
            raise ValueError(f"Unknown model name: {name}. Supported models: 'mlp', 'resnet'.")
