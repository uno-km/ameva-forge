from typing import Any, Sequence, Union, Optional
import numpy as np
from .tensor import Tensor, tensor
from . import nn as nn
from . import optim as optim
from . import functional as functional
from .device import is_available, init_webgpu

__version__: str

__all__ = [
    "Tensor",
    "tensor",
    "nn",
    "optim",
    "functional",
    "is_available",
    "init_webgpu",
    "__version__",
]
