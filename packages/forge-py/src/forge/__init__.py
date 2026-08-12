"""
__init__.py — forge 패키지 공개 API

M-07 Fix: wildcard import 제거 → 명시적 __all__ 정의.
NL-05 Fix: ones_like를 공개 API에 추가.
NL-07 Fix: QuotaExceededError, SecurityError 추가 (TypeScript와 대칭).
"""
from .device import init, is_available, current_device
from .ops import (
    tensor, random, randn, matmul, relu, add, sub, mul, div, neg, transpose,
    ones_like, zeros_like, zeros, ones, full,  # NL-05: 공개 API에 추가
    to_numpy, to_numpy_async, dispose
)
from .tensor import Tensor
from .autograd import no_grad, set_max_graph_nodes
from .ops import set_debug_mode, get_debug_mode
from .errors import (
    AMEVAForgeError,
    AMEVAForgeShapeError,
    AMEVAForgeDTypeError,
    AMEVAForgeDeviceError,
    AMEVAForgeDisposedError,
    AMEVAForgeWebGPUUnavailableError,
    AMEVAForgeQuotaExceededError,   # NL-07: TypeScript와 대칭
    AMEVAForgeSecurityError,         # NL-07: TypeScript와 대칭
)

__version__ = "2.0.0"

from .ops import sum_op as sum, mean_op as mean, exp_op as exp, log_op as log
from .ops import sigmoid, tanh_op as tanh, reshape, sum_axis
from . import nn
from . import optim  
from . import functional as F
from .data import DataLoader


__all__ = [
    # 초기화
    "init",
    "is_available",
    "current_device",
    # 텐서 생성
    "tensor",
    "random",
    "randn",
    "ones_like",
    "zeros_like",
    "zeros",
    "sum",
    "mean",
    "exp",
    "log",
    "sigmoid",
    "tanh",
    "reshape",
    "sum_axis",
    "nn",
    "optim",
    "F",
    "DataLoader",
    "ones",
    "full",
    "Tensor",
    "no_grad",
    # 연산
    "matmul",
    "relu",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "transpose",
    # 유틸
    "to_numpy",
    "to_numpy_async",
    "dispose",
    # 에러
    "AMEVAForgeError",
    "AMEVAForgeShapeError",
    "AMEVAForgeDTypeError",
    "AMEVAForgeDeviceError",
    "AMEVAForgeDisposedError",
    "AMEVAForgeWebGPUUnavailableError",
    "AMEVAForgeQuotaExceededError",
    "AMEVAForgeSecurityError",
    # 디버그/설정
    "set_debug_mode",
    "get_debug_mode",
    "set_max_graph_nodes",
    # 메타
    "__version__",
]
