"""
__init__.py — forge 패키지 공개 API

M-07 Fix: wildcard import 제거 → 명시적 __all__ 정의.
NL-05 Fix: ones_like를 공개 API에 추가.
NL-07 Fix: Quota 대칭).

================================================================================
파일 이력 (Historical Metadata)
Created: 2026-08-12 12:14:52 +0900 (첫 커밋 기준)
Modified:
  - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
  - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
  - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
================================================================================
WHAT: AMEVA-Forge 패키지의 주요 컴포넌트들을 한 곳에 모아 외부에 노출하는 진입점 파일입니다.
WHY: 사용자가 패키지 내부 구조를 알 필요 없이 최상위 모듈에서 편리하게 필요한 기능들을 가져다 쓸 수 있도록 하기 위함입니다.
HOW: 내부의 각 서브모듈(ops, tensor, nn, optim 등)에서 필요한 클래스나 함수를 명시적으로 임포트하여 __all__ 리스트를 통해 노출합니다.
"""
# 하드웨어 디바이스 초기화 및 상태 관리 관련 모듈 임포트
from .device import init, is_available, current_device
# 기본 텐서 연산 및 수학/배열 조작 함수들을 ops 모듈에서 임포트
from .ops import (
    tensor, random, randn, matmul, relu, add, sub, mul, div, neg, transpose,
    ones_like, zeros_like, zeros, ones, full,  # NL-05: 공개 API에 추가
    to_numpy, to_numpy_async, dispose,
    unsqueeze, squeeze, flatten, permute, max_op, max_axis, var, std, sqrt,
    cat, where, pad, gather, scatter, dropout, conv2d, max_pool2d, avg_pool2d, bmm,
    maximum, minimum, clamp
)
# 텐서 객체 자체를 정의하는 클래스 및 GC 함수 임포트
from .tensor import Tensor, flush_gc
# 자동 미분 기능 관련 모듈 임포트
from .autograd import no_grad, enable_grad, is_grad_enabled, set_grad_enabled, set_max_graph_nodes
# 디버그 모드를 설정하고 가져오는 유틸리티 함수 임포트
from .ops import set_debug_mode, get_debug_mode
# 패키지 내에서 발생할 수 있는 여러 커스텀 에러 클래스 모음 임포트
from .errors import (
    AMEVAForgeError,
    AMEVAForgeShapeError,
    AMEVAForgeDTypeError,
    AMEVAForgeDeviceError,
    AMEVAForgeDisposedError,
    AMEVAForgeWebGPUUnavailableError,
    AMEVAForgeQuotaExceededError,   # NL-07: TypeScript와 대칭
    AMEVAForgeSecurityError,         # NL-07: TypeScript와 대칭
    AMEVAForgeValidationError,
    AMEVAForgeOutOfMemoryError,
    AMEVAForgeInternalGPUError,
    AMEVAForgeDeviceLostError,
    AMEVAForgeStaleHandleError,
    AMEVAForgeUnsupportedOperationError,
)

# WHAT: 패키지 버전을 정의하는 변수입니다.
# WHY: 패키지 사용자나 다른 도구가 현재 설치된 패키지의 버전을 확인할 수 있게 합니다.
# HOW: 문자열 형태로 버전을 저장하여 외부에서 접근 가능하게 합니다.
__version__ = "0.1.0"

# 기본 통계 및 수학 연산 함수들을 편의상 다른 이름으로 매핑하여 임포트
from .ops import sum_op as sum, mean_op as mean, exp_op as exp, log_op as log
from .ops import sigmoid, tanh_op as tanh, reshape, sum_axis
# 신경망 구성 요소가 있는 nn 서브모듈과 관련 레이어 임포트
from . import nn
from .nn import BatchNorm2d, Dropout, LayerNorm, MultiheadAttention, TransformerEncoderLayer, PositionalEncoding, RNNCell, LSTMCell, RNN, LSTM, RMSNorm, RotaryEmbedding, SwiGLU, ModuleList
# 최적화 알고리즘이 있는 optim 서브모듈과 학습률 스케줄러 임포트
from . import optim  
from .optim import clip_grad_norm, clip_grad_value, StepLR, CosineAnnealingLR, ReduceLROnPlateau
# 모델 저장 및 로드 유틸리티 임포트
from .serialization import save_model, load_model
# 함수형 API 묶음 임포트
from . import functional as F
from .functional import batch_norm2d, scaled_dot_product_attention, rms_norm, swiglu, rope
# 데이터 로드 유틸리티 임포트
from .data import DataLoader
# 고수준 AI 모델, 강화학습, 파이프라인 모듈 임포트
from . import models
from . import rl
from .pipeline import pipeline

# WHAT: 외부에서 `from forge import *`를 호출할 때 노출될 이름들의 리스트입니다.
# WHY: 패키지 내부에서만 쓰이는 숨겨진(private) 모듈이나 변수가 실수로 노출되지 않도록 제어하기 위해 사용합니다.
# HOW: 외부에 공개해야 할 함수, 클래스, 서브모듈들의 이름을 문자열 리스트로 모아서 정의합니다.
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
    "models",
    "rl",
    "pipeline",
    "F",
    "DataLoader",
    "ones",
    "full",
    "Tensor",
    "flush_gc",
    "no_grad",
    "BatchNorm2d",
    "Dropout",
    "LayerNorm",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "MultiheadAttention",
    "TransformerEncoderLayer",
    "PositionalEncoding",
    "RNNCell",
    "LSTMCell",
    "RNN",
    "LSTM",
    "batch_norm2d",
    "scaled_dot_product_attention",
    "rms_norm",
    "swiglu",
    "rope",
    "dropout",
    "bmm",
    # 연산
    "matmul",
    "relu",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "transpose",
    "cat",
    "where",
    "pad",
    "gather",
    "scatter",
    "conv2d",
    "max_pool2d",
    "avg_pool2d",
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
    "AMEVAForgeValidationError",
    "AMEVAForgeOutOfMemoryError",
    "AMEVAForgeInternalGPUError",
    "AMEVAForgeDeviceLostError",
    "AMEVAForgeStaleHandleError",
    "AMEVAForgeUnsupportedOperationError",
    # 디버그/설정/Autograd
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "set_grad_enabled",
    "set_debug_mode",
    "get_debug_mode",
    "set_max_graph_nodes",
    # 메타
    "__version__",
    "unsqueeze", "squeeze", "flatten", "permute", 
    "max_op", "max_axis", "var", "std", "sqrt", 
    "clip_grad_norm", "clip_grad_value",
    "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau",
    "save_model", "load_model"
]
