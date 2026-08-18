# 🚀 AMEVA-Forge Release 1.0 Complete Verified Source Code Dump

**최종 수정일**: 2026-08-18  
**상태**: 100% Verified & Validated (Jest 95/95, Python 163/163, Playwright WebGPU 28/28 Pass)

이 문서는 AMEVA-Forge 1차 릴리즈의 TypeScript 코어 런타임, WGSL 셰이더 커널군, FFI 브리지, Python 프론트엔드 전체 소스코드의 전수 덤프입니다.

---

## 📂 파일 목차 (Table of Contents)

- [`packages/forge-py/src/forge/__init__.py`](#packages-forge-py-src-forge-__init__-py)
- [`packages/forge-py/src/forge/__init__.pyi`](#packages-forge-py-src-forge-__init__-pyi)
- [`packages/forge-py/src/forge/autograd.py`](#packages-forge-py-src-forge-autograd-py)
- [`packages/forge-py/src/forge/bridge.py`](#packages-forge-py-src-forge-bridge-py)
- [`packages/forge-py/src/forge/data.py`](#packages-forge-py-src-forge-data-py)
- [`packages/forge-py/src/forge/device.py`](#packages-forge-py-src-forge-device-py)
- [`packages/forge-py/src/forge/errors.py`](#packages-forge-py-src-forge-errors-py)
- [`packages/forge-py/src/forge/functional.py`](#packages-forge-py-src-forge-functional-py)
- [`packages/forge-py/src/forge/generated/__init__.py`](#packages-forge-py-src-forge-generated-__init__-py)
- [`packages/forge-py/src/forge/generated/op_schema.py`](#packages-forge-py-src-forge-generated-op_schema-py)
- [`packages/forge-py/src/forge/graph.py`](#packages-forge-py-src-forge-graph-py)
- [`packages/forge-py/src/forge/nn.py`](#packages-forge-py-src-forge-nn-py)
- [`packages/forge-py/src/forge/nn.pyi`](#packages-forge-py-src-forge-nn-pyi)
- [`packages/forge-py/src/forge/ops.py`](#packages-forge-py-src-forge-ops-py)
- [`packages/forge-py/src/forge/optim.py`](#packages-forge-py-src-forge-optim-py)
- [`packages/forge-py/src/forge/optim.pyi`](#packages-forge-py-src-forge-optim-pyi)
- [`packages/forge-py/src/forge/serialization.py`](#packages-forge-py-src-forge-serialization-py)
- [`packages/forge-py/src/forge/tensor.py`](#packages-forge-py-src-forge-tensor-py)
- [`packages/forge-py/src/forge/tensor.pyi`](#packages-forge-py-src-forge-tensor-pyi)
- [`packages/forge/src/bridge/pyodideBridge.ts`](#packages-forge-src-bridge-pyodidebridge-ts)
- [`packages/forge/src/bridge/safeCopy.ts`](#packages-forge-src-bridge-safecopy-ts)
- [`packages/forge/src/devtools/inspector.ts`](#packages-forge-src-devtools-inspector-ts)
- [`packages/forge/src/errors.ts`](#packages-forge-src-errors-ts)
- [`packages/forge/src/generated/opSchema.ts`](#packages-forge-src-generated-opschema-ts)
- [`packages/forge/src/index.ts`](#packages-forge-src-index-ts)
- [`packages/forge/src/tensor/broadcastParams.ts`](#packages-forge-src-tensor-broadcastparams-ts)
- [`packages/forge/src/tensor/dispatchShape.ts`](#packages-forge-src-tensor-dispatchshape-ts)
- [`packages/forge/src/tensor/gpuCore.ts`](#packages-forge-src-tensor-gpucore-ts)
- [`packages/forge/src/tensor/graphExecutor.ts`](#packages-forge-src-tensor-graphexecutor-ts)
- [`packages/forge/src/tensor/kernels/add.wgsl.ts`](#packages-forge-src-tensor-kernels-add-wgsl-ts)
- [`packages/forge/src/tensor/kernels/avgpool2d.wgsl.ts`](#packages-forge-src-tensor-kernels-avgpool2d-wgsl-ts)
- [`packages/forge/src/tensor/kernels/axpy.wgsl.ts`](#packages-forge-src-tensor-kernels-axpy-wgsl-ts)
- [`packages/forge/src/tensor/kernels/batched_matmul.wgsl.ts`](#packages-forge-src-tensor-kernels-batched_matmul-wgsl-ts)
- [`packages/forge/src/tensor/kernels/cat.wgsl.ts`](#packages-forge-src-tensor-kernels-cat-wgsl-ts)
- [`packages/forge/src/tensor/kernels/col2im.wgsl.ts`](#packages-forge-src-tensor-kernels-col2im-wgsl-ts)
- [`packages/forge/src/tensor/kernels/div.wgsl.ts`](#packages-forge-src-tensor-kernels-div-wgsl-ts)
- [`packages/forge/src/tensor/kernels/dropout.wgsl.ts`](#packages-forge-src-tensor-kernels-dropout-wgsl-ts)
- [`packages/forge/src/tensor/kernels/exp.wgsl.ts`](#packages-forge-src-tensor-kernels-exp-wgsl-ts)
- [`packages/forge/src/tensor/kernels/fill.wgsl.ts`](#packages-forge-src-tensor-kernels-fill-wgsl-ts)
- [`packages/forge/src/tensor/kernels/gather.wgsl.ts`](#packages-forge-src-tensor-kernels-gather-wgsl-ts)
- [`packages/forge/src/tensor/kernels/im2col.wgsl.ts`](#packages-forge-src-tensor-kernels-im2col-wgsl-ts)
- [`packages/forge/src/tensor/kernels/log.wgsl.ts`](#packages-forge-src-tensor-kernels-log-wgsl-ts)
- [`packages/forge/src/tensor/kernels/matmul.wgsl.ts`](#packages-forge-src-tensor-kernels-matmul-wgsl-ts)
- [`packages/forge/src/tensor/kernels/matmul_bias_relu.wgsl.ts`](#packages-forge-src-tensor-kernels-matmul_bias_relu-wgsl-ts)
- [`packages/forge/src/tensor/kernels/max.wgsl.ts`](#packages-forge-src-tensor-kernels-max-wgsl-ts)
- [`packages/forge/src/tensor/kernels/max_axis.wgsl.ts`](#packages-forge-src-tensor-kernels-max_axis-wgsl-ts)
- [`packages/forge/src/tensor/kernels/max_axis_backward.wgsl.ts`](#packages-forge-src-tensor-kernels-max_axis_backward-wgsl-ts)
- [`packages/forge/src/tensor/kernels/maxpool2d.wgsl.ts`](#packages-forge-src-tensor-kernels-maxpool2d-wgsl-ts)
- [`packages/forge/src/tensor/kernels/mul.wgsl.ts`](#packages-forge-src-tensor-kernels-mul-wgsl-ts)
- [`packages/forge/src/tensor/kernels/neg.wgsl.ts`](#packages-forge-src-tensor-kernels-neg-wgsl-ts)
- [`packages/forge/src/tensor/kernels/pad.wgsl.ts`](#packages-forge-src-tensor-kernels-pad-wgsl-ts)
- [`packages/forge/src/tensor/kernels/permute.wgsl.ts`](#packages-forge-src-tensor-kernels-permute-wgsl-ts)
- [`packages/forge/src/tensor/kernels/relu.wgsl.ts`](#packages-forge-src-tensor-kernels-relu-wgsl-ts)
- [`packages/forge/src/tensor/kernels/relu_backward.wgsl.ts`](#packages-forge-src-tensor-kernels-relu_backward-wgsl-ts)
- [`packages/forge/src/tensor/kernels/scatter.wgsl.ts`](#packages-forge-src-tensor-kernels-scatter-wgsl-ts)
- [`packages/forge/src/tensor/kernels/sigmoid.wgsl.ts`](#packages-forge-src-tensor-kernels-sigmoid-wgsl-ts)
- [`packages/forge/src/tensor/kernels/sigmoid_backward.wgsl.ts`](#packages-forge-src-tensor-kernels-sigmoid_backward-wgsl-ts)
- [`packages/forge/src/tensor/kernels/sub.wgsl.ts`](#packages-forge-src-tensor-kernels-sub-wgsl-ts)
- [`packages/forge/src/tensor/kernels/sum.wgsl.ts`](#packages-forge-src-tensor-kernels-sum-wgsl-ts)
- [`packages/forge/src/tensor/kernels/sum_axis.wgsl.ts`](#packages-forge-src-tensor-kernels-sum_axis-wgsl-ts)
- [`packages/forge/src/tensor/kernels/tanh.wgsl.ts`](#packages-forge-src-tensor-kernels-tanh-wgsl-ts)
- [`packages/forge/src/tensor/kernels/tanh_backward.wgsl.ts`](#packages-forge-src-tensor-kernels-tanh_backward-wgsl-ts)
- [`packages/forge/src/tensor/kernels/transpose.wgsl.ts`](#packages-forge-src-tensor-kernels-transpose-wgsl-ts)
- [`packages/forge/src/tensor/kernels/where.wgsl.ts`](#packages-forge-src-tensor-kernels-where-wgsl-ts)
- [`packages/forge/src/tensor/tensorRegistry.ts`](#packages-forge-src-tensor-tensorregistry-ts)
- [`packages/forge/src/tensor/validateDType.ts`](#packages-forge-src-tensor-validatedtype-ts)
- [`packages/forge/src/tensor/validateShape.ts`](#packages-forge-src-tensor-validateshape-ts)
- [`packages/forge/src/types.ts`](#packages-forge-src-types-ts)
- [`packages/forge/src/webgpu/buffers.ts`](#packages-forge-src-webgpu-buffers-ts)
- [`packages/forge/src/webgpu/device.ts`](#packages-forge-src-webgpu-device-ts)
- [`packages/forge/src/webgpu/pipelineCache.ts`](#packages-forge-src-webgpu-pipelinecache-ts)
- [`packages/forge/src/webgpu/quota.ts`](#packages-forge-src-webgpu-quota-ts)
- [`packages/forge/src/webgpu/shaderGuard.ts`](#packages-forge-src-webgpu-shaderguard-ts)
- [`packages/forge/src/webgpu/uniformPool.ts`](#packages-forge-src-webgpu-uniformpool-ts)
- [`packages/forge/src/webgpu/validateWasmRange.ts`](#packages-forge-src-webgpu-validatewasmrange-ts)

---

## `packages/forge-py/src/forge/__init__.py`

```python
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
    cat, where, pad, gather, scatter, dropout, conv2d, max_pool2d, avg_pool2d, bmm
)
# 텐서 객체 자체를 정의하는 클래스 및 GC 함수 임포트
from .tensor import Tensor, flush_gc
# 자동 미분 기능 관련 모듈 임포트
from .autograd import no_grad, set_max_graph_nodes
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
from .nn import BatchNorm2d, Dropout, LayerNorm, MultiheadAttention, TransformerEncoderLayer, PositionalEncoding, RNNCell, LSTMCell, RNN, LSTM
# 최적화 알고리즘이 있는 optim 서브모듈과 학습률 스케줄러 임포트
from . import optim  
from .optim import clip_grad_norm, clip_grad_value, StepLR, CosineAnnealingLR, ReduceLROnPlateau
# 모델 저장 및 로드 유틸리티 임포트
from .serialization import save_model, load_model
# 함수형 API 묶음 임포트
from . import functional as F
from .functional import batch_norm2d
# 데이터 로드 유틸리티 임포트
from .data import DataLoader

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
    "MultiheadAttention",
    "TransformerEncoderLayer",
    "PositionalEncoding",
    "RNNCell",
    "LSTMCell",
    "RNN",
    "LSTM",
    "batch_norm2d",
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
    # 디버그/설정
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

```

---

## `packages/forge-py/src/forge/__init__.pyi`

```typescript
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

```

---

## `packages/forge-py/src/forge/autograd.py`

```python
"""
=============================================================================
[파일 이력 메타데이터]
- Created: 2026-08-12 12:14:52 +0900
- Modified:
  - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
=============================================================================
자동 미분(Autograd) 엔진의 핵심 컴포넌트를 정의하는 모듈입니다.
"""
from typing import Any, Tuple, Optional, List
import contextlib

# _grad_mode는 현재 시스템이 그래디언트를 추적하고 있는지 여부를 나타내는 전역 상태 변수입니다.
# 기본값은 True이며, 그래디언트 계산이 필요한 연산들이 이 값을 참조하여 연산 그래프를 구성할지 결정합니다.
_grad_mode = True


@contextlib.contextmanager
def no_grad():
    """
    [WHAT] 
    그래디언트 추적을 임시로 비활성화하는 컨텍스트 매니저입니다.
    
    [WHY] 
    추론(Inference) 단계나 그래디언트 업데이트 시 불필요한 메모리 사용과 연산 오버헤드를 방지하기 위해 존재합니다.
    
    [HOW] 
    진입 시 전역 상태 변수인 _grad_mode의 기존 값을 저장하고 False로 변경한 뒤, 블록이 끝나면 원래 값으로 복원하는 방식으로 동작합니다.
    """
    # 전역 상태 변수를 수정하기 위해 global 키워드를 사용합니다.
    global _grad_mode
    # 현재 _grad_mode 값을 백업 변수인 prev에 저장합니다.
    prev = _grad_mode
    # 그래디언트 추적을 비활성화하기 위해 _grad_mode를 False로 설정합니다.
    _grad_mode = False
    try:
        # yield를 통해 제어권을 컨텍스트 블록 내부로 넘깁니다.
        yield
    finally:
        # 컨텍스트 블록의 실행이 끝나거나 예외가 발생하더라도 _grad_mode를 원래 상태로 반드시 복원합니다.
        _grad_mode = prev


class Context:
    """
    [WHAT] 
    순전파(Forward pass) 중에 계산된 중간 결과물들을 역전파(Backward pass)를 위해 저장하는 객체입니다.
    
    [WHY] 
    자동 미분을 수행하려면 순전파 때 사용된 입력 텐서나 중간값이 역전파의 그래디언트 계산 식에 필요한 경우가 많으므로 이를 안전하게 보관해야 합니다.
    
    [HOW] 
    인스턴스 변수인 saved_tensors 튜플에 필요한 데이터를 참조로 유지하며, save_for_backward 메서드를 통해 데이터를 저장합니다.
    """
    def __init__(self):
        """
        [WHAT] 
        Context 클래스의 생성자입니다.
        
        [WHY] 
        인스턴스가 생성될 때 텐서를 저장할 빈 공간을 초기화하기 위해 필요합니다.
        
        [HOW] 
        self.saved_tensors를 빈 튜플로 초기화하여 이후 텐서들이 불변 시퀀스 형태로 저장될 수 있게 준비합니다.
        """
        # 역전파 시 사용할 텐서들을 보관할 빈 튜플을 초기화합니다.
        self.saved_tensors = ()
        self.saved_versions = ()

    def save_for_backward(self, *args):
        """
        [WHAT] 주어진 인자들과 그 시점의 버전을 역전파 계산을 위해 저장하는 메서드입니다.
        """
        self.saved_tensors = args
        self.saved_versions = tuple(
            getattr(tensor, "_version", 0)
            for tensor in args
        )

    def validate_saved_tensor_versions(self):
        """
        [WHAT] 순전파 시 저장된 텐서들이 이후 in-place 수정되었는지 검증합니다.
        """
        for tensor, saved_version in zip(self.saved_tensors, self.saved_versions):
            current_version = getattr(tensor, "_version", 0)
            if current_version != saved_version:
                raise RuntimeError(
                    "Saved tensor was modified in-place after forward execution. "
                    "Re-run forward before backward."
                )


class Function:
    """
    [WHAT] 
    모든 미분 가능한 연산(Operation)의 베이스 클래스입니다.
    
    [WHY] 
    자동 미분을 지원하는 사용자 정의 연산을 일관된 인터페이스(forward, backward, apply)로 작성할 수 있도록 표준 구조를 제공하기 위해 존재합니다.
    
    [HOW] 
    순전파는 forward를, 역전파는 backward를 오버라이드하여 구현하며, 사용자는 apply 클래스 메서드를 통해 연산을 실행합니다.
    """
    @classmethod
    def apply(cls, *args, **kwargs):
        """
        [WHAT] 
        Forward pass를 실행하고 autograd 그래프를 구성합니다.
        
        [WHY] 
        단순한 순전파 실행뿐만 아니라, 결과 텐서가 이전 텐서들과 어떻게 연결되는지(의존성)를 기록하여 연산 그래프를 만들기 위해 필요합니다.
        
        [HOW] 
        _grad_mode 상태와 인자들의 requires_grad 여부를 확인한 뒤, forward를 실행하고 결과 텐서에 Context와 부모 노드 정보를 기록하여 그래프를 연결합니다.
        """
        # _grad_mode가 켜져 있고, 인자 중 하나라도 requires_grad 속성이 True이면 현재 연산도 그래디언트를 추적해야 함을 결정합니다.
        requires_grad = _grad_mode and any(
            (hasattr(a, 'requires_grad') and a.requires_grad) for a in args
        )

        # 현재 연산의 상태를 저장할 빈 Context 객체를 생성합니다.
        ctx = Context()
        # 생성한 ctx 객체와 함께 실제 순전파 로직인 forward 메서드를 호출하여 결과를 얻어옵니다.
        result = cls.forward(ctx, *args, **kwargs)

        if requires_grad:
            # 결과 텐서가 미분 가능해야 하므로 requires_grad를 True로 설정합니다.
            result.requires_grad = True
            # 역전파 시 필요한 데이터가 담긴 ctx를 결과 텐서에 연결합니다.
            result._ctx = ctx
            # 현재 텐서를 만들어낸 연산 클래스가 무엇인지 기록합니다.
            result._op_cls = cls
            # 역전파 과정에서 부모 노드로 거슬러 올라가기 위해, 미분 가능한 입력 인자들만 추려 _grad_parents 튜플로 저장합니다.
            # NC-05 Fix: autograd 부모를 _grad_parents에 저장 (레이지 그래프의 _parents를 덮어쓰지 않음)
            # NH-08 Fix: isinstance 대신 hasattr로 Tensor 여부 확인 (서브클래스도 처리)
            result._grad_parents = tuple(
                a for a in args
                if hasattr(a, 'requires_grad') and hasattr(a, 'shape')
            )

        return result

    @staticmethod
    def forward(ctx: Context, *args, **kwargs) -> Any:
        """
        [WHAT] 
        연산의 순전파 로직을 정의하는 정적 메서드입니다.
        
        [WHY] 
        각 연산(더하기, 곱하기 등)마다 실제 수행해야 할 계산 방식을 구체적으로 명시해야 하기 때문입니다.
        
        [HOW] 
        서브클래스에서 이 메서드를 오버라이드하여 입력 텐서들을 이용해 결과를 계산하고 반환하도록 구현하며, 필요 시 ctx에 중간값을 저장합니다.
        """
        # 베이스 클래스에서는 순전파가 정의되어 있지 않으므로 NotImplementedError를 발생시킵니다.
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, ...]:
        """
        [WHAT] 
        연산의 역전파 로직을 정의하는 정적 메서드입니다.
        
        [WHY] 
        출력에 대한 그래디언트(grad_output)가 주어졌을 때, 체인 룰(Chain Rule)을 이용해 입력에 대한 그래디언트를 계산하기 위해 필요합니다.
        
        [HOW] 
        서브클래스에서 이 메서드를 오버라이드하여, ctx에 저장된 데이터와 grad_output을 조합해 각 입력에 대응하는 그래디언트 튜플을 반환하도록 구현합니다.
        """
        # 베이스 클래스에서는 역전파가 정의되어 있지 않으므로 NotImplementedError를 발생시킵니다.
        raise NotImplementedError


# VUL-009: 그래프 노드 수 제한 설정
# 메모리 누수나 무한 루프를 방지하기 위해 생성 가능한 최대 그래프 노드 수를 경고하기 위한 설정값(기본 10000)입니다.
_max_graph_nodes_warning: int = 10000
# 그래프 노드 수가 이 값을 넘으면 강제로 에러를 발생시켜 프로세스 중단을 유도하는 제한값(None이면 무제한)입니다.
_max_graph_nodes_hard_limit: Optional[int] = None  # None = 무제한


def set_max_graph_nodes(warning: int = 10000, hard_limit: Optional[int] = None) -> None:
    """
    [WHAT] 
    Autograd 그래프 노드 수 제한을 설정하는 함수입니다.
    
    [WHY] 
    사용자가 모델 크기나 메모리 한계에 맞춰 그래프 허용치를 동적으로 조절하여, OOM(Out of Memory)과 같은 치명적 오류를 사전에 예방할 수 있도록 하기 위함입니다.
    
    [HOW] 
    전역 변수 _max_graph_nodes_warning과 _max_graph_nodes_hard_limit의 값을 인자로 받은 값으로 업데이트합니다.
    """
    # 전역 변수들을 수정할 수 있도록 global로 선언합니다.
    global _max_graph_nodes_warning, _max_graph_nodes_hard_limit
    # 경고 발생 임계값을 새로운 값으로 설정합니다.
    _max_graph_nodes_warning = warning
    # 강제 종료 임계값을 새로운 값으로 설정합니다.
    _max_graph_nodes_hard_limit = hard_limit


def build_topological_sort(root) -> List[Any]:
    """
    [WHAT] 
    주어진 루트 텐서부터 시작하여 역방향 연산 그래프의 위상 정렬(Topological sort) 리스트를 구축합니다.
    
    [WHY] 
    역전파 과정은 출력 노드에서 입력 노드 방향으로 순서대로 그래디언트를 전파해야 하므로, 노드들을 올바른 순서(자식에서 부모로)로 나열하기 위해 필요합니다.
    
    [HOW] 
    반복적 깊이 우선 탐색(Iterative DFS)을 사용하여 부모 노드(_grad_parents)를 재귀 없이 순회하며, 방문이 완료된 노드를 리스트에 추가하고 반환합니다.
    
    NH-08 Fix: id(v) 기반으로 visited 추적 (PyTorch, JAX 방식).
    Iterative DFS — 재귀 제거 (RecursionError 방지).
    backward를 위한 _grad_parents를 사용하여 그래프 탐색.
    VUL-009: 그래프 노드 수 warning/hard limit.
    """
    # 위상 정렬된 노드들을 순서대로 담아 반환할 결과 리스트입니다.
    topo = []
    # 이미 방문한 노드들의 메모리 주소(id)를 기록하여 중복 방문 및 사이클을 방지하는 집합입니다.
    visited = set()
    # 반복적 DFS를 위한 스택으로, 요소는 (현재 노드, 처리할 부모의 인덱스) 쌍을 담고 있습니다.
    stack = [(root, 0)]
    # 초기 루트 노드의 id를 방문 집합에 추가하여 방문 처리합니다.
    visited.add(id(root))
    # 노드 수가 경고 임계값을 넘었을 때 한 번만 경고 메시지를 출력하도록 상태를 기록하는 플래그 변수입니다.
    _warned = False

    # 스택이 비어있지 않은 동안 DFS 탐색을 계속 진행합니다.
    while stack:
        # 스택의 최상단에서 현재 노드와 그 노드의 탐색 중인 부모 인덱스를 가져옵니다.
        node, idx = stack[-1]
        # 현재 노드의 역전파 부모 노드들을 가져오되, 없을 경우 빈 튜플을 기본값으로 반환합니다.
        grad_parents = getattr(node, '_grad_parents', ())
        
        if idx < len(grad_parents):
            # 아직 탐색할 부모 노드가 남아있다면, 현재 노드의 상태를 (idx + 1)로 갱신하여 스택을 업데이트합니다.
            stack[-1] = (node, idx + 1)
            # 다음으로 방문할 부모 노드 객체를 가져옵니다.
            p = grad_parents[idx]
            # 해당 부모 노드의 고유 메모리 주소값을 구합니다.
            pid = id(p)
            
            if pid not in visited:
                # 해당 부모 노드를 아직 방문하지 않았다면 방문 집합에 추가합니다.
                visited.add(pid)
                # 새로운 부모 노드 탐색을 시작하기 위해 스택에 추가(인덱스 0)합니다.
                stack.append((p, 0))

                # 현재까지 방문한 전체 노드 수를 계산하여 노드 제한을 확인합니다.
                node_count = len(visited)
                if _max_graph_nodes_hard_limit is not None and node_count > _max_graph_nodes_hard_limit:
                    # 방문한 노드 수가 강제 제한 임계치를 초과했다면 RuntimeError를 발생시킵니다.
                    raise RuntimeError(
                        f"Autograd graph exceeds hard limit of {_max_graph_nodes_hard_limit} nodes. "
                        f"Use set_max_graph_nodes(hard_limit=N) to adjust."
                    )
                if not _warned and node_count > _max_graph_nodes_warning:
                    # 노드 수가 경고 임계치를 넘었고 아직 경고하지 않았다면 warnings 모듈을 통해 경고를 발생시킵니다.
                    import warnings
                    warnings.warn(
                        f"[AMEVA] Autograd graph has {node_count} nodes (warning threshold: "
                        f"{_max_graph_nodes_warning}). This may consume excessive memory. "
                        f"Consider checkpointing or reducing computation depth.",
                        RuntimeWarning, stacklevel=2
                    )
                    # 중복 경고를 방지하기 위해 플래그를 True로 변경합니다.
                    _warned = True
        else:
            # 현재 노드의 모든 부모 노드들에 대한 탐색이 끝났다면 스택에서 꺼냅니다.
            stack.pop()
            # 탐색이 완료된 노드는 자식들의 의존성이 모두 해결되었으므로 결과 리스트(topo)에 추가합니다.
            topo.append(node)

    # 역방향으로 위상 정렬된 모든 텐서들의 리스트를 반환합니다.
    return topo


```

---

## `packages/forge-py/src/forge/bridge.py`

```python
"""
=============================================================================
[파일 이력 메타데이터]
- Created: 2026-08-12 12:14:52 +0900
- Modified:
  - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
=============================================================================
bridge.py — Python ↔ TypeScript FFI 미들웨어

NC-01 Fix: is_pyodide() SyntaxError 수정 (try 블록 미완성 → 단순 sys.platform 체크)
NC-02 Fix: Dict 타입힌트 미정의 → Optional[dict] 사용
NC-03 Fix: js_execute_graph()의 *inputs 스프레드 → to_js() 단일 배열 전달
NH-06 Fix: flush_gc() 원자성 — dispose 성공 후에만 큐에서 제거
M-06 Fix: batch dispose 구현
"""
from typing import Any, List, Optional
import sys
import warnings

from .errors import (
    AMEVAForgeValidationError,
    AMEVAForgeOutOfMemoryError,
    AMEVAForgeInternalGPUError,
    AMEVAForgeDeviceLostError,
    AMEVAForgeQuotaExceededError
)


def is_pyodide() -> bool:
    """
    [WHAT] 
    현재 런타임 환경이 Pyodide(웹 어셈블리 브라우저 환경)인지 검사합니다.
    
    [WHY] 
    브라우저 외부(Node.js, CPython)와 내부 환경 간의 동작을 다르게 분기처리하여, FFI 호출 시 에러를 방지하기 위함입니다.
    
    [HOW] 
    파이썬의 sys.platform 값이 'emscripten' 문자열과 일치하는지 비교하여 불리언 결과를 반환합니다.
    """
    # 현재 시스템 플랫폼 이름이 Emscripten 기반인지 확인하여 결과를 반환합니다.
    return sys.platform == 'emscripten'


def get_js_core() -> Any:
    """
    [WHAT] 
    자바스크립트 전역 객체인 globalThis.amevaForge 바인딩을 찾아 반환합니다.
    
    [WHY] 
    파이썬 코드에서 자바스크립트로 구현된 WebGPU 엔진의 메서드들에 접근하고 통신할 수 있는 브릿지 객체가 필요하기 때문입니다.
    
    [HOW] 
    Pyodide의 내장 js 모듈을 임포트한 뒤, globalThis에 해당 속성이 존재하는지 검사하고 있으면 객체를 반환, 없으면 RuntimeError를 던집니다.
    """
    # Pyodide에서 자바스크립트 네임스페이스에 접근하기 위해 js 모듈을 임포트합니다.
    import js
    if not hasattr(js.globalThis, 'amevaForge'):
        # JS 전역 공간에 엔진 코어 객체가 등록되어 있지 않다면 치명적인 에러를 발생시켜 실행을 중단합니다.
        raise RuntimeError(
            "[AMEVA Tensor] WebGPU core not found on globalThis.amevaForge. "
            "Ensure registerPyodideBridge() was called before init()."
        )
    # 확인된 JS 엔진 코어 객체를 파이썬 쪽으로 반환합니다.
    return js.globalThis.amevaForge


def init_bridge(config: Optional[dict] = None) -> Any:
    """
    [WHAT] 
    자바스크립트 측 WebGPU 엔진의 초기화를 트리거하는 함수입니다.
    
    [WHY] 
    GPU 디바이스와 렌더링 컨텍스트 등의 필수 인프라를 사용하기 전에 올바르게 셋업되도록 보장해야 하기 때문입니다.
    
    [HOW] 
    get_js_core()로 JS 바인딩을 얻어온 뒤, 전달받은 config 사전을 인자로 넘기며 엔진 코어의 init 메서드를 호출합니다.
    """
    # 전역 JS 브릿지 객체를 가져옵니다.
    core = get_js_core()
    # 브릿지 객체의 init 메서드를 호출하여 설정값을 전달하고 실행합니다.
    return core.init(config)


async def js_map_async(handle: str) -> None:
    """
    [WHAT] 
    자바스크립트의 mapBufferAsync 함수를 비동기적으로 호출하는 함수입니다.
    
    [WHY] 
    WebGPU 버퍼 메모리를 파이썬(CPU) 측에서 읽거나 쓸 수 있도록 GPU 메모리를 매핑하는 비동기 처리가 필요하기 때문입니다.
    
    [HOW] 
    get_js_core()를 통해 얻은 JS 코어 객체의 mapBufferAsync 메서드를 await 키워드를 통해 대기(await)하며 실행합니다.
    """
    # 전역 JS 브릿지 객체를 가져옵니다.
    core = get_js_core()
    # 주어진 버퍼 핸들에 대해 메모리 매핑을 요청하고 비동기적으로 완료되기를 기다립니다.
    await core.mapBufferAsync(handle)


def js_read_mapped_into(handle: str, arr: Any) -> None:
    """
    [WHAT] 
    매핑된 GPU 버퍼의 데이터를 파이썬 측의 배열(예: TypedArray, 메모리 뷰)로 직접 복사해오는 함수입니다.
    
    [WHY] 
    GPU 연산이 끝난 결과를 파이썬에서 접근 가능한 numpy 배열이나 리스트 형태로 변환하기 위해 데이터를 읽어야 하기 때문입니다.
    
    [HOW] 
    JS 코어의 readMappedInto 메서드에 버퍼 핸들과 데이터를 받을 대상 배열 객체를 넘겨주어 메모리 복사를 수행합니다.
    """
    # 전역 JS 브릿지 객체를 가져옵니다.
    core = get_js_core()
    # 매핑된 버퍼의 내용을 대상 객체(arr) 내부로 복사합니다.
    core.readMappedInto(handle, arr)


def js_dispose(handle: str) -> None:
    """
    [WHAT] 
    자바스크립트 측에 할당된 단일 GPU 버퍼 리소스를 해제(dispose)합니다.
    
    [WHY] 
    더 이상 사용되지 않는 텐서 메모리를 방치하면 GPU OOM(Out of Memory)이 발생하므로, 가비지 컬렉터와 연계하여 메모리를 직접 해제해주기 위함입니다.
    
    [HOW] 
    핸들이 유효한지 검사한 후, JS 코어 객체의 dispose 메서드에 핸들을 넘겨 리소스를 삭제하며, 예외 발생 시 무시(pass)합니다.
    """
    if handle is None:
        # 삭제할 핸들이 없으면 아무 작업도 하지 않고 리턴합니다.
        return
    try:
        # 전역 JS 브릿지 객체를 가져옵니다.
        core = get_js_core()
        # 해당 핸들이 가리키는 버퍼 리소스를 해제하도록 요청합니다.
        core.dispose(handle)
    except Exception as e:
        warnings.warn(f"[AMEVA Bridge] GPU handle disposal failed for {handle}: {e}", RuntimeWarning)


def js_dispose_batch(handles: list) -> None:
    """
    [WHAT] 
    여러 개의 GPU 버퍼 핸들을 리스트 형태로 받아 한 번의 FFI 호출로 일괄 해제합니다 (M-06 Fix).
    
    [WHY] 
    단일 해제를 반복 호출하는 것은 파이썬과 JS 경계를 넘나드는 FFI 오버헤드가 크기 때문에, 성능을 개선하기 위해 배치 처리를 도입했습니다.
    
    [HOW] 
    핸들 리스트를 JS 배열로 변환(to_js)한 뒤, 코어에 disposeBatch 메서드가 있으면 한 번에 호출하고, 없으면 fallback으로 순회하며 개별 해제합니다.
    """
    if not handles:
        # 해제할 핸들 목록이 비어있다면 즉시 반환하여 불필요한 실행을 막습니다.
        return
    # 전역 JS 브릿지 객체를 가져옵니다.
    core = get_js_core()
    # 파이썬 객체를 JS 객체로 변환하기 위한 유틸리티를 Pyodide에서 가져옵니다.
    from pyodide.ffi import to_js
    js_handles = to_js(handles)
    try:
        if hasattr(core, 'disposeBatch'):
            # JS 측에 배치 해제 기능이 존재한다면, 변환된 배열을 넘겨 일괄 해제를 요청합니다.
            core.disposeBatch(js_handles)
        else:
            # JS 측에 배치 기능이 없다면, 반복문을 돌며 기존 단일 해제 메서드를 사용합니다.
            for h in handles:
                # 리스트에 담긴 개별 핸들에 대해 순차적으로 해제를 요청합니다.
                core.dispose(h)
    finally:
        try:
            if hasattr(js_handles, 'destroy'):
                js_handles.destroy()
        except Exception:
            pass


def _map_js_error(e: Exception) -> None:
    """Map JS error names to Python typed exceptions."""
    msg = str(e)
    if 'AMEVAForgeValidationError' in msg:
        raise AMEVAForgeValidationError(msg) from e
    elif 'AMEVAForgeOutOfMemoryError' in msg or 'OOM' in msg:
        raise AMEVAForgeOutOfMemoryError(msg) from e
    elif 'AMEVAForgeInternalGPUError' in msg:
        raise AMEVAForgeInternalGPUError(msg) from e
    elif 'AMEVAForgeDeviceLostError' in msg or 'device lost' in msg.lower():
        raise AMEVAForgeDeviceLostError(msg) from e
    elif 'AMEVAForgeQuotaExceededError' in msg:
        raise AMEVAForgeQuotaExceededError(msg) from e

async def js_execute_graph(instructions_json: str, inputs) -> dict:
    """Execute graph via JS bridge (async - executeGraph returns Promise)."""
    core = get_js_core()
    js_inputs = None
    result_proxy = None
    try:
        # Convert inputs to JS
        if inputs is not None:
            from pyodide.ffi import to_js
            # Validate float32
            for arr in inputs:
                if hasattr(arr, 'dtype') and str(arr.dtype) != 'float32':
                    raise TypeError(f"Expected float32, got {arr.dtype}")
            js_inputs = to_js(inputs, depth=1)
        
        # Call async executeGraph — returns Promise in Pyodide
        result_proxy = await core.executeGraph(instructions_json, js_inputs)
        
        # Convert JS result to Python dict
        result = result_proxy.to_py()
        return result
    except Exception as e:
        _map_js_error(e)
        raise
    finally:
        if js_inputs is not None:
            try:
                js_inputs.destroy()
            except Exception as e:
                import warnings
                warnings.warn(f"[AMEVA Bridge] Proxy cleanup failed: {e}", RuntimeWarning)
        if result_proxy is not None:
            try:
                result_proxy.destroy()
            except Exception as e:
                import warnings
                warnings.warn(f"[AMEVA Bridge] Proxy cleanup failed: {e}", RuntimeWarning)


```

---

## `packages/forge-py/src/forge/data.py`

```python
"""
=============================================================================
[파일 이력 메타데이터]
- Created: 2026-08-12 12:14:52 +0900
- Modified:
  - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
=============================================================================
데이터 로딩 및 미니배치 생성을 담당하는 유틸리티 모듈입니다.
"""
import numpy as np
from .ops import tensor
from .tensor import Tensor


class DataLoader:
    """
    [WHAT] 
    모델 학습에 사용할 데이터를 일정한 크기의 미니배치(mini-batch)로 쪼개어 제공하는 이터레이터 클래스입니다.
    
    [WHY] 
    전체 데이터를 한 번에 GPU 메모리에 올리면 OOM이 발생할 수 있으므로, 설정한 batch_size만큼 분할하여 효율적인 학습 반복을 수행하기 위해 필요합니다.
    
    [HOW] 
    생성자에서 입력과 라벨 데이터를 저장하고 인덱스를 셔플링할지 결정하며, __iter__ 메서드를 통해 미니배치 분량만큼의 데이터를 Tensor 객체로 래핑해 반환합니다.
    
    VUL-010 Fix: label_dtype 파라미터 추가.
      - 'float32' (기본): regression/one-hot target용
      - 'int64': classification 정수 라벨 유지
      - 'auto': y_data의 원본 dtype이 정수형이면 int64, 아니면 float32
    """
    def __init__(self, x_data, y_data, batch_size=32, shuffle=True, label_dtype='auto'):
        """
        [WHAT] 
        DataLoader 클래스의 인스턴스를 초기화하는 생성자입니다.
        
        [WHY] 
        데이터셋 원본을 클래스 내부에 보관하고, 미니배치 크기나 셔플 여부, 라벨의 데이터 타입 같은 동작 설정값들을 인스턴스에 저장하기 위함입니다.
        
        [HOW] 
        입력 특성인 x_data를 float32 numpy 배열로 변환하고, y_data의 경우 label_dtype 옵션에 맞춰 적절한 numpy 자료형을 선택한 후 타입 변환을 수행하여 인스턴스 변수에 할당합니다.
        """
        # 입력 데이터 x_data를 가져와 모든 연산의 기본인 float32 타입의 numpy 배열로 강제 변환하여 저장합니다.
        self.x_data = np.array(x_data, dtype=np.float32)

        # VUL-010: 라벨 dtype 보존
        # 라벨 데이터 y_data도 우선 기본적인 numpy 배열로 캐스팅합니다.
        y_arr = np.array(y_data)
        
        if label_dtype == 'auto':
            # 사용자가 auto로 설정한 경우 원본 배열의 데이터 타입을 확인합니다.
            if np.issubdtype(y_arr.dtype, np.integer):
                # 원본 타입이 정수 계열이라면 손실 없이 분류 문제를 풀기 위해 int64 타입을 선택합니다.
                self._label_dtype = np.int64
            else:
                # 정수형이 아니라면 기본적으로 회귀 문제나 원-핫 인코딩으로 간주해 float32 타입을 선택합니다.
                self._label_dtype = np.float32
        elif label_dtype == 'int64':
            # 사용자가 명시적으로 int64를 요구했다면 해당 타입을 선택합니다.
            self._label_dtype = np.int64
        else:
            # 그 외의 모든 경우(주로 float32가 들어옴)에는 기본적으로 float32 타입을 선택합니다.
            self._label_dtype = np.float32

        # 위에서 결정된 타입(_label_dtype)을 적용하여 y_data를 안전하게 형변환한 최종 라벨 배열을 저장합니다.
        self.y_data = y_arr.astype(self._label_dtype)
        # 미니배치 한 개당 몇 개의 샘플을 포함할지 결정하는 batch_size 크기를 저장합니다.
        self.batch_size = batch_size
        # 에포크마다 데이터를 무작위로 섞을지 여부를 결정하는 boolean 플래그를 저장합니다.
        self.shuffle = shuffle

    def __iter__(self):
        """
        [WHAT] 
        클래스 객체를 반복 가능(iterable)하게 만들어주는 매직 메서드입니다.
        
        [WHY] 
        파이썬의 for 루프에서 이 로더를 바로 순회하면서 미니배치 쌍(x_batch, y_batch)을 하나씩 꺼내어 학습 루프에 공급해야 하기 때문입니다.
        
        [HOW] 
        데이터의 총 개수만큼 인덱스 배열을 생성하고(옵션에 따라 셔플 적용), batch_size만큼 슬라이싱한 인덱스를 이용해 데이터를 추출, Tensor 형태로 래핑하여 차례대로 yield 합니다.
        """
        # 전체 데이터 셋의 총 샘플 개수(n)를 구합니다.
        n = len(self.x_data)
        # 0부터 n-1까지의 연속된 정수 인덱스 배열을 생성합니다.
        indices = np.arange(n)
        
        if self.shuffle:
            # 셔플 플래그가 True라면 인덱스 배열을 무작위 순서로 섞어 배치 구성이 랜덤하게 이루어지게 합니다.
            np.random.shuffle(indices)
            
        # 0부터 n까지 batch_size만큼의 간격(step)으로 루프를 돌며 배치의 시작 인덱스(start)를 잡습니다.
        for start in range(0, n, self.batch_size):
            # 시작 인덱스에서 batch_size를 더하여 배치의 끝 인덱스(end)를 구하되, 총 개수 n을 초과하지 않도록 보정합니다.
            end = min(start + self.batch_size, n)
            # 설정된 범위(start:end)만큼 인덱스 배열을 슬라이싱하여 현재 배치의 데이터 인덱스 묶음을 만듭니다.
            batch_idx = indices[start:end]
            # 해당 인덱스에 매칭되는 입력 특성 데이터를 추출하고 Tensor 객체로 래핑하여 텐서 그래프에 연결할 준비를 합니다.
            x_batch = tensor(self.x_data[batch_idx])
            # 라벨은 원본 dtype 유지하여 Tensor 생성
            # 해당 인덱스에 매칭되는 라벨 원본 데이터를 추출합니다.
            y_batch_data = self.y_data[batch_idx]
            
            if self._label_dtype == np.int64:
                # 정수 라벨: float32 Tensor로 변환하되 값은 정수로 유지
                # cross_entropy 등에서 .astype(np.int64)로 복원 가능
                # 현재 엔진이 float32 텐서 구조를 가정하므로 우선 float32로 캐스팅하여 Tensor 객체를 생성합니다.
                y_batch = tensor(y_batch_data.astype(np.float32))
            else:
                # 라벨 타입이 float32 계열이라면 별도의 변환 없이 곧바로 Tensor 객체를 생성합니다.
                y_batch = tensor(y_batch_data)
                
            # 구성이 완료된 미니배치 쌍(x_batch, y_batch)을 호출자(학습 루프 등)에게 반환하고 실행 상태를 일시정지합니다.
            yield x_batch, y_batch

    def __len__(self):
        """
        [WHAT] 
        전체 데이터셋을 처리할 때 발생할 수 있는 전체 미니배치(스텝)의 개수를 반환합니다.
        
        [WHY] 
        진행률(progress bar) 표시, 스텝 단위 스케줄러 업데이트 등 학습 중에 총 반복 횟수를 사전에 알아야 할 때가 많기 때문입니다.
        
        [HOW] 
        전체 데이터 크기에서 batch_size - 1을 더한 값을 batch_size로 나누는 올림 나눗셈을 통해 총 미니배치 수를 정수로 계산하여 반환합니다.
        """
        # 총 데이터 갯수를 배치 사이즈로 나눈 후 남은 나머지 데이터도 한 배치를 구성하게 되므로 이 식을 이용해 총 배치 수를 도출하여 반환합니다.
        return (len(self.x_data) + self.batch_size - 1) // self.batch_size


```

---

## `packages/forge-py/src/forge/device.py`

```python
"""
device.py — WebGPU 디바이스 초기화 및 상태 관리

[역사적 메타데이터]
- Created: Wed Aug 12 12:14:52 2026 +0900 (초기 커밋)
- Modified:
  - Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories

H-03 Fix: 무성(silent) CPU 폴백 완전 제거.
L-04 Fix: is_pyodide(), init_bridge()를 bridge.py에서 올바르게 임포트.
NM-04 Fix: JS Promise await 방식을 Pyodide 버전 독립적으로 처리.
NM-08 Fix: init() 호출 상태를 추적하여 미초기화 상태에서 GPU 텐서 사용 시 경고.
"""
from typing import Any, Optional
from .bridge import is_pyodide, init_bridge
from .errors import AMEVAForgeWebGPUUnavailableError

# 현재 사용 중인 디바이스 상태를 저장하는 전역 변수 ('cpu' 또는 'gpu')
# 무엇을: 활성화된 디바이스 상태를 문자열로 보관한다.
# 왜: 현재 시스템이 CPU를 쓰는지 GPU를 쓰는지 전역적으로 추적하기 위해 존재한다.
# 어떻게: 초기화 전에는 'cpu'를 가지며, WebGPU가 성공적으로 로드된 후 'gpu'로 변경된다.
_CURRENT_DEVICE: str = "cpu"

# WebGPU 엔진 초기화 함수(init)가 호출되었는지 여부를 저장하는 전역 변수
# 무엇을: 시스템 초기화 완료 여부를 불리언 값으로 보관한다.
# 왜: 초기화되지 않은 상태에서 GPU 자원을 요청할 때 적절한 경고를 주기 위해 존재한다.
# 어떻게: init() 함수가 성공적으로 완료된 후 True로 상태가 변경된다.
_INIT_CALLED: bool = False


async def init(experimental_zero_copy: bool = False) -> None:
    """
    WebGPU 엔진을 초기화한다.

    무엇을: 비동기 방식으로 브라우저의 WebGPU 기능을 활성화하고 런타임 환경을 셋업하는 역할을 한다.
    왜: WebGPU 기반의 텐서 연산을 수행하기 위해 반드시 브라우저 환경 및 JS 브릿지가 초기화되어야 하기 때문이다.
    어떻게: Pyodide 환경인지 검증한 뒤 JS 브릿지를 로드하며, 반환된 JS Promise를 Python의 asyncio Future로 매핑하여 비동기 대기를 수행한다.

    H-03: 초기화 실패 시 CPU 폴백 없이 AMEVAForgeWebGPUUnavailableError를 던진다.
    NM-04: Pyodide JS Promise await를 버전 독립적으로 처리한다.

    Raises:
        AMEVAForgeWebGPUUnavailableError: Pyodide 환경이 아니거나 WebGPU 초기화 실패 시.
        RuntimeError: JS 브릿지 연결 실패 시.
    """
    global _CURRENT_DEVICE, _INIT_CALLED

    if not is_pyodide():
        raise AMEVAForgeWebGPUUnavailableError(
            "AMEVA-Forge requires a Pyodide (browser/WASM) environment. "
            "Non-browser Python runtimes are not supported for GPU execution."
        )

    # JS 브릿지 초기화 (실패하면 예외 전파 — 폴백 없음)
    res = init_bridge({"experimental_zero_copy": experimental_zero_copy})

    if res is not None:
        import asyncio
        if hasattr(res, 'then') and callable(getattr(res, 'then', None)):
            print("[device.py] JS Promise detected. Awaiting via asyncio.Future bridge...")
            future = asyncio.get_running_loop().create_future()
            def resolve(val): 
                # 무엇을: JS Promise가 성공적으로 완료(resolve)되었을 때 호출될 콜백 함수이다.
                # 왜: JS에서 완료된 결과를 Python의 asyncio.Future에 성공 상태로 반영하기 위해 존재한다.
                # 어떻게: Future 객체가 아직 완료되지 않았다면 set_result 메서드를 통해 결과 값을 설정한다.
                if not future.done(): future.set_result(val)
            def reject(err): 
                # 무엇을: JS Promise가 실패(reject)되었을 때 호출될 에러 처리 콜백 함수이다.
                # 왜: JS 측에서 발생한 에러를 Python 레벨의 예외로 전환 및 전파하기 위해 존재한다.
                # 어떻게: Future 객체가 아직 완료되지 않았다면 RuntimeError와 함께 set_exception을 호출하여 예외를 발생시킨다.
                if not future.done(): future.set_exception(RuntimeError(str(err)))
            res.then(resolve).catch(reject)
            await future
            print("[device.py] JS Promise resolved successfully.")
        else:
            print(f"[device.py] res is not a Promise. Type: {type(res).__name__}")
            try:
                await res
            except TypeError:
                pass

    _CURRENT_DEVICE = "gpu"
    _INIT_CALLED = True



def is_available() -> bool:
    """
    현재 WebGPU가 초기화되어 있는지 반환한다.
    
    무엇을: 현재 실행 환경에서 WebGPU 사용이 가능한 상태인지 여부를 불리언(Boolean) 값으로 반환한다.
    왜: 텐서 연산 시 사용 가능한 디바이스(WebGPU 지원 여부)에 따라 동적인 분기를 처리하기 위해 필요하다.
    어떻게: _CURRENT_DEVICE 전역 변수의 값이 "gpu"인지 직접 비교 연산하여 그 결과를 반환한다.
    """
    return _CURRENT_DEVICE == "gpu"


def current_device() -> str:
    """
    현재 기본 디바이스 문자열을 반환한다 ('gpu' 또는 'cpu').
    
    무엇을: 현재 시스템에 활성화된 연산 디바이스의 이름을 문자열 형태로 반환한다.
    왜: 텐서 생성 시 디폴트 디바이스 값을 결정하거나 현재 활성 디바이스 상태를 외부 모듈에서 조회할 수 있도록 하기 위해 존재한다.
    어떻게: _CURRENT_DEVICE 전역 변수에 저장된 문자열 값을 단순히 리턴하여 제공한다.
    """
    return _CURRENT_DEVICE


def check_gpu_initialized() -> None:
    """
    NM-08 Fix: GPU 텐서 사용 전 초기화 상태를 검사한다.
    초기화되지 않은 상태에서 GPU 텐서를 생성하면 경고를 출력한다.
    
    무엇을: GPU 자원을 사용하려는 시점에서 엔진이 올바르게 초기화(init)되었는지 확인하고, 그렇지 않다면 런타임 경고 메시지를 발생시킨다.
    왜: 사용자가 실수로 초기화 함수를 호출하지 않고 GPU 텐서를 할당하려 할 때, 나중에 연산 시점에서 발생할 수 있는 크래시에 대해 선제적으로 안내하기 위함이다.
    어떻게: 전역 상태인 _INIT_CALLED와 _CURRENT_DEVICE를 조건문으로 검사하고, 초기화 요건을 충족하지 못했다면 warnings 모듈을 통해 적절한 경고 메시지를 띄운다.
    """
    if not _INIT_CALLED or _CURRENT_DEVICE != "gpu":
        import warnings
        warnings.warn(
            "GPU device is not initialized. Call 'await at.init()' before creating GPU tensors. "
            "Current device is 'cpu'. GPU operations will fail at realize() time.",
            RuntimeWarning,
            stacklevel=3
        )

```

---

## `packages/forge-py/src/forge/errors.py`

```python
"""
errors.py — AMEVA-Forge 에러 계층 구조

[역사적 메타데이터]
- Created: Wed Aug 12 12:14:52 2026 +0900 (초기 커밋)
- Modified:
  - Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories

NL-07 Fix: Python errors.py와 TypeScript errors.ts의 에러 클래스 대칭 맞춤.
  추가: AMEVAForgeQuotaExceededError, AMEVAForgeSecurityError
"""


class AMEVAForgeError(Exception):
    """
    AMEVA-Forge 기본 에러 클래스.
    
    무엇을: AMEVA-Forge 패키지 내에서 발생하는 모든 사용자 정의 예외의 최상위 부모 클래스이다.
    왜: 모든 내부 에러를 하나의 예외 타입으로 묶어, 사용자가 AMEVAForgeError 하나만으로 모든 라이브러리 예외를 캐치할 수 있게 하기 위함이다.
    어떻게: 파이썬의 내장 Exception 클래스를 상속받아 구현되었다.
    """
    pass


class AMEVAForgeShapeError(AMEVAForgeError):
    """
    텐서 shape 불일치 또는 유효하지 않은 shape.
    
    무엇을: 텐서 연산 시 요구되는 크기(Shape) 조건이 맞지 않을 때 발생하는 에러 클래스이다.
    왜: 행렬 곱, 덧셈 등 형태(Shape)가 일치해야 하는 연산에서 차원 불일치를 명확히 알리기 위해 존재한다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeDTypeError(AMEVAForgeError):
    """
    지원하지 않는 dtype 또는 dtype 불일치.
    
    무엇을: 텐서의 데이터 타입(dtype)이 지원되지 않거나, 연산 간 데이터 타입이 맞지 않을 때 던지는 에러 클래스이다.
    왜: 호환되지 않는 데이터 타입끼리의 연산을 시도하거나 시스템이 지원하지 않는 타입을 사용할 때 이를 차단하기 위함이다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeDeviceError(AMEVAForgeError):
    """
    디바이스 관련 에러 (기기 불일치, 초기화 실패 등).
    
    무엇을: CPU와 GPU 텐서 간의 연산 등 이기종 디바이스 연산 시도 또는 초기화 문제 발생 시 던지는 에러이다.
    왜: 서로 다른 메모리 공간에 있는 텐서들 간의 연산을 막고, 디바이스 연결 오류를 디버깅하기 쉽게 만들기 위함이다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeDisposedError(AMEVAForgeError):
    """
    이미 해제된 텐서에 접근 시 발생.
    
    무엇을: 메모리에서 이미 해제(dispose)된 텐서의 자원(데이터나 버퍼)에 다시 접근하려 할 때 발생하는 에러이다.
    왜: 댕글링 포인터 혹은 잘못된 메모리 참조로 인해 시스템이 크래시되는 것을 파이썬 레벨에서 안전하게 방지하기 위함이다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeWebGPUUnavailableError(AMEVAForgeError):
    """
    WebGPU를 사용할 수 없는 환경 (비-브라우저, 미지원 GPU 등).
    
    무엇을: 현재 실행 중인 환경이 WebGPU를 지원하지 않을 때 던져지는 에러 클래스이다.
    왜: Pyodide가 아닌 일반 파이썬 환경이거나 브라우저에서 WebGPU가 활성화되지 않았을 때 유의미한 에러 메시지를 제공하기 위함이다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeQuotaExceededError(AMEVAForgeError):
    """
    VRAM 쿼터 초과. NL-07: TypeScript AMEVAForgeQuotaExceededError와 대칭.
    
    무엇을: 브라우저나 GPU 장치에 할당된 메모리 한도(VRAM Quota)를 초과했을 때 발생하는 에러이다.
    왜: 무분별한 텐서 생성으로 인한 메모리 고갈(OOM) 상황을 명확하게 에러로 감지하기 위해 존재한다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeSecurityError(AMEVAForgeError):
    """
    보안 위반 (허용되지 않은 op, 셰이더 인젝션 시도 등). NL-07: TypeScript와 대칭.
    
    무엇을: 허용되지 않은 보안 정책 위반이나, 악의적인 셰이더 코드 주입 시도 등이 감지되었을 때 발생하는 에러이다.
    왜: 웹 환경(WebGPU)에서 안전하지 않은 동작을 즉각적으로 차단하여 보안 취약점을 예방하기 위함이다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeValidationError(AMEVAForgeError):
    """GPU validation error scope에서 감지된 오류."""
    pass


class AMEVAForgeOutOfMemoryError(AMEVAForgeError):
    """GPU out-of-memory error scope에서 감지된 오류."""
    pass


class AMEVAForgeInternalGPUError(AMEVAForgeError):
    """GPU internal error scope에서 감지된 오류."""
    pass


class AMEVAForgeDeviceLostError(AMEVAForgeError):
    """GPU device lost 오류."""
    pass


class AMEVAForgeStaleHandleError(AMEVAForgeError):
    """이전 generation의 stale handle 접근 오류."""
    pass


class AMEVAForgeUnsupportedOperationError(AMEVAForgeDeviceError):
    """Release 1에서 지원하지 않는 연산 오류."""
    pass

```

---

## `packages/forge-py/src/forge/functional.py`

```python
"""
functional.py - AMEVA-Forge 함수형 연산 모음

[역사적 메타데이터]
- Created: Wed Aug 12 12:14:52 2026 +0900 (초기 커밋)
- Modified:
  - Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
  - Wed Aug 12 12:59:35 2026 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
"""
from typing import Tuple
from .tensor import Tensor
from .autograd import Function, Context
from .ops import exp_op, log_op, sum_op, sub, mul, neg, div, mean_op, add, permute
import numpy as np

class SoftmaxFunction(Function):
    """
    무엇을: Softmax 연산과 그 역전파(gradient)를 정의하는 autograd Function 클래스이다.
    왜: 신경망의 출력층 등에서 값들을 확률 분포로 변환하기 위해 필요하며 자동 미분을 지원하기 위해 존재한다.
    어떻게: Function 베이스 클래스를 상속받고, forward와 backward 정적 메서드를 구현하여 연산을 정의한다.
    """
    @staticmethod
    def forward(ctx: Context, x: Tensor, axis: int = -1) -> Tensor:
        """
        무엇을: Softmax의 순전파 연산을 수행한다.
        왜: 입력 텐서 x를 지정된 축(axis)에 대해 확률값(0~1)으로 스케일링하기 위함이다.
        어떻게: 수치적 안정성을 위해 최대값을 빼고 exp를 취한 뒤, 그 합으로 나눈다. CPU와 GPU 디바이스에 따라 분기하여 처리한다.
        """
        # 무엇을: 축 정보를 컨텍스트에 저장한다.
        # 왜: backward 시 동일한 축(axis)을 기준으로 그래디언트를 계산해야 하기 때문이다.
        # 어떻게: ctx.axis 속성에 axis 값을 할당한다.
        ctx.axis = axis
        if x.device == 'cpu':
            from .ops import _require_cpu_data
            # 무엇을: CPU 데이터를 가져온다.
            # 왜: numpy 연산을 수행하기 위함이다.
            # 어떻게: _require_cpu_data를 호출한다.
            data = _require_cpu_data(x, 'x')
            
            # 무엇을: 해당 축에서 최대값을 찾는다.
            # 왜: exp() 계산 시 오버플로우(overflow)를 방지하기 위한 수치적 안정화 기법이다.
            # 어떻게: np.max를 사용한다.
            max_val = np.max(data, axis=axis, keepdims=True)
            
            # 무엇을: 지수 함수를 취한다.
            # 왜: 각 원소를 양수로 만들고 크기에 비례하게 증폭시키기 위함이다.
            # 어떻게: 데이터에서 최대값을 뺀 후 np.exp를 호출한다.
            exp_data = np.exp(data - max_val)
            
            # 무엇을: 지수 값들의 합을 구한다.
            # 왜: 확률 분포로 정규화하기 위한 분모를 얻기 위해서다.
            # 어떻게: np.sum을 사용한다.
            sum_exp = np.sum(exp_data, axis=axis, keepdims=True)
            
            # 무엇을: 정규화를 수행한다.
            # 왜: 합이 1이 되는 확률 값을 얻기 위함이다.
            # 어떻게: exp_data를 sum_exp로 나눈다.
            result = exp_data / sum_exp
            
            # 무엇을: 결과 텐서를 컨텍스트에 저장한다.
            # 왜: backward 시 야코비안 계산을 위해 Softmax 결과값이 필요하기 때문이다.
            # 어떻게: ctx.save_for_backward에 텐서를 래핑하여 넘긴다.
            ctx.save_for_backward(Tensor(shape=result.shape, dtype='float32', device='cpu', data=result))
            return Tensor(shape=result.shape, dtype='float32', device='cpu', data=result)
        else:
            from .ops import exp_op, div, sub, sum_axis, max_axis, reshape
            norm_axis = axis if axis >= 0 else axis + len(x.shape)
            
            # 수치 안정성 (Numerical Stability): x - max(x) <= 0
            m = max_axis(x, axis=norm_axis)
            s_shape = list(x.shape)
            s_shape[norm_axis] = 1
            m_reshaped = reshape(m, tuple(s_shape))
            shifted_x = sub(x, m_reshaped)

            # 무엇을: 수치 안정화된 텐서에 대해 exp를 취한다 (exp(x - max(x)) <= 1.0, overflow 불가).
            e = exp_op(shifted_x)
            
            # 무엇을: N차원 범용 축소(sum_axis)를 수행하고 브로드캐스팅 형태로 변환한다.
            s = sum_axis(e, axis=norm_axis)
            s_reshaped = reshape(s, tuple(s_shape))
                
            # 무엇을: 분자(e)를 분모(s_reshaped)로 나눈다.
            res = div(e, s_reshaped)
            ctx.save_for_backward(res)
            return res

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        """
        무엇을: Softmax의 역전파 연산을 수행한다.
        왜: 출력에 대한 그래디언트(grad_output)를 받아 입력(x)에 대한 그래디언트를 체인 룰을 통해 계산하기 위함이다.
        어떻게: 수식 `grad_in = res * (grad_out - sum(grad_out * res))`을 CPU와 GPU 각각의 방식으로 구현한다.
        """
        # 무엇을: 순전파에서 저장한 결과를 꺼낸다.
        # 왜: 역전파 수식 계산에 사용하기 위해서다.
        # 어떻게: 언패킹을 통해 할당한다.
        res, = ctx.saved_tensors
        if res.device == 'cpu':
            from .ops import _require_cpu_data
            res_data = _require_cpu_data(res, 'res')
            grad_data = _require_cpu_data(grad_output, 'grad_output')
            axis = ctx.axis
            
            # 무엇을: 그래디언트와 결과의 내적(sum)을 구한다.
            # 왜: 야코비안 행렬과 벡터의 곱셈을 스칼라 형태로 최적화하여 풀기 위함이다.
            # 어떻게: 원소별 곱 후 축(axis)에 대해 sum을 취한다.
            sum_val = np.sum(grad_data * res_data, axis=axis, keepdims=True)
            
            # 무엇을: 최종 입력 그래디언트를 구한다.
            # 왜: 이전 계층으로 오차를 전파하기 위해서다.
            # 어떻게: res * (grad - sum_val) 식을 numpy로 계산한다.
            grad_in = res_data * (grad_data - sum_val)
            return (Tensor(shape=res.shape, dtype='float32', device='cpu', data=grad_in),)
        else:
            from .ops import mul, sub, sum_axis, reshape
            norm_axis = ctx.axis if ctx.axis >= 0 else ctx.axis + len(res.shape)
            
            # 무엇을: 출력 그래디언트와 순전파 결과를 원소별로 곱한다.
            # 왜: 그래디언트 합(sum)을 구하기 위한 중간 단계이다.
            # 어떻게: mul 연산을 사용한다.
            m = mul(grad_output, res)
            
            s = sum_axis(m, axis=norm_axis)
            s_shape = list(res.shape)
            s_shape[norm_axis] = 1
            sum_val = reshape(s, tuple(s_shape))
                
            # 무엇을: grad_output에서 sum_val을 뺀다.
            # 왜: Softmax 야코비안 수식의 괄호 안 부분을 계산하기 위함이다.
            # 어떻게: sub 연산을 사용한다.
            diff = sub(grad_output, sum_val)
            return (mul(res, diff),)

def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    
    무엇을: Softmax 연산을 수행하는 래퍼(wrapper) 함수이다.
    왜: 사용자가 Function.apply를 직접 호출하지 않고 직관적으로 함수를 사용할 수 있도록 하기 위함이다.
    어떻게: SoftmaxFunction.apply를 호출하여 텐서를 넘긴다.
    """
    return SoftmaxFunction.apply(x, axis=axis)

class LogSoftmaxFunction(Function):
    """
    무엇을: Log-Softmax 연산과 그 역전파를 정의하는 클래스이다.
    왜: Softmax 후 Log를 취하는 것보다 수치적으로 훨씬 안정적이고 빠르기 때문이다.
    어떻게: Function을 상속받아 forward/backward를 구현한다.
    """
    @staticmethod
    def forward(ctx: Context, x: Tensor, axis: int = -1) -> Tensor:
        """
        무엇을: LogSoftmax의 순전파 연산을 수행한다.
        왜: 입력 텐서를 log 확률 형태로 안정적으로 변환하기 위함이다.
        어떻게: x - log(sum(exp(x - max))) 공식을 적용한다.
        """
        ctx.axis = axis
        if x.device == 'cpu':
            from .ops import _require_cpu_data
            data = _require_cpu_data(x, 'x')
            max_val = np.max(data, axis=axis, keepdims=True)
            shifted = data - max_val
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
            result = shifted - log_sum_exp
            
            # 무엇을: backward를 위해 softmax 확률을 저장한다.
            # 왜: log-softmax의 미분 시 exp(log_softmax) 즉 softmax 결과값이 필요하기 때문이다.
            # 어떻게: np.exp(result)를 취한 후 텐서화하여 저장한다.
            ctx.save_for_backward(Tensor(shape=result.shape, dtype='float32', device='cpu', data=np.exp(result)))
            return Tensor(shape=result.shape, dtype='float32', device='cpu', data=result)
        else:
            from .ops import exp_op, div, sum_axis, max_axis, reshape, log_op, sub
            norm_axis = axis if axis >= 0 else axis + len(x.shape)
            
            # 수치 안정성 (Numerical Stability): x - max(x) - log(sum(exp(x - max(x))))
            m = max_axis(x, axis=norm_axis)
            s_shape = list(x.shape)
            s_shape[norm_axis] = 1
            m_reshaped = reshape(m, tuple(s_shape))
            shifted_x = sub(x, m_reshaped)
            
            e = exp_op(shifted_x)
            s = sum_axis(e, axis=norm_axis)
            s_reshaped = reshape(s, tuple(s_shape))
            log_sum = log_op(s_reshaped)
            
            result = sub(shifted_x, log_sum)
            # backward를 위해 softmax 확률 저장
            res = div(e, s_reshaped)
            ctx.save_for_backward(res)
            return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        """
        무엇을: LogSoftmax의 역전파 연산을 수행한다.
        왜: 출력 오차를 입력으로 전파하기 위함이다.
        어떻게: 수식 `grad_in = grad_out - softmax(x) * sum(grad_out)`을 적용한다.
        """
        softmax_x, = ctx.saved_tensors
        if grad_output.device == 'cpu':
            grad_data = grad_output.numpy()
            softmax_data = softmax_x.numpy()
            axis = ctx.axis
            sum_grad = np.sum(grad_data, axis=axis, keepdims=True)
            grad_in = grad_data - softmax_data * sum_grad
            return (Tensor(shape=grad_output.shape, dtype='float32', device='cpu', data=grad_in),)
        else:
            from .ops import mul, sub, transpose, sum_axis, reshape
            axis = ctx.axis
            
            if axis == -1 or axis == 1:
                t = transpose(grad_output)
                s = sum_axis(t, axis=0)
                sum_grad = reshape(s, (grad_output.shape[0], 1))
            elif axis == 0:
                s = sum_axis(grad_output, axis=0)
                sum_grad = reshape(s, (1, grad_output.shape[1]))
            else:
                s = sum_axis(grad_output, axis=0)
                sum_grad = reshape(s, (1, grad_output.shape[1]))
                
            return (sub(grad_output, mul(softmax_x, sum_grad)),)

def log_softmax(x, axis=-1):
    """
    Numerically stable log-softmax.
    
    무엇을: Log-Softmax를 호출하는 사용자 편의 함수이다.
    왜: 직접 LogSoftmaxFunction.apply를 타이핑하는 수고를 덜어주기 위함이다.
    어떻게: 함수 내부에서 apply를 위임 호출한다.
    """
    return LogSoftmaxFunction.apply(x, axis=axis)

class CrossEntropyFunction(Function):
    """
    무엇을: 교차 엔트로피 손실(Cross Entropy Loss) 연산 및 역전파 클래스이다.
    왜: 분류 문제에서 모델의 예측 분포와 정답 레이블 간의 차이를 측정하여 손실을 구하기 위해 존재한다.
    어떻게: 예측값에 Log-Softmax를 취한 뒤, 정답 레이블에 해당하는 인덱스의 값을 추출해 평균을 낸다.
    """
    @staticmethod
    def forward(ctx: Context, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        무엇을: Cross Entropy 손실값을 계산한다.
        왜: 모델 학습의 최적화 목표가 되는 단일 스칼라 손실(loss)을 얻기 위함이다.
        어떻게: 입력(predictions)을 수치 안정적 방식으로 처리하고, 정답 타겟 위치의 음의 로그 확률 평균을 구한다.
        """
        ctx.save_for_backward(predictions, targets)
        
        if predictions.device == 'cpu':
            from .ops import _require_cpu_data
            pred_data = _require_cpu_data(predictions, 'pred')
            target_data = _require_cpu_data(targets, 'targets').astype(np.int64)
            
            max_val = np.max(pred_data, axis=-1, keepdims=True)
            shifted = pred_data - max_val
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
            log_probs = shifted - log_sum_exp
            
            n = pred_data.shape[0]
            # 무엇을: 배치 내의 정답 레이블에 해당하는 확률만 추출해 평균 음수 값을 취한다.
            # 왜: NLLLoss (Negative Log Likelihood Loss) 연산을 수행하기 위함이다.
            # 어떻게: numpy의 인덱싱 기법(fancy indexing)을 활용한다.
            loss = -np.mean(log_probs[np.arange(n), target_data])
            
            probs = np.exp(log_probs)
            ctx.probs = Tensor(shape=probs.shape, dtype='float32', device='cpu', data=probs)
            ctx.target_data = target_data
            return Tensor(shape=(), dtype='float32', device='cpu', data=np.array(loss, dtype=np.float32))
        else:
            from .ops import _require_cpu_data, tensor, mul, sum_op, div, neg
            if targets.device == 'cpu':
                target_data = _require_cpu_data(targets, 'targets').astype(np.int64)
            elif hasattr(targets, '_data') and targets._data is not None:
                target_data = targets._data.astype(np.int64)
            else:
                from .errors import AMEVAForgeDeviceError
                raise AMEVAForgeDeviceError(
                    "CrossEntropyLoss expects target class indices on CPU (e.g., forge.tensor(y, device='cpu')) "
                    "in Release 1 for host-side one-hot indexing."
                )
            n, c = predictions.shape
            if n * c > 4_000_000:
                from .errors import AMEVAForgeUnsupportedOperationError
                raise AMEVAForgeUnsupportedOperationError(
                    f"GPU CrossEntropy currently uses dense one-hot targets in Release 1. "
                    f"Requested one-hot size is {n}x{c} ({n * c * 4 / (1024 * 1024):.1f} MB). "
                    f"Use smaller class count or wait for sparse_cross_entropy GPU kernel in Release 2."
                )
            
            # 1. GPU Log-Softmax 계산
            log_probs = log_softmax(predictions, axis=-1)
            
            # 2. Host에서 One-Hot 행렬 생성 후 GPU 텐서로 변환
            one_hot = np.zeros((n, c), dtype=np.float32)
            one_hot[np.arange(n), target_data] = 1.0
            one_hot_t = tensor(one_hot, device=predictions.device, requires_grad=False)
            
            # 3. NLL Loss = -sum(one_hot * log_probs) / n
            nll = neg(sum_op(mul(one_hot_t, log_probs)))
            loss = div(nll, tensor(float(n), device=predictions.device, requires_grad=False))
            
            # 4. Softmax 확률 및 메타데이터 저장 (역전파용)
            probs = softmax(predictions, axis=-1)
            ctx.probs = probs
            ctx.one_hot_t = one_hot_t
            ctx.batch_size = float(n)
            return loss

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """
        무엇을: Cross Entropy 손실 함수의 역전파를 수행한다.
        왜: 소프트맥스와 NLLLoss의 결합 도함수인 (probs - one_hot) / N 을 통해 입력 로짓의 그래디언트를 구하기 위함이다.
        어떻게: GPU/CPU 디바이스별로 최적화된 연산 체인을 적용하여 그래디언트를 산출한다.
        """
        predictions, targets = ctx.saved_tensors
        if predictions.device == 'cpu':
            probs_data = ctx.probs.numpy()
            target_data = ctx.target_data
            n = probs_data.shape[0]
            grad_pred = probs_data.copy()
            grad_pred[np.arange(n), target_data] -= 1.0
            grad_pred = grad_pred / n
            
            if grad_output.shape != ():
                grad_pred = grad_pred * grad_output.numpy()
            else:
                grad_pred = grad_pred * float(grad_output.numpy())
            return (Tensor(shape=grad_pred.shape, dtype='float32', device='cpu', data=grad_pred), None)
        else:
            from .ops import sub, div, mul, tensor
            probs = ctx.probs
            one_hot_t = ctx.one_hot_t
            n = ctx.batch_size
            
            # grad_pred = (probs - one_hot) / n
            grad_unscaled = sub(probs, one_hot_t)
            grad_pred = div(grad_unscaled, tensor(n, device=predictions.device, requires_grad=False))
            
            if grad_output.shape != ():
                grad_pred = mul(grad_pred, grad_output)
            elif float(getattr(grad_output, '_data', 1.0) if hasattr(grad_output, '_data') and grad_output._data is not None else 1.0) != 1.0:
                grad_pred = mul(grad_pred, grad_output)
                
            return (grad_pred, None)

def cross_entropy(predictions, targets):
    """
    Cross-entropy loss. predictions: (N, C), targets: (N,) integer class indices.
    
    무엇을: 크로스 엔트로피 함수 래퍼.
    왜: 사용자 편의성을 위해 제공.
    어떻게: CrossEntropyFunction.apply 호출.
    """
    return CrossEntropyFunction.apply(predictions, targets)

def mse_loss(predictions, targets):
    """
    Mean Squared Error loss.
    
    무엇을: 평균 제곱 오차(MSE)를 계산한다.
    왜: 회귀(Regression) 문제 등에서 두 텐서 간의 값 차이를 손실로 산출하기 위함이다.
    어떻게: 예측값에서 정답값을 뺀 후(sub), 그 결과를 제곱하고(mul), 전체 평균(mean_op)을 구한다.
    """
    diff = sub(predictions, targets)
    sq = mul(diff, diff)
    return mean_op(sq)

def _move_tensor_state(dst, src) -> None:
    """
    WHAT: src 텐서의 상태와 지연 연산 그래프를 dst 텐서로 안전하게 이동(Move)합니다.
    WHY: BatchNorm의 running_mean/running_var 같은 in-place 통계량 갱신 시,
         src의 식별자/그래프/데이터 소유권을 dst로 이전하여 dst 객체의 참조 동일성을 유지하기 위함입니다.
    HOW: dst 필드 덮어쓰기 -> src 필드 None 초기화.
    """
    dst._data = src._data
    dst._handle = src._handle
    if hasattr(dst, "_handle_cell"):
        dst._handle_cell.handle = src._handle

    dst._lazy_op = getattr(src, "_lazy_op", None)
    dst._op = getattr(src, "_op", None)
    dst._parents = getattr(src, "_parents", ())
    dst._op_params = getattr(src, "_op_params", None)

    dst.shape = src.shape
    dst.dtype = src.dtype
    dst.device = src.device
    dst.requires_grad = False
    dst.grad = None
    dst._version += 1

    if dst.device == "gpu" and not getattr(dst, "_finalizer_registered", False):
        import weakref
        from .tensor import Tensor
        weakref.finalize(dst, Tensor._finalize_buffer, dst._handle_cell)
        dst._finalizer_registered = True

    src._handle = None
    if hasattr(src, "_handle_cell"):
        src._handle_cell.handle = None
    src._data = None

def batch_norm2d(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5):
    """
    무엇을: 2D 배치 정규화(Batch Normalization)를 수행한다.
    왜: 신경망 각 층의 입력을 정규화하여 학습(Internal Covariate Shift 방지)을 안정적이고 빠르게 만들기 위함이다.
    어떻게: 채널(Channel) 차원을 기준으로 배치, 높이, 너비에 대한 평균과 분산을 구하고, 이를 이용해 데이터를 정규화한 뒤 학습 가능한 weight와 bias를 적용한다.
    """
    from .ops import sub, mul, div, add, reshape, mean_axis, tensor, sqrt, full
    import numpy as np
    
    if training:
        # 무엇을: 배치 차원(0)과 공간 차원(2, 3)을 순차적으로 평균 내어 채널별 평균을 구한다.
        # 왜: 채널 단위의 분포 통계량을 얻기 위함이다.
        # 어떻게: mean_axis를 연쇄 호출한다.
        m_c = mean_axis(mean_axis(mean_axis(x, 0), 1), 1)
        m_view = reshape(m_c, (1, x.shape[1], 1, 1))
        
        diff = sub(x, m_view)
        diff_sq = mul(diff, diff)
        v_c = mean_axis(mean_axis(mean_axis(diff_sq, 0), 1), 1)
        v_view = reshape(v_c, (1, x.shape[1], 1, 1))
        
        n = x.shape[0] * x.shape[2] * x.shape[3]
        if x.device == 'cpu':
            unbiased_v = v_c._data * (n / (n - 1)) if n > 1 else v_c._data
            # 무엇을: 이동 평균(running stats)을 업데이트한다.
            # 왜: 추론(Inference) 시 현재 배치가 아닌 전체 데이터셋의 통계량을 사용하기 위해 모멘텀을 적용해 누적하기 위함이다.
            # 어떻게: 지수 이동 평균(EMA) 수식을 적용한다.
            running_mean._data = (1 - momentum) * running_mean._data + momentum * m_c._data
            running_var._data = (1 - momentum) * running_var._data + momentum * unbiased_v
            running_mean._version += 1
            running_var._version += 1
        else:
            old_rm = Tensor(
                shape=running_mean.shape,
                dtype=running_mean.dtype,
                device=running_mean.device,
                handle=running_mean._handle,
                data=running_mean._data,
                op=running_mean._lazy_op,
                parents=running_mean._parents,
                op_params=running_mean._lazy_params
            )
            old_rv = Tensor(
                shape=running_var.shape,
                dtype=running_var.dtype,
                device=running_var.device,
                handle=running_var._handle,
                data=running_var._data,
                op=running_var._lazy_op,
                parents=running_var._parents,
                op_params=running_var._lazy_params
            )
            new_rm = add(mul(old_rm, full(running_mean.shape, 1 - momentum, device=x.device)), mul(m_c, full(m_c.shape, momentum, device=x.device)))
            unbiased_v = mul(v_c, full(v_c.shape, n / (n - 1) if n > 1 else 1.0, device=x.device))
            new_rv = add(mul(old_rv, full(running_var.shape, 1 - momentum, device=x.device)), mul(unbiased_v, full(unbiased_v.shape, momentum, device=x.device)))
            _move_tensor_state(running_mean, new_rm)
            _move_tensor_state(running_var, new_rv)
            
        mean_use, var_use = m_view, v_view
    else:
        mean_use = reshape(running_mean, (1, x.shape[1], 1, 1))
        var_use = reshape(running_var, (1, x.shape[1], 1, 1))
        
    eps_t = full(var_use.shape, eps, device=x.device)
    denom = sqrt(add(var_use, eps_t))
    x_norm = div(sub(x, mean_use), denom)
    
    w_view = reshape(weight, (1, x.shape[1], 1, 1))
    b_view = reshape(bias, (1, x.shape[1], 1, 1))
    
    out = add(mul(x_norm, w_view), b_view)
    return out

def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """
    무엇을: 레이어 정규화(Layer Normalization)를 수행한다.
    왜: 트랜스포머(Transformer) 등에서 시퀀스나 토큰 단위로 데이터의 스케일을 맞춰주기 위함이다.
    어떻게: 가장 마지막 차원(dim=-1)을 기준으로 평균과 분산을 구하여 정규화한 뒤 아핀(affine) 변환을 수행한다.
    """
    from .ops import sub, mul, div, add, mean_axis, full, sqrt, unsqueeze
    dim = -1
    
    m = mean_axis(x, dim)
    m_view = unsqueeze(m, dim)
    
    diff = sub(x, m_view)
    diff_sq = mul(diff, diff)
    
    v = mean_axis(diff_sq, dim)
    v_view = unsqueeze(v, dim)
    
    eps_t = full(v_view.shape, eps, device=x.device)
    denom = sqrt(add(v_view, eps_t))
    x_norm = div(diff, denom)
    
    out = x_norm
    if weight is not None:
        out = mul(out, weight)
    if bias is not None:
        out = add(out, bias)
        
    return out

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, training=False):
    """
    무엇을: 스케일드 닷 프로덕트 어텐션(Scaled Dot-Product Attention)을 계산한다.
    왜: 트랜스포머 구조에서 토큰 간의 연관성(Attention weight)을 구하고 정보를 집계하기 위함이다.
    어떻게: Q와 K의 전치를 내적하고 스케일링한 후, Softmax를 통과시켜 V와 가중합을 계산한다.
    """
    from .ops import bmm, transpose, div, full, reshape, dropout
    import math
    
    orig_shape = query.shape
    if len(orig_shape) == 4:
        B, H, L, D = orig_shape
        # 무엇을: 배치와 헤드 차원을 하나로 합친다.
        # 왜: bmm(Batch Matrix Multiplication)을 3차원 텐서에 대해 쉽게 적용하기 위함이다.
        # 어떻게: reshape 연산을 통해 (B*H, L, D) 형태로 변환한다.
        query = reshape(query, (B * H, L, D))
        key = reshape(key, (B * H, key.shape[2], D))
        value = reshape(value, (B * H, value.shape[2], value.shape[3]))
        
    d_k = query.shape[-1]
    query_t = query
    key_t = permute(key, (0, 2, 1)) if len(key.shape) == 3 else permute(key, (0, 1, 3, 2))
    
    # 무엇을: Q와 K^T의 내적을 통해 어텐션 스코어를 구한다.
    # 왜: 토큰 간의 유사도를 측정하기 위함이다.
    # 어떻게: 차원에 따라 bmm 또는 matmul을 사용한다.
    scores = bmm(query_t, key_t) if len(query_t.shape) == 3 else matmul(query_t, key_t)
    
    # 무엇을: 스코어를 sqrt(d_k)로 나눈다.
    # 왜: 차원이 클수록 내적값이 커져 Softmax 기울기가 소실되는 것을 방지하기 위한 스케일링 작업이다.
    # 어떻게: full 텐서를 만들고 div 연산을 적용한다.
    scores = div(scores, full(scores.shape, math.sqrt(d_k), device=query.device))
    
    if is_causal:
        from .ops import tensor
        L_q, L_k = scores.shape[-2], scores.shape[-1]
        mask_np = np.triu(np.full((L_q, L_k), -1e4, dtype=np.float32), k=1)
        causal_mask = tensor(mask_np, device=scores.device)
        scores = add(scores, causal_mask)
    elif attn_mask is not None:
        scores = add(scores, attn_mask)

    attn = softmax(scores, axis=-1)
    
    if dropout_p > 0.0:
        attn = dropout(attn, dropout_p, training)
        
    out = bmm(attn, value)
    
    if len(orig_shape) == 4:
        # 무엇을: 출력 형태를 원래의 4차원으로 복구한다.
        # 왜: 다중 헤드 어텐션(Multi-Head Attention)의 다음 단계 처리를 위해 형태를 맞춰주기 위함이다.
        # 어떻게: 저장해둔 orig_shape로 reshape한다.
        out = reshape(out, orig_shape)
        
    return out

```

---

## `packages/forge-py/src/forge/generated/__init__.py`

```python
"""Generated contracts package."""

```

---

## `packages/forge-py/src/forge/generated/op_schema.py`

```python
"""
AUTO-GENERATED FILE. DO NOT MODIFY MANUALLY.
Generated from packages/forge/schema/release1-ops.json
Run `py -3 scripts/generate_release1_contracts.py` to regenerate.
"""

from typing import Dict, Any, List

RELEASE1_OP_SCHEMA: Dict[str, Dict[str, Any]] = {'add': {'inputs': 2, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'sub': {'inputs': 2, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'mul': {'inputs': 2, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'div': {'inputs': 2, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'neg': {'inputs': 1, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'matmul': {'inputs': 2, 'params': [{'name': 'M', 'type': 'positive-int'}, {'name': 'N', 'type': 'positive-int'}, {'name': 'K', 'type': 'positive-int'}], 'output': 'matmul-2d', 'dtypes': ['float32']}, 'transpose': {'inputs': 1, 'params': [{'name': 'M', 'type': 'positive-int'}, {'name': 'N', 'type': 'positive-int'}], 'output': 'transpose-2d', 'dtypes': ['float32']}, 'reshape': {'inputs': 1, 'params': [{'name': 'targetShape', 'type': 'shape-tuple'}], 'output': 'reshape', 'dtypes': ['float32']}, 'sum': {'inputs': 1, 'params': [], 'output': 'scalar', 'dtypes': ['float32']}, 'relu': {'inputs': 1, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'relu_backward': {'inputs': 2, 'params': [{'name': 'numElements', 'type': 'positive-int'}], 'output': 'same-shape', 'dtypes': ['float32']}, 'mse_loss': {'inputs': 2, 'params': [], 'output': 'scalar', 'dtypes': ['float32']}, 'mse_loss_backward': {'inputs': 2, 'params': [{'name': 'numElements', 'type': 'positive-int'}], 'output': 'same-shape', 'dtypes': ['float32']}, 'axpy': {'inputs': 2, 'params': [{'name': 'numElements', 'type': 'positive-int'}, {'name': 'alpha', 'type': 'float32'}], 'output': 'alias-input-0', 'dtypes': ['float32']}}

RELEASE1_OPS: List[str] = list(RELEASE1_OP_SCHEMA.keys())

```

---

## `packages/forge-py/src/forge/graph.py`

```python
"""
graph.py - WebGPU 연산 그래프 빌더

[역사적 메타데이터]
- Created: Wed Aug 12 12:14:52 2026 +0900 (초기 커밋)
- Modified:
  - Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
"""
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class GraphBuilder:
    """
    무엇을: 연산 노드들을 순서대로 기록하여 JSON 형태의 연산 그래프로 빌드해주는 클래스이다.
    왜: WebGPU 브릿지(JS)로 전송할 때 복잡한 텐서 연산을 하나의 배치(배치 연산)로 최적화하여 넘기기 위함이다.
    어떻게: 각 노드를 딕셔너리 형태로 nodes 리스트에 추가하고, 고유한 ID를 부여하여 그래프로 연결한다.
    """
    def __init__(self):
        """
        무엇을: GraphBuilder 인스턴스를 초기화한다.
        왜: 그래프를 담을 리스트와 노드 ID 발급 상태를 리셋하기 위함이다.
        어떻게: 빈 노드 리스트(nodes), 입력 리스트(inputs), 그리고 시작 ID를 1로 설정한다.
        """
        # 무엇을: 그래프를 구성하는 연산 노드들의 정보를 담은 딕셔너리 리스트.
        # 왜: 나중에 JSON으로 직렬화하기 위함이다.
        # 어떻게: add_op 등의 메서드가 호출될 때마다 새로운 노드가 추가된다.
        self.nodes: List[Dict[str, Any]] = []
        
        # 무엇을: 연산 그래프에서 사용하는 초기 입력 데이터 리스트.
        # 왜: 업로드 시 전달할 실제 데이터를 저장하기 위함이다.
        # 어떻게: add_upload를 통해 외부 데이터가 주입될 때마다 순서대로 추가된다.
        self.inputs: List[Any] = []
        
        # 무엇을: 다음 노드에 부여할 고유 식별자 번호.
        # 왜: 그래프 내 각 노드 간의 의존성을 ID 기반으로 연결하기 위함이다.
        # 어떻게: alloc_id 호출 시마다 1씩 증가한다.
        self.next_id = 1

    def alloc_id(self) -> int:
        """
        무엇을: 새로운 노드용 고유 ID를 할당하여 반환한다.
        왜: 연산 그래프 내 노드 간 입출력 관계(dependency)를 추적하기 위한 식별자가 필요하기 때문이다.
        어떻게: 현재 next_id 값을 리턴하고, next_id 자체는 1 증가시킨다.
        """
        idx = self.next_id
        self.next_id += 1
        return idx

    def add_upload(self, shape: Tuple[int, ...], data: Any = None) -> int:
        """
        무엇을: 외부 데이터를 GPU 메모리로 업로드하는 'upload' 노드를 그래프에 추가한다.
        왜: CPU 데이터를 초기 텐서 노드로 변환하기 위함이다.
        어떻게: 새 ID를 받고, shape의 각 차원을 int로 변환해 JSON 직렬화 가능하게 만들며, 데이터가 있으면 inputs 리스트에 추가한다.
        """
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"Shape must be a tuple or list of ints, got {type(shape)}")
        clean_shape = []
        for d in shape:
            if not isinstance(d, (int, np.integer)) or int(d) < 0:
                raise ValueError(f"Shape dimensions must be non-negative integers, got {d}")
            clean_shape.append(int(d))
            
        if len(clean_shape) > 8:
            raise ValueError(f"Shape dimensions cannot exceed 8, got {len(clean_shape)}")

        node_id = self.alloc_id()
        self.nodes.append({"op": "upload", "id": node_id, "shape": clean_shape})
        if data is not None:
            self.inputs.append(data)
        return node_id

    def add_load(self, shape: Tuple[int, ...], handle: str) -> int:
        """
        무엇을: 이미 GPU에 존재하는 데이터를 로드하는 'load' 노드를 추가한다.
        왜: 기존 캐싱된 메모리나 다른 파이프라인에서 생성된 리소스를 그래프 안으로 끌어오기 위함이다.
        어떻게: 새 ID를 발급받고 노드 정보(핸들 포함)를 기록한 뒤 해당 ID를 리턴한다.
        """
        if not isinstance(handle, str) or not handle.strip():
            raise ValueError(f"Handle must be a non-empty string, got {handle}")
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"Shape must be a tuple or list of ints, got {type(shape)}")
        clean_shape = [int(d) for d in shape]
        
        node_id = self.alloc_id()
        self.nodes.append({
            "op": "load",
            "id": node_id,
            "shape": clean_shape,
            "handle": handle
        })
        return node_id

    def add_op(
        self,
        op_name: str,
        shape: Tuple[int, ...],
        in_ids: List[int],
        params: Optional[List[Any]] = None
    ) -> int:
        """
        무엇을: 일반적인 텐서 연산(op) 노드를 그래프에 추가한다.
        왜: 사용자가 호출한 수학적 연산(add, mul 등)을 그래프에 기록하여 연산 순서를 정하기 위함이다.
        어떻게: 입력 노드 ID 리스트(in_ids)를 의존성으로 기록하고, 추가 파라미터가 있다면 JSON 포맷에 맞게 형변환하여 딕셔너리에 추가한다.
        """
        if not isinstance(op_name, str) or not op_name.strip():
            raise ValueError(f"op_name must be a valid non-empty string, got {op_name}")
        if op_name in ("__proto__", "constructor", "prototype",
                       "__import__", "eval", "exec", "compile",
                       "__subclasses__", "__globals__", "__builtins__",
                       "system", "popen", "subprocess"):
            raise ValueError(f"Prohibited op_name: {op_name}")
            
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"Shape must be a tuple or list of ints, got {type(shape)}")
        clean_shape = []
        for d in shape:
            if not isinstance(d, (int, np.integer)) or int(d) < 0:
                raise ValueError(f"Shape dimensions must be non-negative integers, got {d}")
            clean_shape.append(int(d))
            
        if len(clean_shape) > 8:
            raise ValueError(f"Shape dimensions cannot exceed 8, got {len(clean_shape)}")

        clean_in_ids = []
        for i_id in in_ids:
            if not isinstance(i_id, (int, np.integer)) or int(i_id) <= 0:
                raise ValueError(f"in_ids must contain positive integers, got {i_id}")
            clean_in_ids.append(int(i_id))

        node_id = self.alloc_id()
        node: Dict[str, Any] = {
            "op": op_name,
            "id": node_id,
            "shape": clean_shape,
            "in": clean_in_ids
        }
        if params is not None:
            clean_params = []
            for p in params:
                if isinstance(p, bool):
                    clean_params.append(p)
                elif isinstance(p, (float, np.floating)):
                    clean_params.append(float(p))
                elif isinstance(p, (int, np.integer)):
                    clean_params.append(int(p))
                else:
                    clean_params.append(p)
            node["params"] = clean_params
        self.nodes.append(node)
        return node_id

    def compile(self) -> Tuple[str, List[Any]]:
        """
        무엇을: 현재까지 모인 노드 정보를 JSON 문자열과 입력 데이터 리스트로 묶어 반환한다.
        왜: 생성된 그래프를 JS 브릿지(WebGPU 백엔드) 측에 통신 가능한 형태로 전달하기 위함이다.
        어떻게: 내장 json 모듈을 이용해 nodes 리스트를 직렬화(dumps)하고, 입력 데이터(inputs)와 함께 튜플로 리턴한다.
        """
        return json.dumps(self.nodes), self.inputs

```

---

## `packages/forge-py/src/forge/nn.py`

```python
"""
================================================================================
[AMEVA-Forge 역사 기록 (Historical Metadata)]
생성일 (Created): Wed Aug 12 12:14:52 2026 +0900
수정 내역 (Modified):
- Wed Aug 12 12:59:35 2026 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
- Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Forge and reorganize directories
================================================================================
"""

# WHAT: typing 모듈에서 List 타입을 임포트합니다.
# WHY: 타입 힌팅을 통해 코드의 가독성을 높이고 정적 분석을 용이하게 하기 위함입니다.
# HOW: 반환 값 등의 타입 명시에 사용됩니다.
from typing import List
from collections import OrderedDict
from .tensor import Tensor
from .errors import AMEVAForgeUnsupportedOperationError

# WHAT: 내부 연산 모듈에서 다양한 수학적 연산 함수들을 임포트합니다.
# WHY: 신경망 계층 내에서 순전파 연산을 수행하기 위해 필요한 연산들을 제공하기 때문입니다.
# HOW: 포워드 패스에서 데이터 변환 및 활성화 함수로 호출됩니다.
from .ops import random, zeros, transpose, relu, matmul

# WHAT: 외부 라이브러리인 numpy를 임포트합니다.
# WHY: 수치 해석 및 다차원 배열 연산을 최적화하여 빠르고 효율적으로 처리하기 위함입니다.
# HOW: 초기 가중치 생성, 데이터 타입 변환 등의 기반 연산에 사용됩니다.
import numpy as np

# WHAT: 모든 신경망 모듈의 기본이 되는 베이스 클래스입니다.
# WHY: 계층(layer)들의 파라미터 관리, 상태 저장/불러오기, 훈련/평가 모드 전환 등 공통적인 기능을 제공하기 위해 존재합니다.
# HOW: 사용자가 정의하는 모든 계층이나 모델은 이 클래스를 상속받아 `forward` 메서드를 구현하여 동작합니다.
class Module:
    # WHAT: Module 인스턴스의 초기화 메서드입니다.
    # WHY: 객체가 생성될 때 필요한 내부 상태(서브모듈, 파라미터, 훈련 모드)를 설정하기 위함입니다.
    # HOW: object.__setattr__를 통해 속성 초기화 시 발생할 수 있는 무한 루프나 속성 충돌을 방지합니다.
    def __init__(self):
        # VUL-011 Fix: object.__setattr__로 안전하게 초기화
        # __setattr__ 오버라이드 전에 내부 딕셔너리를 먼저 생성
        
        # WHAT: 현재 모듈에 등록된 하위 모듈들을 저장하는 딕셔너리입니다.
        # WHY: 계층적 구조를 가지는 신경망에서 자식 모듈들을 관리하여 파라미터 추출 및 상태 관리를 용이하게 하기 위함입니다.
        # HOW: 속성 설정 시 값이 Module 인스턴스일 경우 이 딕셔너리에 추가됩니다.
        object.__setattr__(self, '_modules', {})
        
        # WHAT: 현재 모듈에 직접 속한 학습 가능한 파라미터들을 저장하는 딕셔너리입니다.
        # WHY: 그래디언트 업데이트 시 대상이 되는 파라미터들(가중치, 편향 등)을 추적하기 위함입니다.
        # HOW: 속성 설정 시 값이 requires_grad=True인 Tensor일 경우 이 딕셔너리에 추가됩니다.
        object.__setattr__(self, '_params', {})
        
        # WHAT: 모듈의 현재 동작 상태(훈련 중인지 여부)를 나타내는 불리언 변수입니다.
        # WHY: Dropout이나 BatchNorm처럼 훈련 시와 평가 시 동작이 다른 계층들을 제어하기 위함입니다.
        # HOW: True일 경우 훈련 모드, False일 경우 평가(추론) 모드로 동작하게 됩니다.
        object.__setattr__(self, 'training', True)
        
    # WHAT: 모듈과 모든 하위 모듈의 훈련 모드를 설정하는 메서드입니다.
    # WHY: 전체 네트워크의 상태를 일괄적으로 변경하여 훈련 또는 평가에 적합한 동작을 하도록 만들기 위함입니다.
    # HOW: 자기 자신의 상태를 변경한 후, 루프를 돌며 자식 모듈들에 재귀적으로 호출합니다.
    def train(self, mode=True):
        # WHAT: 현재 모듈의 훈련 상태를 인자로 받은 mode로 설정합니다.
        # WHY: 훈련/평가 모드 플래그를 업데이트하기 위함입니다.
        # HOW: self.training 속성에 mode 값을 대입합니다.
        self.training = mode
        
        # WHAT: 자식 모듈들을 순회하며 상태를 전파하는 루프입니다.
        # WHY: 중첩된 신경망 구조에서 하위 모듈들까지 동일한 훈련 상태를 가지도록 하기 위함입니다.
        # HOW: self._modules 딕셔너리의 값들을 하나씩 꺼내어 train(mode)를 호출합니다.
        for m in self._modules.values():
            m.train(mode)
            
    # WHAT: 모듈을 평가(추론) 모드로 전환하는 메서드입니다.
    # WHY: 사용자가 직관적으로 모델을 평가 상태로 바꿀 수 있게 편의성을 제공하기 위함입니다.
    # HOW: 내부적으로 self.train(False)를 호출하여 훈련 상태를 해제합니다.
    def eval(self):
        self.train(False)
    
    # WHAT: 모듈의 순전파 연산을 정의하는 메서드입니다.
    # WHY: 입력 데이터가 이 모듈을 통과할 때 어떤 연산이 일어나는지 명세하기 위함입니다.
    # HOW: 하위 클래스에서 오버라이드하여 구체적인 계산 로직을 구현해야 하며, 기본적으로는 NotImplementedError를 발생시킵니다.
    def forward(self, *args):
        raise NotImplementedError
    
    # WHAT: 모듈 객체를 함수처럼 호출할 수 있게 해주는 매직 메서드입니다.
    # WHY: model(x) 와 같은 직관적인 문법으로 순전파를 실행할 수 있게 지원하기 위함입니다.
    # HOW: 전달받은 인자들을 그대로 forward 메서드로 넘겨주어 반환값을 돌려줍니다.
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    # WHAT: 모델 내부의 모든 학습 가능한 파라미터들을 리스트로 반환하는 메서드입니다.
    # WHY: 옵티마이저(Optimizer)가 어떤 가중치들을 업데이트해야 하는지 알아야 하기 때문입니다.
    # HOW: 자신의 파라미터들을 리스트로 만들고, 모든 자식 모듈을 순회하며 그들의 파라미터도 수집하여 합칩니다.
    def parameters(self) -> List[Tensor]:
        # WHAT: 현재 모듈에 직접 포함된 파라미터들을 리스트 형태로 가져옵니다.
        # WHY: 파라미터 수집의 시작점 역할을 하기 위함입니다.
        # HOW: self._params 딕셔너리의 값들을 list로 변환합니다.
        params = list(self._params.values())
        
        # WHAT: 자식 모듈들을 순회하는 루프입니다.
        # WHY: 하위 모듈들에 숨겨진 파라미터들도 모두 찾아내기 위함입니다.
        # HOW: 각 자식 모듈 m의 parameters() 메서드를 재귀적으로 호출하여 반환된 리스트를 params에 연장(extend)합니다.
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def to(self, device: str):
        """
        WHAT: 모델의 모든 파라미터(Parameter), 버퍼(Buffer: running_mean 등), 서브모듈을 지정된 디바이스로 일괄 이동합니다.
        WHY: GPU 가속 또는 CPU 평가를 위해 모델 내 모든 텐서 자원의 연산 장치를 일치시키기 위함입니다.
        HOW: _params와 인스턴스 내 모든 Tensor 객체들을 to(device)로 변환하고 _modules를 재귀 호출합니다.
        """
        if device not in ("cpu", "gpu"):
            from .errors import AMEVAForgeDeviceError
            raise AMEVAForgeDeviceError(
                f"Unsupported device: {device!r}. "
                "Expected 'cpu' or 'gpu'."
            )
        for name, parameter in self._params.items():
            parameter.move_to_(device)
        for name, value in self.__dict__.items():
            if name.startswith('_'):
                continue
            if isinstance(value, Tensor) and name not in self._params:
                value.move_to_(device)
        for module in self._modules.values():
            module.to(device)
        return self

    # WHAT: 모델의 전체 파라미터 상태를 딕셔너리 형태로 추출하는 메서드입니다.
    # WHY: 모델의 가중치를 파일로 저장(Serialization)하거나 다른 모델로 복사할 때 사용하기 위함입니다.
    # HOW: OrderedDict를 생성한 후 자신의 파라미터를 추가하고 자식 모듈을 순회하며 상태를 누적합니다.
    def state_dict(self, prefix='', keep_vars=False):
        # WHAT: 순서가 보장되는 딕셔너리를 임포트하고 생성합니다.
        # WHY: 파라미터의 구조와 순서를 일정하게 유지하기 위함입니다.
        # HOW: collections 모듈에서 OrderedDict를 불러와 인스턴스를 생성합니다.
        from collections import OrderedDict
        state = OrderedDict()
        
        # WHAT: 현재 모듈의 파라미터들을 순회하는 루프입니다.
        # WHY: 상태 사전에 각 파라미터의 이름과 데이터를 저장하기 위함입니다.
        # HOW: self._params.items()를 통해 이름과 파라미터 객체를 가져옵니다.
        for name, param in self._params.items():
            # WHAT: 계층적 구조를 반영한 전체 파라미터 식별 키입니다.
            # WHY: 글로벌 상태 사전 내에서 이름 충돌을 방지하기 위함입니다.
            # HOW: 전달받은 prefix와 현재 파라미터 name을 결합합니다.
            key = prefix + name
            
            if keep_vars:
                # WHAT: Tensor 객체 자체를 보존하여 상태 사전에 저장합니다.
                # WHY: 그래디언트 정보 등 텐서의 고유 메타데이터가 필요할 때 사용하기 위함입니다.
                # HOW: key에 param 객체를 그대로 할당합니다.
                state[key] = param
            else:
                # WHAT: 파라미터의 실제 수치 데이터(NumPy 배열 등)만 추출하여 저장합니다.
                # WHY: 모델 저장 시 불필요한 메타데이터를 제외하고 순수 가중치만 저장하기 위함입니다.
                # HOW: CPU 텐서는 _data 또는 numpy()로 추출하고, GPU 텐서는 _data가 없을 경우 명시적 안내 에러를 발생시킵니다.
                if param.device == 'cpu':
                    state[key] = param._data if param._data is not None else param.numpy()
                elif hasattr(param, '_data') and param._data is not None:
                    state[key] = param._data
                else:
                    from .errors import AMEVAForgeDeviceError
                    raise AMEVAForgeDeviceError(
                        f"state_dict(keep_vars=False) cannot synchronously readback GPU parameter '{key}'. "
                        "Use model.state_dict(keep_vars=True) to retain GPU tensor handles, "
                        "or transfer model to CPU first: model.to('cpu').state_dict()."
                    )
                
        # WHAT: 하위 모듈들을 순회하는 루프입니다.
        # WHY: 트리 구조로 얽힌 모든 파라미터의 상태를 빠짐없이 수집하기 위함입니다.
        # HOW: self._modules를 반복하며 각 모듈에 대해 재귀적으로 state_dict를 호출하고 결과를 업데이트합니다.
        for name, module in self._modules.items():
            if module is not None:
                state.update(module.state_dict(prefix + name + '.', keep_vars))
                
        return state

    # WHAT: 저장된 상태 딕셔너리로부터 모델 파라미터를 복원하는 메서드입니다.
    # WHY: 저장소나 파일에서 불러온 가중치를 현재 모델 객체에 덮어씌워 사용할 수 있게 하기 위함입니다.
    # HOW: 현재 모델의 state_dict를 텐서 형태로 가져온 뒤, 입력된 상태 딕셔너리와 키를 매칭시켜 데이터를 교체합니다.
    def load_state_dict(self, state_dict):
        # WHAT: 현재 모듈의 전체 파라미터 텐서 객체들을 포함하는 딕셔너리를 가져옵니다.
        # WHY: 파라미터 객체의 값을 안전하게 제자리에서(in-place) 덮어씌우기 위해 참조를 확보하기 위함입니다.
        # HOW: keep_vars=True 인자를 주어 데이터뿐 아니라 객체 자체를 리턴받습니다.
        my_state = self.state_dict(keep_vars=True)
        
        # WHAT: 현재 모델 파라미터들을 하나씩 검사하는 루프입니다.
        # WHY: 입력받은 state_dict에 매칭되는 데이터가 있는지 확인하고 복원하기 위함입니다.
        # HOW: my_state 딕셔너리에서 이름과 파라미터 참조를 가져와서 비교합니다.
        for name, param in my_state.items():
            if name in state_dict:
                # WHAT: state_dict에서 복원할 데이터 값을 꺼내옵니다.
                # WHY: 실제 덮어씌울 데이터를 확보하기 위함입니다.
                # HOW: name을 키로 사용하여 값을 조회합니다.
                val = state_dict[name]
                
                if hasattr(val, 'numpy'):
                    # WHAT: 불러온 값이 텐서류 객체일 경우 numpy 배열로 변환합니다.
                    # WHY: 내부 파라미터 데이터는 numpy 배열 기반으로 관리되기 때문입니다.
                    # HOW: numpy() 메서드를 호출하여 다차원 배열을 추출합니다.
                    val = val.numpy()
                    
                # WHAT: 현재 파라미터 객체의 내부 데이터를 새 값으로 대체합니다.
                # WHY: 모델의 가중치를 업데이트하여 복원을 마무리하기 위함입니다.
                # HOW: val을 numpy 배열로 감싸고 기존 데이터 타입과 일치시킨 후 param._data에 덮어씁니다.
                param._data = np.array(val, dtype=param._data.dtype if param._data is not None else np.float32)
    
    # WHAT: 객체의 속성을 설정할 때 호출되는 매직 메서드 오버라이드입니다.
    # WHY: 새로운 속성이 Module인지 Tensor(파라미터)인지 자동으로 감지하여 내부 딕셔너리에 등록하기 위함입니다.
    # HOW: 속성 이름과 값을 분석하여 적절한 내부 컬렉션(_modules 또는 _params)에 추가한 뒤 원본 객체의 __setattr__를 호출합니다.
    def __setattr__(self, name, value):
        if name.startswith('_'):
            # WHAT: 프라이빗(private) 속성에 대한 설정 로직입니다.
            # WHY: 내부 상태 변수들이 _modules나 _params로 오분류되는 것을 방지하기 위함입니다.
            # HOW: 별도 처리 없이 기본 object.__setattr__를 이용해 직접 속성을 설정하고 종료합니다.
            object.__setattr__(self, name, value)
            return
            
        if isinstance(value, Module):
            # WHAT: 할당되는 값이 서브모듈(Module 인스턴스)일 경우의 처리입니다.
            # WHY: 네트워크 구조에 속하는 계층을 모듈 트리에 등록하기 위함입니다.
            # HOW: _modules 딕셔너리에 name을 키로 하여 저장합니다.
            self._modules[name] = value
            
        if isinstance(value, Tensor) and getattr(value, 'requires_grad', False):
            # WHAT: 할당되는 값이 학습을 필요로 하는 텐서(파라미터)일 경우의 처리입니다.
            # WHY: 그래디언트를 계산해야 하는 가중치 및 편향을 파라미터 리스트에 자동으로 등록하기 위함입니다.
            # HOW: _params 딕셔너리에 name을 키로 하여 저장합니다.
            self._params[name] = value
            
        # WHAT: 모든 확인 과정을 거친 후 객체 인스턴스에 실제 속성을 설정합니다.
        # WHY: 클래스 인스턴스가 런타임에 올바른 상태를 유지하도록 하기 위함입니다.
        # HOW: 내장 object.__setattr__ 메서드를 호출합니다.
        object.__setattr__(self, name, value)


# WHAT: 완전 연결 계층(Fully Connected Layer)을 구현한 클래스입니다.
# WHY: 입력 피처와 가중치 간의 선형 변환(Linear Transformation)을 수행하여 특징을 추출하거나 분류를 수행하기 위함입니다.
# HOW: 가중치 행렬 및 편향 벡터를 파라미터로 가지고 입력 데이터와의 행렬 곱을 수행합니다.
class Linear(Module):
    # WHAT: Linear 계층 인스턴스의 초기화 메서드입니다.
    # WHY: 입력 및 출력 차원수를 바탕으로 가중치와 편향 파라미터를 초기화하고 등록하기 위함입니다.
    # HOW: He(Kaiming) 초기화 기법을 사용하여 분산을 보정하고 Tensor 객체로 파라미터를 생성합니다.
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Kaiming initialization
        # WHAT: 가중치 초기화의 스케일을 설정하는 변수입니다.
        # WHY: 깊은 신경망에서 기울기 소실이나 폭발을 방지하기 위해 분산을 2/in_features로 조정하기 위함입니다.
        # HOW: (2.0 / in_features)의 제곱근을 계산하여 적용합니다.
        scale = (2.0 / in_features) ** 0.5
        
        # WHAT: 정규 분포 기반 무작위 값으로 초기화된 가중치 데이터를 생성하는 변수입니다.
        # WHY: 가중치가 동일한 값으로 시작되는 대칭성을 파괴하고, 학습이 정상적으로 이루어지도록 하기 위함입니다.
        # HOW: numpy.random.randn을 통해 표준정규분포에서 추출한 후 scale을 곱합니다.
        w_data = np.random.randn(out_features, in_features).astype(np.float32) * scale
        
        # WHAT: 모델의 가중치 파라미터 텐서입니다.
        # WHY: 선형 변환 과정에서 입력 데이터와 곱해질 행렬 공간을 유지하기 위함입니다.
        # HOW: requires_grad=True를 주어 그래디언트 계산을 활성화하고 Tensor 인스턴스를 self.weight로 저장합니다.
        self.weight = Tensor(shape=(out_features, in_features), dtype='float32', device='cpu', data=w_data, requires_grad=True)
        
        if bias:
            # WHAT: 모델의 편향(Bias) 파라미터 텐서입니다.
            # WHY: 데이터를 원점으로부터 평행 이동시켜 모델의 표현력을 높이기 위함입니다.
            # HOW: 0으로 초기화된 out_features 크기의 Tensor를 생성하고 requires_grad=True로 설정합니다.
            self.bias = Tensor(shape=(out_features,), dtype='float32', device='cpu',
                             data=np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None
    
    # WHAT: Linear 계층의 순전파 연산을 정의하는 메서드입니다.
    # WHY: 입력 데이터를 받아 선형 결합 수식(Wx + b)을 실제로 계산하기 위함입니다.
    # HOW: 입력 데이터 x와 가중치의 전치행렬 간 행렬 곱(matmul)을 구한 후, 존재한다면 편향을 더합니다.
    def forward(self, x):
        # WHAT: 입력과 가중치의 행렬 곱셈 결과입니다.
        # WHY: 공간 변환 및 특징 추출을 진행하기 위함입니다.
        # HOW: matmul(x, transpose(self.weight)) 연산을 수행합니다.
        out = matmul(x, transpose(self.weight))
        if self.bias is not None:
            # WHAT: 덧셈 연산을 제공하는 함수 임포트입니다.
            # WHY: 편향을 선형 결합 결과에 더해주기 위함입니다.
            # HOW: 브로드캐스팅이 지원되는 내부 ops.add를 호출합니다.
            from .ops import add
            out = add(out, self.bias)  # broadcasting: (batch, out) + (out,)
        return out


# WHAT: ReLU (Rectified Linear Unit) 활성화 함수 계층 클래스입니다.
# WHY: 모델에 비선형성을 부여하여 복잡한 패턴을 학습할 수 있게 하고, 그래디언트 소실 문제를 줄이기 위함입니다.
# HOW: 순전파 시 입력 데이터의 모든 음수를 0으로 변환하고 양수는 그대로 통과시킵니다.
class ReLU(Module):
    # WHAT: ReLU의 순전파 연산입니다.
    # WHY: 입력 텐서 요소별로 비선형 변환을 적용하기 위함입니다.
    # HOW: 내부의 relu 연산 함수에 텐서를 전달하여 결과를 반환합니다.
    def forward(self, x):
        return relu(x)


# WHAT: 시그모이드(Sigmoid) 활성화 함수 계층 클래스입니다.
# WHY: 출력값을 0과 1 사이로 압축하여 확률 등과 같은 스케일로 변환하거나 게이트 제어에 사용하기 위함입니다.
# HOW: 1 / (1 + exp(-x)) 수식을 각 요소에 적용합니다.
class Sigmoid(Module):
    # WHAT: Sigmoid의 순전파 연산입니다.
    # WHY: 텐서의 각 요소에 시그모이드 함수를 통과시키기 위함입니다.
    # HOW: 내부 ops.sigmoid 함수를 임포트하여 적용 결과를 반환합니다.
    def forward(self, x):
        from .ops import sigmoid
        return sigmoid(x)


# WHAT: Tanh (Hyperbolic Tangent) 활성화 함수 계층 클래스입니다.
# WHY: 출력값을 -1과 1 사이로 압축하고, 데이터의 중심을 0으로 맞추어 학습 효율을 개선하기 위함입니다.
# HOW: 하이퍼볼릭 탄젠트 수식을 요소별로 연산합니다.
class Tanh(Module):
    # WHAT: Tanh의 순전파 연산입니다.
    # WHY: 텐서 각 요소를 쌍곡탄젠트 공간으로 맵핑하기 위함입니다.
    # HOW: 내부 ops.tanh_op 함수를 임포트하여 계산된 값을 리턴합니다.
    def forward(self, x):
        from .ops import tanh_op
        return tanh_op(x)


# WHAT: 여러 신경망 계층들을 순차적으로 이어붙여 단일 모듈로 만들어주는 컨테이너 클래스입니다.
# WHY: 복잡한 네트워크 구조를 리스트 형태로 쉽게 정의하고 한 번의 forward 호출로 연속 처리를 가능하게 하기 위함입니다.
# HOW: 초기화 시 인자로 받은 계층들을 내부 딕셔너리에 순서대로 저장하고 순전파 시 차례대로 통과시킵니다.
class Sequential(Module):
    # WHAT: Sequential 인스턴스를 초기화하는 메서드입니다.
    # WHY: 사용자가 제공한 다수의 계층 인스턴스들을 모듈 트리에 등록하기 위함입니다.
    # HOW: 위치 인자(layers)들을 받아 순회하며 문자열로 된 인덱스를 키로 _modules에 저장합니다.
    def __init__(self, *layers):
        super().__init__()
        # WHAT: 전달된 계층들을 순회하며 등록하는 루프입니다.
        # WHY: 순서를 보장하면서 각 레이어 모듈을 자식 모듈로 관리하기 위함입니다.
        # HOW: enumerate를 사용해 인덱스를 얻고, 문자열로 변환하여 키로 사용합니다.
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer
    
    # WHAT: Sequential 모듈의 순전파 연산입니다.
    # WHY: 등록된 계층들을 순서대로 통과시켜 최종 결과를 얻기 위함입니다.
    # HOW: _modules에 저장된 하위 모듈들을 차례로 호출하며 이전 출력값을 다음 입력값으로 갱신합니다.
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)


class MSELoss(Module):
    """
    Mean Squared Error loss module.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        from .functional import mse_loss
        return mse_loss(input, target)


# WHAT: 2차원 공간 상의 최대 풀링(Max Pooling) 연산을 수행하는 계층입니다.
# WHY: 공간적 해상도를 줄이면서 중요한 특징(가장 강한 신호)을 보존하여 위치 불변성을 얻고 계산량을 감소시키기 위함입니다.
# HOW: 정해진 커널 크기와 보폭(stride)으로 텐서를 순회하며 최댓값만을 추출합니다.
class MaxPool2d(Module):
    # WHAT: MaxPool2d 클래스의 초기화 메서드입니다.
    # WHY: 풀링 연산의 파라미터(커널 크기, 보폭, 패딩)를 설정하고 저장하기 위함입니다.
    # HOW: 인자로 받은 값을 인스턴스의 속성으로 할당합니다.
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        # WHAT: 풀링 윈도우의 크기입니다.
        # WHY: 추출할 영역의 크기를 결정하기 위함입니다.
        # HOW: 단일 정수 또는 튜플로 저장됩니다.
        self.kernel_size = kernel_size
        
        # WHAT: 윈도우가 이동하는 간격(보폭)입니다.
        # WHY: 출력 특성 맵의 크기와 다운샘플링 비율을 결정하기 위함입니다.
        # HOW: 주어지지 않으면 커널 크기와 동일하게 사용되도록 보존됩니다.
        self.stride = stride
        
        # WHAT: 입력 텐서의 경계에 덧붙일 패딩 크기입니다.
        # WHY: 모서리 부분의 정보 손실을 막거나 출력 크기를 정교하게 맞추기 위함입니다.
        # HOW: 저장해두었다가 순전파 시 연산 함수에 전달됩니다.
        self.padding = padding
        
    # WHAT: MaxPool2d의 순전파 메서드입니다.
    # WHY: 입력 텐서에 2D 맥스 풀링을 적용하기 위함입니다.
    # HOW: ops 모듈의 max_pool2d 함수를 호출하여 계산된 결과를 반환합니다.
    def forward(self, x):
        from .ops import max_pool2d
        return max_pool2d(x, self.kernel_size, self.stride, self.padding)

# WHAT: 2차원 공간 상의 평균 풀링(Average Pooling) 연산을 수행하는 계층입니다.
# WHY: 윈도우 내의 평균값을 취해 특징 맵을 부드럽게 줄이고 전체적인 정보를 유지하기 위함입니다.
# HOW: 커널 크기와 보폭을 지정하고 해당 영역 값들의 평균을 계산합니다.
class AvgPool2d(Module):
    # WHAT: AvgPool2d의 초기화 메서드입니다.
    # WHY: 평균 풀링에 필요한 하이퍼파라미터를 세팅하기 위함입니다.
    # HOW: 커널 크기, 보폭, 패딩을 객체 속성으로 저장합니다.
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    # WHAT: AvgPool2d의 순전파 메서드입니다.
    # WHY: 입력 피처맵 데이터에 대해 평균 풀링 연산을 수행하기 위함입니다.
    # HOW: 내부의 avg_pool2d 함수에 텐서와 인자들을 전달해 결과를 얻습니다.
    def forward(self, x):
        from .ops import avg_pool2d
        return avg_pool2d(x, self.kernel_size, self.stride, self.padding)

# WHAT: 다차원 텐서를 연속된 1차원 데이터로 펼치는(Flatten) 계층 클래스입니다.
# WHY: 합성곱(CNN) 층을 거친 다차원 피처맵을 완전 연결(Linear) 계층의 입력으로 주입할 수 있도록 형태를 변환하기 위함입니다.
# HOW: start_dim부터 end_dim까지의 차원을 결합하여 새로운 형태의 텐서를 만듭니다.
class Flatten(Module):
    # WHAT: Flatten 계층을 초기화하는 메서드입니다.
    # WHY: 텐서에서 어느 차원 구간을 평탄화할지 범위를 설정하기 위함입니다.
    # HOW: 기본적으로 배치 차원(0)은 유지하고 1번째 차원부터 마지막 차원까지를 속성으로 저장합니다.
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        
    # WHAT: Flatten 모듈의 순전파 메서드입니다.
    # WHY: 입력 텐서의 형태를 실제로 변환하기 위함입니다.
    # HOW: ops.flatten 함수를 임포트하고 저장된 차원 인자와 함께 호출합니다.
    def forward(self, x):
        from .ops import flatten
        return flatten(x, self.start_dim, self.end_dim)


# WHAT: 2차원 배치 정규화(Batch Normalization 2D) 계층 클래스입니다.
# WHY: 신경망 학습 시 내부 공변량 변화(Internal Covariate Shift)를 줄여 학습 속도와 안정성을 극대화하기 위함입니다.
# HOW: 배치(Batch) 단위로 채널별 평균과 분산을 구해 정규화하고, 학습 가능한 스케일(weight)과 시프트(bias) 파라미터를 적용합니다.
class BatchNorm2d(Module):
    # WHAT: BatchNorm2d 초기화 메서드입니다.
    # WHY: 정규화 시 채널 개수에 맞는 학습 파라미터(감마, 베타)와 이동 평균 데이터를 준비하기 위함입니다.
    # HOW: 가중치(1)와 편향(0)은 학습 파라미터로, 이동 평균과 분산은 비학습 텐서로 초기화하여 속성에 할당합니다.
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        # WHAT: 입력 텐서의 채널 수입니다.
        # WHY: 각 파라미터들의 크기 형상을 결정하기 위함입니다.
        # HOW: 클래스 내부에 저장합니다.
        self.num_features = num_features
        
        # WHAT: 분모에 더해지는 매우 작은 상숫값입니다.
        # WHY: 분산이 0에 가까울 때 발생할 수 있는 0으로 나누기 오류나 수치적 불안정을 방지하기 위함입니다.
        # HOW: 엡실론 값을 멤버 변수로 저장하여 식에 사용합니다.
        self.eps = eps
        
        # WHAT: 이동 평균과 분산을 업데이트할 때 사용되는 모멘텀 수치입니다.
        # WHY: 과거 통계량과 현재 배치의 통계량을 어느 비율로 섞을지 정하여 학습을 안정화하기 위함입니다.
        # HOW: 지수 이동 평균 공식에 모멘텀 가중치로 활용됩니다.
        self.momentum = momentum
        
        # WHAT: 정규화된 값에 곱해지는 스케일(감마) 파라미터 텐서입니다.
        # WHY: 정규화 후에도 네트워크가 기존 데이터의 표현력을 회복할 수 있게 학습시키기 위함입니다.
        # HOW: 채널 크기만큼 1로 초기화되며 그래디언트 계산을 활성화(requires_grad=True)합니다.
        self.weight = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.ones(num_features, dtype=np.float32), requires_grad=True)
        
        # WHAT: 정규화된 값에 더해지는 이동(베타) 파라미터 텐서입니다.
        # WHY: 원점을 유연하게 조정하여 모델의 비선형적 성능을 유지하기 위함입니다.
        # HOW: 채널 크기만큼 0으로 초기화되며 학습을 활성화합니다.
        self.bias = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.zeros(num_features, dtype=np.float32), requires_grad=True)
        
        # WHAT: 전체 훈련 데이터의 채널별 평균을 추적하는 이동 평균 텐서입니다.
        # WHY: 평가(eval) 시 현재 배치가 아닌 전체 데이터 분포를 기반으로 정규화하기 위함입니다.
        # HOW: 0으로 초기화하고 백프로퍼게이션에 참여하지 않도록 requires_grad=False로 설정합니다.
        self.running_mean = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.zeros(num_features, dtype=np.float32), requires_grad=False)
        
        # WHAT: 전체 훈련 데이터의 채널별 분산을 추적하는 이동 분산 텐서입니다.
        # WHY: 평가 시 변동성이 큰 단일 배치의 분산 대신 누적된 전역 분산을 사용하기 위함입니다.
        # HOW: 1로 초기화하고 학습(requires_grad=False) 대상에서 제외합니다.
        self.running_var = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.ones(num_features, dtype=np.float32), requires_grad=False)

    # WHAT: BatchNorm2d의 순전파 메서드입니다.
    # WHY: 입력 데이터를 정규화하고 아핀(affine) 변환을 수행하기 위함입니다.
    # HOW: functional 모듈의 batch_norm2d를 호출하며, 훈련/평가 모드 플래그(self.training)를 함께 전달합니다.
    def forward(self, x):
        from .functional import batch_norm2d
        return batch_norm2d(x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)


# WHAT: 드롭아웃(Dropout) 정규화 기법을 적용하는 계층 클래스입니다.
# WHY: 훈련 중 랜덤하게 특정 뉴런의 출력을 0으로 만들어, 네트워크가 특정 특징에 과적합(Overfitting)되는 것을 방지하기 위함입니다.
# HOW: 정해진 확률 p에 따라 요소들을 마스킹하고, 나머지 값들을 1/(1-p)로 스케일링하여 기댓값을 유지합니다.
class Dropout(Module):
    # WHAT: 드롭아웃 인스턴스를 초기화하는 메서드입니다.
    # WHY: 요소를 0으로 만들 확률값 p를 설정하기 위함입니다.
    # HOW: 파라미터 p를 인스턴스 속성으로 저장합니다.
    def __init__(self, p=0.5):
        super().__init__()
        # WHAT: 뉴런 출력이 무작위로 0이 될 확률입니다.
        # WHY: 모델의 정규화 강도를 조절하기 위함입니다.
        # HOW: self.p 변수에 저장하여 forward 시 사용합니다.
        self.p = p

    # WHAT: 드롭아웃 계층의 순전파 연산입니다.
    # WHY: 훈련 시 무작위 마스킹을 수행하고 추론(eval) 시에는 데이터 손실 없이 그대로 통과시키기 위함입니다.
    # HOW: ops.dropout 함수에 입력 텐서와 함께 현재 모델 상태(self.training)를 넘겨줍니다.
    def forward(self, x):
        from .ops import dropout
        return dropout(x, self.p, self.training)

# WHAT: 2차원 합성곱(Convolution 2D) 계층 클래스입니다.
# WHY: 이미지와 같은 2D 데이터 공간에서 국소적인 특징(가장자리, 질감 등)을 추출하기 위함입니다.
# HOW: 학습 가능한 커널(필터)을 입력 데이터 위로 슬라이딩하면서 합성곱 연산을 수행합니다.
class Conv2d(Module):
    # WHAT: Conv2d 계층의 인스턴스를 초기화하는 메서드입니다.
    # WHY: 입력/출력 채널, 커널 크기 등의 구조적 파라미터를 설정하고 가중치 텐서를 할당하기 위함입니다.
    # HOW: 채널과 커널 크기를 기반으로 He/Kaiming 초기화 범위(k)를 계산하여 가중치와 편향을 균등 분포로 생성합니다.
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        # WHAT: 입력 텐서의 채널 수입니다.
        # WHY: 커널의 깊이 차원을 맞추기 위함입니다.
        # HOW: 속성으로 저장합니다.
        self.in_channels = in_channels
        
        # WHAT: 생성될 출력 피처맵의 채널 수(필터의 개수)입니다.
        # WHY: 추출할 특징의 다양성을 결정하기 위함입니다.
        # HOW: 속성으로 저장합니다.
        self.out_channels = out_channels
        
        # WHAT: 합성곱 커널(필터)의 가로세로 크기입니다.
        # WHY: 수용 영역(Receptive Field)을 결정하기 위함입니다.
        # HOW: 속성으로 저장합니다.
        self.kernel_size = kernel_size
        
        # WHAT: 필터가 이동하는 보폭입니다.
        # WHY: 출력 특성맵의 공간적 크기를 조절하기 위함입니다.
        # HOW: 속성으로 저장합니다.
        self.stride = stride
        
        # WHAT: 입력 주변에 채울 0의 크기입니다.
        # WHY: 합성곱 후 공간 차원이 줄어드는 것을 보정하기 위함입니다.
        # HOW: 속성으로 저장합니다.
        self.padding = padding
        
        import math
        # WHAT: 가중치 초기화 상한/하한값입니다.
        # WHY: 입력 노드 수에 반비례하도록 초기화 범위를 조정해 분산을 유지하기 위함입니다.
        # HOW: 1 / sqrt(in_channels * kernel_size * kernel_size) 수식을 사용합니다.
        k = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        
        from .ops import random, tensor
        import numpy as np
        
        # WHAT: 합성곱 필터의 가중치 데이터입니다.
        # WHY: 입력으로부터 패턴을 학습하고 추출하는 파라미터로 사용하기 위함입니다.
        # HOW: [-k, k] 사이의 균등 분포에서 난수를 추출해 (out_channels, in_channels, kernel_size, kernel_size) 모양으로 텐서를 생성합니다.
        weight_data = np.random.uniform(-k, k, (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        self.weight = tensor(weight_data, requires_grad=True)
        
        if bias:
            # WHAT: 채널별 편향(Bias) 파라미터입니다.
            # WHY: 결과값을 이동(shift)시켜 활성화 함수의 비선형성 임계값을 조절하기 위함입니다.
            # HOW: 균등 분포 난수로 1차원 텐서를 생성하고 학습 가능하게 만듭니다.
            bias_data = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = tensor(bias_data, requires_grad=True)
        else:
            self.bias = None

    # WHAT: Conv2d의 순전파 연산 메서드입니다.
    # WHY: 입력 이미지(텐서)에 대해 실제 필터를 적용하여 특징 맵을 반환하기 위함입니다.
    # HOW: 입력과 가중치의 장치(device)가 다르면 동기화한 뒤, ops 모듈의 conv2d 함수를 호출합니다.
    def forward(self, x: 'Tensor') -> 'Tensor':
        from .ops import conv2d
        from .errors import AMEVAForgeDeviceError
        
        if self.weight.device != x.device:
            raise AMEVAForgeDeviceError(
                f"Conv2d weight device '{self.weight.device}' does not match input device '{x.device}'. "
                f"Call model.to('{x.device}') before executing forward or initializing the optimizer."
            )
            
        if self.bias is not None and self.bias.device != x.device:
            raise AMEVAForgeDeviceError(
                f"Conv2d bias device '{self.bias.device}' does not match input device '{x.device}'. "
                f"Call model.to('{x.device}') before executing forward or initializing the optimizer."
            )
            
        # WHAT: 최종 2D 합성곱 연산을 실행합니다.
        # WHY: 입력과 학습된 필터들 간의 크로스 코릴레이션(cross-correlation) 결과를 구하기 위함입니다.
        # HOW: 내부 C/C++ 기반 또는 최적화된 conv2d 함수로 넘깁니다.
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

# WHAT: 레이어 정규화(Layer Normalization) 계층 클래스입니다.
# WHY: 시퀀스 데이터나 자연어 처리에서 미니배치 차원이 아닌 피처(레이어) 차원에 대해 정규화를 수행해 학습을 돕기 위함입니다.
# HOW: 각 샘플별로 주어진 차원들(normalized_shape)에 걸쳐 평균과 분산을 구하고 표준화합니다.
class LayerNorm(Module):
    # WHAT: LayerNorm 초기화 메서드입니다.
    # WHY: 정규화할 형태와 엡실론, 학습 가능한 변환 스케일(Affine) 파라미터를 세팅하기 위함입니다.
    # HOW: 정규화 형태를 튜플로 저장하고, 필요시 가중치와 편향 텐서를 1과 0으로 각각 생성합니다.
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            # WHAT: 정규화 형태를 튜플로 강제 변환합니다.
            # WHY: 단일 정수 입력도 내부 연산에서 일관된 튜플 형태로 다루기 위함입니다.
            # HOW: 요소를 하나 가진 튜플로 감쌉니다.
            normalized_shape = (normalized_shape,)
            
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            from .ops import tensor
            import numpy as np
            # WHAT: 정규화 후 분포의 크기를 복원하기 위한 스케일링 파라미터입니다.
            # WHY: 데이터의 중요한 분산 정보가 무분별하게 사라지는 것을 방지하기 위함입니다.
            # HOW: normalized_shape 크기만큼 1.0 값을 가지는 텐서로 초기화합니다.
            self.weight = tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
            
            # WHAT: 정규화 후 위치를 복원하기 위한 시프트 파라미터입니다.
            # WHY: 데이터의 평균 정보 손실을 보완하기 위함입니다.
            # HOW: 0.0 값을 가지는 텐서로 만듭니다.
            self.bias = tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    # WHAT: LayerNorm의 순전파 연산입니다.
    # WHY: 텐서 내 각 샘플 레이어별로 정규화 연산을 수행하기 위함입니다.
    # HOW: 파라미터가 장치에 맞게 준비되었는지 확인 후 내부 functional.layer_norm 함수를 부릅니다.
    def forward(self, x):
        from .functional import layer_norm
        from .errors import AMEVAForgeDeviceError
        
        if self.weight is not None and self.weight.device != x.device:
            raise AMEVAForgeDeviceError(
                f"LayerNorm weight device '{self.weight.device}' does not match input device '{x.device}'. "
                f"Call model.to('{x.device}') before executing forward or initializing the optimizer."
            )
            
        if self.bias is not None and self.bias.device != x.device:
            raise AMEVAForgeDeviceError(
                f"LayerNorm bias device '{self.bias.device}' does not match input device '{x.device}'. "
                f"Call model.to('{x.device}') before executing forward or initializing the optimizer."
            )
            
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

# WHAT: 멀티헤드 어텐션(Multihead Attention) 계층 클래스입니다.
# WHY: 트랜스포머(Transformer) 모델에서 여러 관점(헤드)으로 동시에 시퀀스 내 요소들 간의 상관관계(어텐션)를 파악하기 위함입니다.
# HOW: Query, Key, Value를 각각 선형 변환한 후 여러 개의 헤드로 나누고 어텐션을 병렬 계산한 뒤 다시 결합합니다.
class MultiheadAttention(Module):
    # WHAT: MultiheadAttention의 초기화 메서드입니다.
    # WHY: 임베딩 차원, 헤드 개수를 정하고 프로젝션을 위한 선형 레이어(Linear)를 구성하기 위함입니다.
    # HOW: 각 프로젝션(q, k, v, out)용 Linear 인스턴스를 생성해 속성으로 등록합니다.
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # WHAT: 단일 어텐션 헤드가 처리할 차원의 크기입니다.
        # WHY: 전체 차원을 헤드 수로 균등하게 분할하여 병렬 연산하기 위함입니다.
        # HOW: 전체 임베딩 차원을 헤드 개수로 나눈 몫을 저장합니다.
        self.head_dim = embed_dim // num_heads
        
        # WHAT: Query 텐서를 프로젝션하기 위한 선형 계층입니다.
        # WHY: 입력 데이터를 어텐션 메커니즘을 위한 질의(Query) 공간으로 매핑하기 위함입니다.
        # HOW: embed_dim 크기의 입출력을 갖는 Linear 모듈로 초기화됩니다.
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        # WHAT: Key 텐서를 프로젝션하기 위한 선형 계층입니다.
        # WHY: 어텐션에서 질의와 비교될 대상(Key) 공간으로 매핑하기 위함입니다.
        # HOW: Linear(embed_dim, embed_dim)으로 구성됩니다.
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        # WHAT: Value 텐서를 프로젝션하기 위한 선형 계층입니다.
        # WHY: 어텐션 가중치가 곱해져 실제 정보(Value)로 쓰일 공간으로 매핑하기 위함입니다.
        # HOW: Linear(embed_dim, embed_dim)으로 구성됩니다.
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        # WHAT: 여러 헤드에서 합쳐진 결과를 최종 차원으로 복원하는 선형 계층입니다.
        # WHY: 병렬 처리된 다중 관점의 정보를 하나로 융합(mix)하기 위함입니다.
        # HOW: 출력 차원인 embed_dim으로 다시 한번 선형 결합합니다.
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        
    # WHAT: 멀티헤드 어텐션의 순전파 메서드입니다.
    # WHY: 입력된 q, k, v에 대해 실제 스케일드 닷 프로덕트 어텐션 연산을 수행하기 위함입니다.
    # HOW: 입력을 리니어 변환하고 차원을 헤드 단위로 쪼갠 뒤 어텐션을 적용하고 합쳐서 최종 리니어 변환합니다.
    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        from .functional import scaled_dot_product_attention
        from .ops import reshape, permute
        
        # WHAT: 입력 텐서들의 형상(Shape) 정보를 추출합니다.
        # WHY: 차원 변환(reshape) 시 필요한 배치 사이즈(B), 시퀀스 길이(L, S), 임베딩 차원(E)을 알기 위함입니다.
        # HOW: query와 key의 shape 튜플을 언패킹합니다.
        B, L, E = query.shape
        _, S, _ = key.shape
        
        # WHAT: 입력 텐서들을 각 프로젝션 계층을 통과시켜 변환합니다.
        # WHY: 어텐션을 계산할 공간(Sub-space)으로 데이터를 매핑하기 위함입니다.
        # HOW: 미리 정의된 q_proj, k_proj, v_proj를 호출합니다.
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # WHAT: 차원을 분할하고 재배열하여 다중 헤드 형태로 만듭니다.
        # WHY: 1개의 거대한 행렬곱을 num_heads개의 독립적인 행렬곱으로 병렬화하기 위함입니다.
        # HOW: 형상을 (Batch, Length, Heads, HeadDim)으로 바꾼 뒤 (Batch, Heads, Length, HeadDim)으로 치환(permute)합니다.
        q = permute(reshape(q, (B, L, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = permute(reshape(k, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = permute(reshape(v, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        
        # WHAT: 어텐션 스코어 및 결과값 계산입니다.
        # WHY: 각 질의에 대해 모든 키와의 유사도를 구해 그 가중치만큼 Value를 혼합하기 위함입니다.
        # HOW: functional의 scaled_dot_product_attention 함수를 호출합니다.
        attn_out = scaled_dot_product_attention(q, k, v, attn_mask, self.dropout, is_causal, self.training)
        
        # WHAT: 다중 헤드 결과를 단일 텐서로 다시 병합합니다.
        # WHY: 다음 레이어로 넘기기 위해 원래 임베딩 차원 형태로 되돌리기 위함입니다.
        # HOW: permute로 헤드와 길이 차원을 되돌린 후 reshape로 묶습니다.
        attn_out = reshape(permute(attn_out, (0, 2, 1, 3)), (B, L, E))
        
        # WHAT: 결합된 결과에 최종 선형 변환을 적용합니다.
        # WHY: 독립적으로 추출된 특징들을 서로 교차(mix)시키고 모델 표현력을 강화하기 위함입니다.
        # HOW: out_proj를 호출하여 결과를 반환합니다.
        return self.out_proj(attn_out)

# WHAT: 트랜스포머 인코더의 단일 레이어 블록 클래스입니다.
# WHY: 자기 주의 메커니즘(Self-Attention)과 피드포워드 네트워크(FFN)를 결합해 시퀀스 내 문맥적 특징을 추출하기 위함입니다.
# HOW: MultiheadAttention, LayerNorm, Linear 레이어들을 순차적이고 잔차 연결(Residual Connection) 형태로 구성합니다.
class TransformerEncoderLayer(Module):
    # WHAT: 인코더 레이어의 초기화 메서드입니다.
    # WHY: 어텐션 계층과 피드포워드 다층 퍼셉트론(MLP) 및 정규화 계층을 생성하기 위함입니다.
    # HOW: 내부 멤버로 각 컴포넌트들을 선언하고 초기화합니다.
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # WHAT: 멀티헤드 셀프 어텐션 모듈입니다.
        # WHY: 시퀀스 데이터 자신 내부 요소들 간의 관계를 파악하기 위함입니다.
        # HOW: MultiheadAttention 클래스를 생성합니다.
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # WHAT: 피드포워드 신경망의 첫 번째 선형 계층입니다.
        # WHY: 어텐션으로 모인 정보를 고차원(보통 4배) 공간으로 확장해 비선형 패턴을 추출하기 위함입니다.
        # HOW: Linear(d_model, dim_feedforward)로 선언합니다.
        self.linear1 = Linear(d_model, dim_feedforward)
        
        # WHAT: 피드포워드 신경망 내부의 드롭아웃 레이어입니다.
        # WHY: 훈련 중 과적합을 방지하기 위함입니다.
        # HOW: Dropout 모듈을 생성합니다.
        self.dropout = Dropout(dropout)
        
        # WHAT: 피드포워드 신경망의 두 번째 선형 계층입니다.
        # WHY: 확장된 차원을 다시 원래의 임베딩 차원(d_model)으로 축소하기 위함입니다.
        # HOW: Linear(dim_feedforward, d_model)로 선언합니다.
        self.linear2 = Linear(dim_feedforward, d_model)
        
        # WHAT: 어텐션 연산 전/후에 적용할 레이어 정규화 모듈들입니다.
        # WHY: 층이 깊어짐에 따라 데이터 분포가 망가지는 것을 막아 안정적 학습을 보장하기 위함입니다.
        # HOW: LayerNorm 클래스로 두 개의 인스턴스를 생성합니다.
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # WHAT: 각 서브 레이어 결과를 기존 값과 더하기 전 적용하는 드롭아웃입니다.
        # WHY: 잔차 연결 부근에서의 정규화 및 과적합 제어를 위함입니다.
        # HOW: Dropout 객체를 생성합니다.
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        # WHAT: 피드포워드 신경망의 비선형 활성화 함수입니다.
        # WHY: 단순한 선형 변환이 아닌 복잡한 맵핑 함수를 학습하기 위함입니다.
        # HOW: ReLU 인스턴스를 생성합니다.
        self.activation = ReLU()
        
    # WHAT: 트랜스포머 인코더 레이어의 순전파 연산입니다.
    # WHY: 입력 시퀀스가 어텐션과 피드포워드를 거치며 어떻게 특징이 갱신되는지 정의하기 위함입니다.
    # HOW: Pre-LN 구조와 달리 Post-LN 형태를 차용하여 잔차 연결과 정규화를 적용합니다.
    def forward(self, src, src_mask=None, is_causal=False):
        from .ops import add
        
        # WHAT: 셀프 어텐션 블록의 연산 및 잔차 연결(Residual Connection)입니다.
        # WHY: 현재 입력(src)에 자기 자신과의 문맥 정보(src2)를 결합하기 위함입니다.
        # HOW: q, k, v 모두 src로 넣어 어텐션을 구한 후 dropout을 거쳐 기존 src에 더합니다.
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, is_causal=is_causal)
        src = add(src, self.dropout1(src2))
        
        # WHAT: 첫 번째 서브 레이어 이후의 정규화입니다.
        # WHY: 데이터 스케일을 안정화하기 위함입니다.
        # HOW: norm1을 통과시킵니다.
        src = self.norm1(src)
        
        # WHAT: 피드포워드 네트워크(FFN) 블록의 연산 및 잔차 연결입니다.
        # WHY: 각 토큰 위치마다 개별적으로 비선형성을 가해 고수준 특징을 얻기 위함입니다.
        # HOW: linear1 -> relu -> dropout -> linear2 -> dropout2를 통과시킨 후 이전 src에 더합니다.
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = add(src, self.dropout2(src2))
        
        # WHAT: 두 번째 서브 레이어 이후의 정규화입니다.
        # WHY: 출력값을 한 번 더 안정화하여 다음 레이어로 무사히 전달하기 위함입니다.
        # HOW: norm2를 통과시킨 후 최종 반환합니다.
        src = self.norm2(src)
        
        return src

# WHAT: 포지셔널 인코딩(Positional Encoding) 계층 클래스입니다.
# WHY: 트랜스포머는 RNN처럼 순차적으로 처리하지 않아 순서 정보가 없으므로, 데이터의 위치(순서) 정보를 인공적으로 부여하기 위함입니다.
# HOW: 사인(Sin)과 코사인(Cos) 함수의 서로 다른 주파수를 활용하여 정적 행렬을 만들어 입력에 더합니다.
class PositionalEncoding(Module):
    # WHAT: PositionalEncoding의 초기화 메서드입니다.
    # WHY: 모델 차원과 최대 길이에 맞춰 사인 곡선 기반의 위치 임베딩 매트릭스를 미리 계산해두기 위함입니다.
    # HOW: 수식에 따라 pe 행렬을 계산한 후 학습되지 않는(requires_grad=False) 상수로 저장합니다.
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        import numpy as np
        from .ops import tensor
        
        # WHAT: 위치 정보를 담을 0으로 초기화된 넘파이 행렬입니다.
        # WHY: 미리 최대 길이(max_len)만큼 생성하여 런타임 계산 비용을 아끼기 위함입니다.
        # HOW: shape가 (1, max_len, d_model)인 배열을 생성합니다.
        pe = np.zeros((1, max_len, d_model), dtype=np.float32)
        
        # WHAT: 시퀀스 내의 절대 위치(인덱스) 벡터입니다.
        # WHY: 주기 함수에 입력으로 들어갈 위치 값을 나타내기 위함입니다.
        # HOW: arange로 생성 후 2차원 컬럼 벡터로 변환합니다.
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        
        # WHAT: 차원 위치마다 다르게 적용될 주파수 조절항(Denominator)입니다.
        # WHY: 차원 단위로 주기를 늘려 각기 다른 스케일의 위치 특징을 담기 위함입니다.
        # HOW: 지수 함수를 이용해 10000 기반의 감쇠 계수를 만듭니다.
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model))
        
        # WHAT: 사인과 코사인 함수를 교차하여 매트릭스에 할당합니다.
        # WHY: 짝수 인덱스와 홀수 인덱스 차원에 서로 90도 위상 차를 두어 상대적 거리를 쉽게 학습할 수 있게 하기 위함입니다.
        # HOW: 짝수 인덱스에는 sin, 홀수 인덱스에는 cos 값을 대입합니다.
        pe[0, :, 0::2] = np.sin(position * div_term)
        # WHAT: 계산된 행렬을 텐서화하여 보관하고, 원본 CPU NumPy 데이터를 별도 보존합니다.
        # WHY: 텐서가 GPU로 이동된 후에도 다양한 가변 시퀀스 길이(seq_len) 슬라이스를 안전하게 생성하기 위함입니다.
        # HOW: _pe_raw에 float32 배열을 보관하고 _pe_cache로 재사용합니다.
        self._pe_raw = pe.copy().astype(np.float32)
        self.pe = tensor(pe, requires_grad=False)
        self._pe_cache = OrderedDict()
        
    # WHAT: 포지셔널 인코딩의 순전파 메서드입니다.
    # WHY: 실제 모델 입력 텐서에 순서 정보를 합성하기 위함입니다.
    # HOW: 디바이스/길이별 캐시에서 pe_slice를 조회하고 원래 입력값 x와 더해서(add) 반환합니다.
    def forward(self, x):
        from .ops import add, tensor
        from .errors import AMEVAForgeDeviceError
        if self.pe.device != x.device:
            raise AMEVAForgeDeviceError(
                f"PositionalEncoding buffer device '{self.pe.device}' does not match input device '{x.device}'. "
                f"Call model.to('{x.device}') before executing forward."
            )
            
        seq_len = x.shape[1]
        cache_key = (x.device, seq_len)
        if cache_key in self._pe_cache:
            self._pe_cache.move_to_end(cache_key)
        else:
            if len(self._pe_cache) >= 32:
                old_key, old_tensor = self._pe_cache.popitem(last=False)
                if getattr(old_tensor, 'device', None) == 'gpu':
                    try:
                        old_tensor.dispose()
                    except Exception:
                        pass
            pe_slice_data = self._pe_raw[:, :seq_len, :].astype(np.float32)
            self._pe_cache[cache_key] = tensor(pe_slice_data, device=x.device, requires_grad=False)
            
        pe_slice = self._pe_cache[cache_key]
        return add(x, pe_slice)


# WHAT: 단어 인덱스를 밀집 벡터(Dense Vector)로 변환하는 임베딩(Embedding) 계층입니다.
# WHY: 자연어 처리 등에서 불연속적인 토큰(예: 단어 ID)을 연속적인 고차원 공간으로 매핑하여 신경망이 의미를 학습할 수 있게 하기 위함입니다.
# HOW: (어휘 사전 크기) x (임베딩 차원) 크기의 가중치 행렬을 만들고, 인덱스를 받아 해당하는 벡터를 룩업(Lookup)합니다.
class Embedding(Module):
    # WHAT: 임베딩 계층 초기화 메서드입니다.
    # WHY: 어휘 사전 크기와 임베딩 차원을 받아 가중치 파라미터를 생성하기 위함입니다.
    # HOW: 랜덤 정규분포를 사용해 초기 가중치 행렬을 구성하고 학습 가능한 텐서로 만듭니다.
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # WHAT: 어휘 사전(Vocabulary)의 총 단어 개수입니다.
        # WHY: 룩업 테이블 행렬의 행(Row) 개수를 결정하기 위함입니다.
        # HOW: 멤버 변수로 저장합니다.
        self.num_embeddings = num_embeddings
        
        # WHAT: 각 단어가 표현될 밀집 벡터의 차원 수입니다.
        # WHY: 룩업 테이블 행렬의 열(Column) 개수를 결정하기 위함입니다.
        # HOW: 멤버 변수로 저장합니다.
        self.embedding_dim = embedding_dim
        
        # Standard normal initialization
        # WHAT: 임베딩 가중치의 초기 데이터 행렬입니다.
        # WHY: 모델이 처음부터 다양한 의미 공간을 탐색하도록 무작위로 분산시키기 위함입니다.
        # HOW: np.random.randn을 통해 (num_embeddings, embedding_dim) 크기의 배열을 생성합니다.
        data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        
        from .ops import tensor
        # WHAT: 임베딩 룩업 테이블 역할을 하는 학습 가능한 텐서입니다.
        # WHY: 훈련 과정을 통해 단어 간의 유사도와 관계를 가중치로 최적화하기 위함입니다.
        # HOW: requires_grad=True로 설정하여 저장합니다.
        self.weight = tensor(data, requires_grad=True)

    # WHAT: 임베딩 계층의 순전파 연산입니다.
    # WHY: 정수 인덱스 시퀀스를 받아 그에 대응하는 실수 벡터 시퀀스로 변환하기 위함입니다.
    # HOW: 내부 ops.embedding 함수를 호출하여 룩업을 수행합니다.
    def forward(self, x):
        from .ops import embedding
        return embedding(self.weight, x)

# WHAT: 기본 순환 신경망 셀(RNN Cell)을 구현한 클래스입니다.
# WHY: 단일 타임스텝(time step)에 대해 입력과 이전 은닉 상태(Hidden State)를 받아 새로운 은닉 상태를 계산하기 위함입니다.
# HOW: 현재 입력값(x)과 이전 상태(hx)를 각각의 가중치로 선형 변환한 후 합치고, Tanh 활성화 함수를 통과시킵니다.
class RNNCell(Module):
    # WHAT: RNNCell 초기화 메서드입니다.
    # WHY: 입력 차원과 은닉 차원에 맞는 가중치(Weight)와 편향(Bias) 파라미터들을 준비하기 위함입니다.
    # HOW: 균등 분포로 입력-은닉 간 가중치, 은닉-은닉 간 가중치, 그리고 각각의 편향을 초기화합니다.
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # WHAT: 파라미터 초기화를 위한 바운딩(k) 값입니다.
        # WHY: 가중치들이 너무 커지거나 작아지지 않도록 은닉층 크기에 반비례하게 제한하기 위함입니다.
        # HOW: 1.0 / hidden_size의 제곱근으로 계산합니다.
        k = (1.0 / hidden_size) ** 0.5
        
        # WHAT: 넘파이 난수로 생성한 가중치 및 편향 데이터 배열들입니다.
        # WHY: 입력-은닉 매핑(w_ih, b_ih)과 은닉-은닉 매핑(w_hh, b_hh)을 무작위로 분산시켜 놓기 위함입니다.
        # HOW: np.random.uniform을 이용해 [-k, k] 범위 내에서 추출합니다.
        w_ih = np.random.uniform(-k, k, (hidden_size, input_size)).astype(np.float32)
        w_hh = np.random.uniform(-k, k, (hidden_size, hidden_size)).astype(np.float32)
        b_ih = np.random.uniform(-k, k, (hidden_size,)).astype(np.float32)
        b_hh = np.random.uniform(-k, k, (hidden_size,)).astype(np.float32)
        
        from .ops import tensor
        # WHAT: 실제 학습에 사용될 텐서 객체들입니다.
        # WHY: 순전파 시 수식을 계산하고, 역전파 시 기울기를 구하기 위함입니다.
        # HOW: 각각을 tensor로 변환하고 requires_grad=True를 켭니다.
        self.weight_ih = tensor(w_ih, requires_grad=True)
        self.weight_hh = tensor(w_hh, requires_grad=True)
        self.bias_ih = tensor(b_ih, requires_grad=True)
        self.bias_hh = tensor(b_hh, requires_grad=True)

    # WHAT: RNNCell의 순전파 메서드입니다.
    # WHY: 타임스텝 t에서의 다음 은닉 상태(h_next)를 구하기 위함입니다.
    # HOW: 수식 h' = tanh(W_ih * x + b_ih + W_hh * h + b_hh)를 계산하여 반환합니다.
    def forward(self, x, hx=None):
        from .ops import zeros, matmul, transpose, add, tanh_op
        
        if hx is None:
            # WHAT: 이전 은닉 상태가 주어지지 않았을 때의 기본값 처리입니다.
            # WHY: 시퀀스의 첫 타임스텝에서는 이전 상태가 없으므로 0으로 초기화하기 위함입니다.
            # HOW: 입력 배치 크기(x.shape[0])와 hidden_size 모양을 갖는 0 텐서를 생성합니다.
            hx = zeros((x.shape[0], self.hidden_size), device=x.device)
            
        # h_next = tanh(x @ weight_ih.T + bias_ih + hx @ weight_hh.T + bias_hh)
        # WHAT: 입력값에 대한 선형 변환 결과(term1)입니다.
        # WHY: 외부 자극(입력)이 현재 상태에 미치는 영향을 계산하기 위함입니다.
        # HOW: 행렬 곱(matmul) 후 편향(bias)을 더합니다.
        term1 = add(matmul(x, transpose(self.weight_ih)), self.bias_ih)
        
        # WHAT: 이전 상태에 대한 선형 변환 결과(term2)입니다.
        # WHY: 과거의 문맥이 현재 상태에 미치는 영향을 계산하기 위함입니다.
        # HOW: 행렬 곱 연산 후 편향을 더합니다.
        term2 = add(matmul(hx, transpose(self.weight_hh)), self.bias_hh)
        
        # WHAT: 최종적으로 새로운 은닉 상태를 생성하는 활성화 함수 통과입니다.
        # WHY: 값의 범위를 [-1, 1]로 제한하여 발산을 막고 비선형성을 더하기 위함입니다.
        # HOW: 두 항을 더한 뒤 tanh_op를 씌웁니다.
        h_next = tanh_op(add(term1, term2))
        return h_next

# WHAT: 장단기 메모리(LSTM) 셀 클래스입니다.
# WHY: 일반적인 RNN의 장기 의존성(Long-Term Dependency) 문제인 기울기 소실을 해결하기 위해 셀 상태(Cell State)와 여러 게이트를 활용하기 위함입니다.
# HOW: 입력, 망각, 출력, 셀 게이트를 동시에 계산하고, 이를 결합하여 새로운 셀 상태와 은닉 상태를 생성합니다.
class LSTMCell(Module):
    # WHAT: LSTMCell 인스턴스를 초기화하는 메서드입니다.
    # WHY: 4개의 게이트(i, f, g, o)를 위한 가중치와 편향 파라미터를 하나로 묶어 할당하기 위함입니다.
    # HOW: 연산 속도 향상을 위해 4배 크기의 파라미터를 한 번에 생성합니다.
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # WHAT: 초기화 상한/하한값 제한 변수입니다.
        # WHY: 가중치 스케일을 통제하기 위함입니다.
        # HOW: 역제곱근 수식을 사용합니다.
        k = (1.0 / hidden_size) ** 0.5
        
        # WHAT: 게이트 4개(입력, 망각, 셀, 출력)의 선형 변환 행렬을 하나로 통합한 배열입니다.
        # WHY: 4번 따로 행렬 곱셈을 하는 것보다 한 번에 크게 곱하는 것이 하드웨어 병렬화에 유리하기 때문입니다.
        # HOW: 4 * hidden_size 크기의 배열로 생성합니다.
        w_ih = np.random.uniform(-k, k, (4 * hidden_size, input_size)).astype(np.float32)
        w_hh = np.random.uniform(-k, k, (4 * hidden_size, hidden_size)).astype(np.float32)
        b_ih = np.random.uniform(-k, k, (4 * hidden_size,)).astype(np.float32)
        b_hh = np.random.uniform(-k, k, (4 * hidden_size,)).astype(np.float32)
        
        from .ops import tensor
        # WHAT: 통합 가중치 및 편향 파라미터 텐서입니다.
        # WHY: 그래디언트 계산을 활성화하여 학습에 사용하기 위함입니다.
        # HOW: tensor로 변환하여 멤버 변수에 등록합니다.
        self.weight_ih = tensor(w_ih, requires_grad=True)
        self.weight_hh = tensor(w_hh, requires_grad=True)
        self.bias_ih = tensor(b_ih, requires_grad=True)
        self.bias_hh = tensor(b_hh, requires_grad=True)

    # WHAT: LSTMCell의 순전파 메서드입니다.
    # WHY: 현재 입력(x)과 이전 상태들(h, c)을 가지고 새로운 상태들(h_next, c_next)을 구하기 위함입니다.
    # HOW: 통합된 선형 연산을 거친 결과를 4등분(chunk)하여 각 게이트에 분배한 뒤 공식을 적용합니다.
    def forward(self, x, hx=None):
        from .ops import zeros, matmul, transpose, add, sigmoid, tanh_op, mul
        
        if hx is None:
            # WHAT: 이전 상태(h, c)가 누락되었을 때의 기본 처리입니다.
            # WHY: 시퀀스 첫 타임스텝에 0으로 된 초기 상태를 주입하기 위함입니다.
            # HOW: h와 c 각각을 0으로 채워진 텐서로 생성합니다.
            h = zeros((x.shape[0], self.hidden_size), device=x.device)
            c = zeros((x.shape[0], self.hidden_size), device=x.device)
        else:
            h, c = hx
            
        # WHAT: 4개의 게이트 입력값을 한 번의 수식으로 구한 결과 행렬입니다.
        # WHY: 효율적인 계산을 위해 결합된 가중치 행렬을 사용해 모두 동시에 계산하기 위함입니다.
        # HOW: W_ih * x + b_ih와 W_hh * h + b_hh를 각각 더합니다.
        gates = add(
            add(matmul(x, transpose(self.weight_ih)), self.bias_ih),
            add(matmul(h, transpose(self.weight_hh)), self.bias_hh)
        )
        
        # WHAT: 입력, 망각, 셀, 출력 게이트의 분할 및 활성화입니다.
        # WHY: 각각의 게이트가 메모리 갱신 과정에서 각자 맡은 역할을 수행하도록 분리하기 위함입니다.
        # HOW: 슬라이싱을 이용해 4등분한 뒤 시그모이드와 쌍곡탄젠트(tanh)를 씌웁니다.
        i_gate = sigmoid(gates[:, 0:self.hidden_size]) # Input gate
        f_gate = sigmoid(gates[:, self.hidden_size:2*self.hidden_size]) # Forget gate
        g_gate = tanh_op(gates[:, 2*self.hidden_size:3*self.hidden_size]) # Cell gate (후보군)
        o_gate = sigmoid(gates[:, 3*self.hidden_size:4*self.hidden_size]) # Output gate
        
        # WHAT: 새로운 셀 상태(Cell State) 업데이트입니다.
        # WHY: 과거의 정보를 지울 부분(f_gate * c)과 새롭게 기억할 부분(i_gate * g_gate)을 합치기 위함입니다.
        # HOW: 요소별 곱(mul)과 덧셈(add)을 사용해 c_next를 구합니다.
        c_next = add(mul(f_gate, c), mul(i_gate, g_gate))
        
        # WHAT: 새로운 은닉 상태(Hidden State) 업데이트입니다.
        # WHY: 다음 타임스텝이나 상위 레이어로 전달할 필터링된 출력을 생성하기 위함입니다.
        # HOW: c_next에 tanh를 적용한 후 출력 게이트와 요소별로 곱합니다.
        h_next = mul(o_gate, tanh_op(c_next))
        
        return h_next, c_next

# WHAT: 다층 시퀀스 처리를 위한 완전한 RNN(Recurrent Neural Network) 모듈입니다.
# WHY: 사용자가 단일 셀을 반복해서 호출하지 않고도, 전체 시퀀스를 한 번에 입력하여 결과를 얻을 수 있도록 감싸기(Wrapper) 위함입니다.
# HOW: RNNCell 인스턴스를 소유하고, 입력 시퀀스를 순회(loop)하며 매 타임스텝마다 셀을 호출해 결과를 누적합니다.
class RNN(Module):
    # WHAT: RNN 모듈 초기화 메서드입니다.
    # WHY: RNNCell을 내부에 생성하고 배치 차원 설정을 기억하기 위함입니다.
    # HOW: 셀 객체를 초기화하여 self.cell에 저장합니다.
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # WHAT: 입력 텐서의 차원 순서(배치가 먼저 오는지 여부) 플래그입니다.
        # WHY: True일 경우 (Batch, Seq, Feature)로, False일 경우 (Seq, Batch, Feature)로 다루기 위함입니다.
        # HOW: 불리언 값으로 저장합니다.
        self.batch_first = batch_first
        
        # WHAT: 연산을 수행할 내부 코어 유닛입니다.
        # WHY: 타임스텝 단위 계산을 델리게이트(위임)하기 위함입니다.
        # HOW: RNNCell 클래스를 인스턴스화합니다.
        self.cell = RNNCell(input_size, hidden_size)
        
    # WHAT: 전체 시퀀스에 대한 순환 연산을 실행하는 메서드입니다.
    # WHY: 각 시점 데이터들을 차례로 셀에 밀어넣고 출력과 최종 상태를 얻기 위함입니다.
    # HOW: 루프를 돌며 상태를 갱신하고 결과를 리스트에 모은 뒤 결합하여 반환합니다.
    def forward(self, x, hx=None):
        if x.device == "gpu":
            raise AMEVAForgeUnsupportedOperationError(
                "GPU RNN is not supported in Release 1. "
                "RNN requires GPU slice/time-step kernels that are not part of the Release 1 scope."
            )
        from .ops import cat, unsqueeze, permute
        
        if self.batch_first:
            # WHAT: 배치 퍼스트 입력일 경우 시퀀스 길이 축을 0번째로 위치하도록 변환합니다.
            # WHY: 시퀀스 루프를 시간 단위(t)로 쉽게 돌기 위함입니다.
            # HOW: (1, 0, 2) 순서로 permute 시킵니다.
            x = permute(x, (1, 0, 2))
            
        # WHAT: 전체 입력 시퀀스의 길이(타임스텝 수)입니다.
        # WHY: 몇 번 루프를 돌릴지 결정하기 위함입니다.
        # HOW: x.shape[0]으로 알아냅니다.
        seq_len = x.shape[0]
        
        # WHAT: 매 타임스텝의 출력 결과를 저장할 빈 리스트입니다.
        # WHY: 마지막에 하나로 결합(Concat)하기 위함입니다.
        # HOW: 빈 리스트를 만듭니다.
        outputs = []
        h = hx
        
        # WHAT: 시퀀스 길이에 따라 타임스텝을 순회하는 메인 루프입니다.
        # WHY: 과거 정보를 다음 단계로 차례로 넘겨주어 연속적인 추론을 수행하기 위함입니다.
        # HOW: t 인덱스를 사용해 x[t]를 뽑고 셀에 투입합니다.
        for t in range(seq_len):
            x_t = x[t]
            h = self.cell(x_t, h)
            # WHAT: 구해진 은닉 상태를 0번째 축(시간축)을 살려 리스트에 넣습니다.
            # WHY: 나중에 차원(dim=0) 기준으로 이어붙이기 위해 차원을 확장(unsqueeze)해 줍니다.
            # HOW: unsqueeze(h, 0)을 호출합니다.
            outputs.append(unsqueeze(h, 0))
            
        # WHAT: 리스트에 담긴 개별 출력값들을 단일 텐서로 합칩니다.
        # WHY: 네트워크의 최종 반환 형태로 만들기 위함입니다.
        # HOW: 차원 0을 기준으로 cat 연산을 수행합니다.
        out = cat(outputs, dim=0)
        
        if self.batch_first:
            # WHAT: 원래 batch_first 형태였다면 출력 형태도 되돌려줍니다.
            # WHY: 입력 포맷과 출력 포맷의 일관성을 유지하기 위함입니다.
            # HOW: 다시 (1, 0, 2) 순서로 permute합니다.
            out = permute(out, (1, 0, 2))
            
        return out, h

# WHAT: 다층 시퀀스 처리를 위한 완전한 LSTM 모듈입니다.
# WHY: 긴 시퀀스에서도 장기 의존성(Long-term dependency)을 안정적으로 학습하고 추론하기 위함입니다.
# HOW: 내부적으로 LSTMCell을 생성하여 시간축(타임스텝)을 따라 입력을 순회하며 연산합니다.
class LSTM(Module):
    # WHAT: LSTM 모듈의 초기화 메서드입니다.
    # WHY: LSTM 셀 인스턴스를 내부에 구성하고 설정들을 저장하기 위함입니다.
    # HOW: 입력 크기, 은닉 크기를 받아 LSTMCell을 초기화합니다.
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # WHAT: 입력 텐서의 배치 축(Batch dimension) 선행 여부를 나타내는 플래그입니다.
        # WHY: 사용자가 (Batch, Seq, ...)와 (Seq, Batch, ...) 중 편리한 데이터 형태를 쓰게 하기 위함입니다.
        # HOW: 불리언 값으로 보관합니다.
        self.batch_first = batch_first
        
        # WHAT: 실제 복잡한 게이트 연산을 담당하는 유닛입니다.
        # WHY: 전체 모듈은 루프만 제어하고 코어 로직은 위임(Delegation)하기 위함입니다.
        # HOW: LSTMCell 인스턴스를 생성해 저장합니다.
        self.cell = LSTMCell(input_size, hidden_size)
        
    # WHAT: 전체 시퀀스에 대한 LSTM 순전파 연산 메서드입니다.
    # WHY: 여러 타임스텝에 걸쳐 입력 데이터를 차례로 처리하여 문맥 결과를 도출하기 위함입니다.
    # HOW: 시퀀스 차원을 기준으로 루프를 반복하며 셀 상태(c)와 은닉 상태(h)를 계속해서 누적 갱신합니다.
    def forward(self, x, hx=None):
        if x.device == "gpu":
            raise AMEVAForgeUnsupportedOperationError(
                "GPU LSTM is not supported in Release 1. "
                "Use CPU LSTM or wait for Release 2 recurrent kernels."
            )
        from .ops import cat, unsqueeze, permute
        if self.batch_first:
            # WHAT: batch_first가 참일 경우 입력 데이터를 타임스텝 우선 순서로 뒤집습니다.
            # WHY: 인덱스로 루프를 돌기 쉽게 x[t] 형태로 맞추기 위함입니다.
            # HOW: permute(1, 0, 2)를 적용합니다.
            x = permute(x, (1, 0, 2))
            
        # WHAT: 처리해야 할 총 타임스텝 수입니다.
        # WHY: 순회할 범위를 알기 위함입니다.
        # HOW: 차원 0의 길이를 가져옵니다.
        seq_len = x.shape[0]
        
        # WHAT: 타임스텝별 반환 은닉 상태를 쌓을 리스트입니다.
        # WHY: 텐서 병합을 위해 임시로 모아두기 위함입니다.
        # HOW: 빈 리스트를 초기화합니다.
        outputs = []
        
        if hx is None:
            # WHAT: 이전 상태(h, c)가 없을 때의 초기화입니다.
            # WHY: None으로 두어 내부 셀(Cell)에서 자체적으로 0 초기화하게 냅두기 위함입니다.
            # HOW: 두 변수에 모두 None을 할당합니다.
            h, c = None, None
        else:
            # WHAT: 튜플(Tuple)로 들어온 이전 상태를 분리합니다.
            # WHY: 각각 은닉 상태(h)와 셀 상태(c)로 나누어 루프 갱신 변수로 쓰기 위함입니다.
            # HOW: 언패킹(Unpacking)을 수행합니다.
            h, c = hx
            
        # WHAT: 시간축(Time step)에 따른 순환 신경망 본체 루프입니다.
        # WHY: 연속적인 과거 정보가 미래로 흐르도록 처리하기 위함입니다.
        # HOW: t가 0부터 seq_len-1까지 순회하며 h, c를 지속적으로 갱신합니다.
        for t in range(seq_len):
            x_t = x[t]
            h, c = self.cell(x_t, (h, c) if h is not None else None)
            
            # WHAT: 얻어낸 은닉 상태를 차원 확장하여 리스트에 추가합니다.
            # WHY: 나중에 시퀀스 차원으로 묶어주기 위해 더미 축(0번)을 추가하기 위함입니다.
            # HOW: unsqueeze를 적용합니다.
            outputs.append(unsqueeze(h, 0))
            
        # WHAT: 모든 타임스텝 출력값을 텐서 덩어리로 합칩니다.
        # WHY: 출력 시퀀스를 구성하기 위함입니다.
        # HOW: 0번 축을 기준으로 cat 연산을 합니다.
        out = cat(outputs, dim=0)
        
        if self.batch_first:
            # WHAT: 원래 batch_first=True 인덱싱으로 데이터를 복구합니다.
            # WHY: 사용자가 주입한 형태와 동일한 출력을 보장하기 위함입니다.
            # HOW: 다시 permute(1, 0, 2)를 적용합니다.
            out = permute(out, (1, 0, 2))
            
        return out, (h, c)

```

---

## `packages/forge-py/src/forge/nn.pyi`

```typescript
from typing import Iterator, List, Optional, Tuple, Union
from .tensor import Tensor

class Module:
    def __init__(self) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor: ...
    def __call__(self, *args: Tensor, **kwargs: Tensor) -> Tensor: ...
    def parameters(self) -> List[Tensor]: ...
    def zero_grad(self) -> None: ...
    def to(self, device: str) -> Module: ...

class Linear(Module):
    in_features: int
    out_features: int
    use_bias: bool
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class ReLU(Module):
    def __init__(self) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MSELoss(Module):
    def __init__(self) -> None: ...
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor: ...

class Sequential(Module):
    def __init__(self, *modules: Module) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Module: ...
    def forward(self, x: Tensor) -> Tensor: ...

```

---

## `packages/forge-py/src/forge/ops.py`

```python
"""
================================================================================
[AMEVA-Forge 역사 기록 (Historical Metadata)]
생성일 (Created): Wed Aug 12 12:14:52 2026 +0900
수정 내역 (Modified):
- Wed Aug 12 13:10:12 2026 +0900: Fix: Add missing bmm implementation and rebuild wheel
- Wed Aug 12 12:59:35 2026 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
- Wed Aug 12 12:23:09 2026 +0900: Docs: Build Apache-style docs and unify tests
- Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
================================================================================

ops.py — 텐서 연산 API

C-09 Fix: 모든 assert를 명시적 에러 클래스로 교체.
M-03 Fix: CPU 경로에서 disposed 텐서 접근 시 AMEVAForgeDisposedError 발생.
M-04 Fix: should_use_gpu 논리 정리.
Lazy Fix: ones_like() GPU 텐서는 lazy upload 노드로 생성 (realize 호출 없음).
NL-05 Fix: ones_like를 공개 API로 노출.
"""
# WHAT: numpy 라이브러리를 임포트합니다.
# WHY: 다차원 배열 연산 및 수학 함수 기능을 고속으로 처리하기 위함입니다.
# HOW: np라는 별칭으로 사용하여 텐서의 내부 데이터(_data)를 다룹니다.
import numpy as np

# WHAT: typing 모듈에서 타입 힌팅을 위한 요소들을 임포트합니다.
# WHY: 함수의 인자와 반환값 타입을 명시하여 코드의 안정성과 가독성을 높이기 위함입니다.
# HOW: Any, Tuple, Optional 등을 사용해 타입을 어노테이션합니다.
from typing import Any, Tuple, Optional

# WHAT: 내부 모듈에서 Tensor 클래스를 임포트합니다.
# WHY: 모든 연산 함수의 입력 및 출력 기본 단위로 텐서를 사용하기 위함입니다.
# HOW: 텐서 객체의 속성(device, data 등)을 검사하고 새로운 텐서를 반환할 때 사용합니다.
from .tensor import Tensor

# WHAT: 내부 에러 모듈에서 커스텀 예외 클래스들을 임포트합니다.
# WHY: 연산 중 발생하는 예외 상황(기기 불일치, 차원 오류, 메모리 해제 등)을 명확하게 처리하기 위함입니다.
# HOW: 조건에 맞지 않을 때 raise 구문을 통해 발생시킵니다.
from .errors import (
    AMEVAForgeDeviceError,
    AMEVAForgeShapeError,
    AMEVAForgeDisposedError,
    AMEVAForgeUnsupportedOperationError,
)

# WHAT: 자동 미분(autograd) 구현을 위한 베이스 클래스들을 임포트합니다.
# WHY: 각 수학 연산이 순전파와 역전파를 지원하는 연산 노드로 동작하게 만들기 위함입니다.
# HOW: 모든 연산 클래스는 Function을 상속받고 forward/backward에서 Context(ctx)를 사용합니다.
from .autograd import Function, Context, no_grad

# ─── Debug Mode ──────────────────────────────────────────────────────────────
# VUL-003/004: NumPy/PyTorch 기본 동작 유지, debug mode에서만 경고 활성화

# WHAT: 디버그 모드의 활성화 여부를 나타내는 전역 불리언 변수입니다.
# WHY: 0으로 나누기 등 수치적 불안정 상황 발생 시 경고(warning)를 출력할지 결정하기 위함입니다.
# HOW: 기본값은 False이며, set_debug_mode를 통해 변경됩니다.
_debug_mode: bool = False

# WHAT: 전역 디버그 모드를 설정하는 함수입니다.
# WHY: 사용자가 코드 외부에서 런타임에 디버그 경고 활성화 여부를 켜고 끌 수 있게 하기 위함입니다.
# HOW: global 키워드를 사용하여 _debug_mode 변수의 값을 인자 enabled로 덮어씁니다.
def set_debug_mode(enabled: bool = True) -> None:
    """디버그 모드를 설정한다. 활성화 시 div-by-zero, log-of-non-positive 등을 경고한다."""
    global _debug_mode
    _debug_mode = enabled

# WHAT: 현재 디버그 모드 상태를 반환하는 함수입니다.
# WHY: 연산 내부에서 경고 메시지를 출력할지 판단하기 위한 조건으로 쓰기 위함입니다.
# HOW: 전역 변수 _debug_mode를 단순히 리턴합니다.
def get_debug_mode() -> bool:
    return _debug_mode


# WHAT: 텐서가 유효한 CPU 데이터를 가지고 있는지 검사하고 반환하는 유틸리티 함수입니다.
# WHY: 메모리 해제(disposed)되었거나 초기화되지 않은 텐서에 접근하여 발생할 수 있는 치명적 에러를 사전에 방지하기 위함입니다.
# HOW: 텐서의 _data 속성이 None인지 확인하고, 그렇다면 명시적 에러를 던지며 아니면 _data를 반환합니다.
def _require_cpu_data(t: Tensor, name: str = "tensor") -> np.ndarray:
    """
    C-09/M-03 Fix: CPU 텐서의 _data를 안전하게 요구한다.
    None이면 disposed 에러 발생 (assert 대신 명시적 에러).
    """
    if t._data is None:
        raise AMEVAForgeDisposedError(
            f"CPU tensor '{name}' has no data. It may have been disposed or not yet initialized."
        )
    return t._data


# WHAT: 두 텐서가 동일한 디바이스(CPU 또는 GPU)에 있는지 확인하는 함수입니다.
# WHY: 이기종 기기 간의 연산을 시도할 때 발생하는 크래시를 방지하고 명시적인 에러 메시지를 제공하기 위함입니다.
# HOW: a.device와 b.device를 비교하여 다르면 AMEVAForgeDeviceError를 발생시킵니다.
def _ensure_same_device(a: Tensor, b: Tensor, op: str = "operation") -> None:
    """
    M-04 Fix: 두 텐서의 기기가 다르면 명시적 에러.
    """
    if a.device != b.device:
        raise AMEVAForgeDeviceError(
            f"Cannot perform '{op}' on tensors from different devices: "
            f"'{a.device}' and '{b.device}'. "
            f"Move tensors to the same device first."
        )


# WHAT: 연산을 GPU에서 수행해야 하는지 판별하는 함수입니다.
# WHY: 텐서의 디바이스 위치에 따라 CPU 분기와 GPU(커널) 분기를 적절히 나누기 위함입니다.
# HOW: b가 주어졌을 때는 둘 다 gpu인지, b가 없을 때는 a가 gpu인지 확인하여 불리언 값을 반환합니다.
def _should_use_gpu(a: Tensor, b: Optional[Tensor] = None) -> bool:
    """
    M-04 Fix: 모든 피연산자가 gpu일 때만 True를 반환.
    """
    if b is not None:
        return a.device == "gpu" and b.device == "gpu"
    return a.device == "gpu"


# ─────────────────────────────────────────────────────────────────────────────
# 텐서 생성 함수들
# ─────────────────────────────────────────────────────────────────────────────

# WHAT: 파이썬 리스트나 넘파이 배열 등으로부터 Tensor 객체를 생성하는 팩토리 함수입니다.
# WHY: 사용자가 원시 데이터를 프레임워크가 이해할 수 있는 텐서 객체로 손쉽게 변환하기 위함입니다.
# HOW: 데이터 타입을 float32로 통일하고, 지정된 디바이스에 맞게 텐서를 반환합니다.
def tensor(
    data: Any,
    device: str = "cpu",
    dtype: str = "float32",
    requires_grad: bool = False
) -> Tensor:
    """데이터로부터 텐서를 생성한다."""
    if device not in {"cpu", "gpu"}:
        from .errors import AMEVAForgeDeviceError
        raise AMEVAForgeDeviceError(
            f"Unsupported device: {device!r}. "
            "Supported devices are 'cpu' and 'gpu'."
        )
        
    if isinstance(data, np.ndarray):
        # WHAT: 입력이 이미 넘파이 배열인 경우의 처리입니다.
        # WHY: float32가 아니면 캐스팅하여 연산 일관성을 맞추기 위함입니다.
        # HOW: dtype을 확인하고 다르면 astype으로 변환합니다.
        arr = data if data.dtype == np.float32 else data.astype(np.float32)
    else:
        # WHAT: 입력이 리스트 등 일반 파이썬 객체인 경우의 처리입니다.
        # WHY: 배열 형태로 표준화하기 위함입니다.
        # HOW: np.array 함수를 호출합니다.
        arr = np.array(data, dtype=np.float32)

    import warnings
    if np.any(np.isnan(arr)):
        warnings.warn(
            "Input contains NaN values. This may cause unexpected results in GPU operations.",
            RuntimeWarning, stacklevel=2
        )
    if np.any(np.isinf(arr)):
        warnings.warn(
            "Input contains Inf values. This may cause unexpected results in GPU operations.",
            RuntimeWarning, stacklevel=2
        )

    if device == "gpu":
        # WHAT: GPU 텐서를 생성합니다.
        # WHY: 데이터를 VRAM으로 업로드하는 명령(upload)을 지연 실행(lazy) 형태로 스케줄링하기 위함입니다.
        # HOW: op='upload' 인자를 포함하여 텐서 객체를 만듭니다.
        return Tensor(
            shape=arr.shape, dtype=dtype, device="gpu",
            requires_grad=requires_grad, data=arr, op='upload'
        )
    else:
        # WHAT: CPU 텐서를 생성합니다.
        # WHY: 로컬 메모리에서 즉시 연산 가능한 텐서를 제공하기 위함입니다.
        # HOW: 데이터를 포함하여 텐서를 반환합니다.
        return Tensor(
            shape=arr.shape, dtype=dtype, device=device,
            requires_grad=requires_grad, data=arr
        )


# WHAT: 0에서 1 사이의 균등 분포 난수로 채워진 텐서를 생성합니다.
# WHY: 가중치 초기화나 랜덤 데이터가 필요할 때 사용하기 위함입니다.
# HOW: np.random.random을 통해 배열을 만들고 텐서 래퍼를 반환합니다.
def random(
    shape: Tuple[int, ...],
    device: str = "cpu",
    dtype: str = "float32",
    requires_grad: bool = False
) -> Tensor:
    """균등 분포 난수 텐서를 생성한다."""
    # WHAT: 생성될 난수 배열 데이터입니다.
    # WHY: 지정된 shape의 초기값을 세팅하기 위함입니다.
    # HOW: 넘파이 랜덤 함수를 사용하고 float32로 캐스팅합니다.
    arr = np.random.random(shape).astype(np.float32)
    if device == "cpu":
        return Tensor(shape=shape, dtype=dtype, device=device,
                      requires_grad=requires_grad, data=arr)
    else:
        return Tensor(shape=shape, dtype=dtype, device="gpu",
                      requires_grad=requires_grad, data=arr, op='upload')


# WHAT: 입력 텐서 x와 동일한 크기와 디바이스를 가지며 1로 채워진 텐서를 반환합니다.
# WHY: 덧셈의 항등원이나 역전파 시 맨 처음 흘려보낼 그래디언트를 생성할 때 사용하기 위함입니다.
# HOW: np.ones로 배열을 만들고 입력 텐서의 속성을 복사하여 텐서를 리턴합니다.
def ones_like(x: Tensor) -> Tensor:
    """
    x와 같은 shape/device의 1-텐서를 생성한다.
    NL-05 Fix: 공개 API로 노출됨 (__init__.py 참조).
    GPU 텐서의 경우 lazy upload 노드로 생성하여
    상위 그래프의 realize() 시점에 함께 제출된다 (레이지 철학 준수).
    """
    # WHAT: 1로 채워진 배열입니다.
    # WHY: 내부 데이터를 초기화하기 위함입니다.
    # HOW: x.shape 크기만큼 np.ones를 호출합니다.
    arr = np.ones(x.shape, dtype=np.float32)
    if x.device == "cpu":
        return Tensor(shape=x.shape, dtype=x.dtype, device="cpu", data=arr)
    else:
        # Lazy upload: realize() 호출 없이 그래프에 합류
        return Tensor(shape=x.shape, dtype="float32", device="gpu",
                   data=arr, op='upload')


# WHAT: 입력 텐서 x와 동일한 크기와 디바이스를 가지며 0으로 채워진 텐서를 생성합니다.
# WHY: 변수 초기화나 마스킹 용도로 빈 공간을 마련하기 위함입니다.
# HOW: np.zeros로 0 행렬을 만든 뒤 텐서로 감쌉니다.
def zeros_like(x: Tensor) -> Tensor:
    """x와 같은 shape/device의 0-텐서를 생성한다."""
    arr = np.zeros(x.shape, dtype=np.float32)
    if x.device == "cpu":
        return Tensor(shape=x.shape, dtype=x.dtype, device="cpu", data=arr)
    else:
        # Lazy upload: realize() 호출 없이 그래프에 합류
        return Tensor(shape=x.shape, dtype="float32", device="gpu",
                   data=arr, op='upload')


# WHAT: 사용자가 직접 shape을 지정하여 0으로 채워진 텐서를 생성합니다.
# WHY: 새로운 편향(Bias) 파라미터나 특정 크기의 초기 텐서를 만들기 위함입니다.
# HOW: np.zeros를 사용 후 tensor() 팩토리 함수를 호출합니다.
def zeros(
    shape: Tuple[int, ...],
    device: str = "cpu",
    dtype: str = "float32",
    requires_grad: bool = False
) -> Tensor:
    """0으로 채워진 텐서를 생성한다."""
    arr = np.zeros(shape, dtype=np.float32)
    return tensor(arr, device=device, dtype=dtype, requires_grad=requires_grad)


# WHAT: 사용자가 지정한 크기로 1로 채워진 텐서를 생성합니다.
# WHY: 가중치의 배율을 1로 초기화하거나 특정 연산의 마스크로 사용하기 위함입니다.
# HOW: np.ones 배열을 만들어 tensor()로 래핑합니다.
def ones(
    shape: Tuple[int, ...],
    device: str = "cpu",
    dtype: str = "float32",
    requires_grad: bool = False
) -> Tensor:
    """1로 채워진 텐서를 생성한다."""
    arr = np.ones(shape, dtype=np.float32)
    return tensor(arr, device=device, dtype=dtype, requires_grad=requires_grad)


# WHAT: 주어진 값을 모든 요소에 채워 넣는 텐서 생성 함수입니다.
# WHY: 임의의 고정된 상수값(예: 0.5, 2.0 등)으로 구성된 텐서가 필요할 때 사용하기 위함입니다.
# HOW: np.full을 이용해 데이터를 채우고 tensor()를 반환합니다.
def full(
    shape: Tuple[int, ...],
    fill_value: float,
    device: str = "cpu",
    dtype: str = "float32",
    requires_grad: bool = False
) -> Tensor:
    """특정 값으로 채워진 텐서를 생성한다."""
    arr = np.full(shape, fill_value, dtype=np.float32)
    return tensor(arr, device=device, dtype=dtype, requires_grad=requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 수학 연산 (Function 기반 autograd)
# ─────────────────────────────────────────────────────────────────────────────

# WHAT: 덧셈 연산을 위한 자동 미분(autograd) 지원 클래스입니다.
# WHY: 두 텐서의 요소별 덧셈을 수행하고 역전파 시 그래디언트를 올바르게 분배하기 위함입니다.
# HOW: Function을 상속받아 forward와 backward 정적 메서드를 구현합니다.
class AddFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # WHAT: 역전파에 필요한 입력 텐서들을 컨텍스트에 저장합니다.
        # WHY: 체인 룰(chain rule) 계산 시 이전 노드로 기울기를 전달하기 위해 원본 텐서 구조가 필요하기 때문입니다.
        # HOW: ctx.save_for_backward()를 호출합니다.
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "add")
        
        # WHAT: 두 텐서의 형상이 다를 때 브로드캐스팅(broadcasting)된 최종 형상을 계산합니다.
        # WHY: 크기가 다른 배열(예: [10, 5] + [5])도 차원 확장을 통해 자연스럽게 더할 수 있도록 하기 위함입니다.
        # HOW: 내부 유틸리티 _broadcast_shapes를 사용합니다.
        out_shape = _broadcast_shapes(a.shape, b.shape)
        
        # WHAT: 원본 텐서 a, b의 형상을 각각 저장합니다.
        # WHY: 역전파 시 브로드캐스트된 기울기를 다시 원래 모양으로 축소(unbroadcast)해야 하기 때문입니다.
        # HOW: 컨텍스트 객체의 속성으로 할당합니다.
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
            
        if _should_use_gpu(a, b):
            a_numel = 1
            for s in a.shape:
                a_numel *= s
            b_numel = 1
            for s in b.shape:
                b_numel *= s
            return Tensor(shape=out_shape, dtype="float32", device="gpu",
                          op='add', parents=(a, b), op_params=[a_numel, b_numel])
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            res = data_a + data_b
            return Tensor(shape=out_shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # WHAT: 덧셈의 역전파로, 흘러들어온 기울기를 두 입력에 그대로 전달합니다.
        # WHY: 덧셈 연산자 f(a,b) = a + b 의 편미분 값은 각각 1이므로 기울기가 그대로 복사되기 때문입니다.
        # HOW: 브로드캐스팅이 일어났을 수 있으므로 _unbroadcast 함수를 거쳐 원본 크기로 맞춥니다.
        return _unbroadcast(grad_output, ctx.a_shape), _unbroadcast(grad_output, ctx.b_shape)


# WHAT: 사용자 친화적인 텐서 덧셈 함수입니다.
# WHY: 클래스 인스턴스화 과정을 숨기고 단순한 함수 호출(add(a, b))로 사용하게 하기 위함입니다.
# HOW: AddFunction의 apply 메서드를 호출합니다.
def add(a: Tensor, b: Tensor) -> Tensor:
    return AddFunction.apply(a, b)


# WHAT: 요소별 곱셈(Element-wise Multiplication)을 위한 연산 클래스입니다.
# WHY: 두 텐서의 위치가 같은 요소들끼리 곱을 수행하고 자동 미분을 지원하기 위함입니다.
# HOW: forward 시 배열 곱을 수행하고, backward 시 교차 곱(cross multiplication)을 적용합니다.
class MulFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "mul")
        out_shape = _broadcast_shapes(a.shape, b.shape)
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
            
        if _should_use_gpu(a, b):
            a_numel = 1
            for s in a.shape:
                a_numel *= s
            b_numel = 1
            for s in b.shape:
                b_numel *= s
            return Tensor(shape=out_shape, dtype="float32", device="gpu",
                          op='mul', parents=(a, b), op_params=[a_numel, b_numel])
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            res = data_a * data_b
            return Tensor(shape=out_shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # WHAT: 컨텍스트에 저장된 입력 텐서들을 꺼내옵니다.
        # WHY: 곱셈의 미분 규칙(d(ab)/da = b, d(ab)/db = a)을 적용하기 위해 원본 값이 필요하기 때문입니다.
        # HOW: ctx.saved_tensors 튜플을 언패킹합니다.
        a, b = ctx.saved_tensors
        # WHAT: 역전파된 기울기에 각각 상대방 텐서를 곱하고 형태를 복원합니다.
        # WHY: 체인 룰을 통해 올바른 기울기를 분배하기 위함입니다.
        # HOW: grad_output * b, grad_output * a 연산을 수행한 뒤 unbroadcast 합니다.
        return _unbroadcast(mul(grad_output, b), ctx.a_shape), _unbroadcast(mul(grad_output, a), ctx.b_shape)


# WHAT: 곱셈 연산 편의 함수입니다.
# WHY: 외부에서 쉽게 호출할 수 있는 인터페이스를 제공하기 위함입니다.
# HOW: MulFunction.apply를 통해 실행합니다.
def mul(a: Tensor, b: Tensor) -> Tensor:
    return MulFunction.apply(a, b)


# WHAT: 행렬 곱(Matrix Multiplication) 연산을 위한 클래스입니다.
# WHY: 신경망의 선형 변환 등에서 필수적으로 사용되는 내적 계산을 제공하고 그래디언트를 역전파하기 위함입니다.
# HOW: 입력 형상을 검사한 뒤 넘파이의 matmul 등을 이용해 순전파를, 전치행렬(transpose)을 이용해 역전파를 수행합니다.
class MatmulFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "matmul")

        if len(a.shape) != 2 or len(b.shape) != 2:
            raise AMEVAForgeShapeError(
                f"Matmul requires 2D tensors, got shapes {a.shape} and {b.shape}"
            )
            
        # WHAT: 행렬 크기 정보를 변수에 할당합니다.
        # WHY: 행렬 곱의 정의(M x K @ K x N -> M x N)에 부합하는지 확인하고 에러를 뿜기 위함입니다.
        # HOW: 튜플 언패킹을 사용합니다.
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            raise AMEVAForgeShapeError(
                f"Matmul inner dimension mismatch: {a.shape} @ {b.shape} "
                f"({K} != {K2})"
            )

        if _should_use_gpu(a, b):
            return Tensor(shape=(M, N), dtype="float32", device="gpu",
                          op='matmul', parents=(a, b), op_params=[int(M), int(N), int(K)])
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            res = np.matmul(data_a, data_b)
            return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        # WHAT: 행렬 곱 역전파 공식을 적용합니다 (dA = dY @ B.T, dB = A.T @ dY).
        # WHY: 손실 함수에 대한 행렬 연산의 미분을 정확히 계산하기 위함입니다.
        # HOW: 전치행렬 함수 transpose를 사용하여 역전파된 기울기와 각각 행렬 곱을 합니다.
        grad_a = matmul(grad_output, transpose(b))
        grad_b = matmul(transpose(a), grad_output)
        return grad_a, grad_b

# WHAT: 행렬 곱 연산 편의 함수입니다.
# WHY: 단순 2D 행렬뿐만 아니라 다차원(배치) 행렬 곱도 지원하기 위해 중간 래퍼 역할을 하기 위함입니다.
# HOW: 차원을 검사해 배치 행렬 곱(bmm)이나 reshape 트릭을 사용하여 MatmulFunction에 전달합니다.
def matmul(a: Tensor, b: Tensor) -> Tensor:
    if len(a.shape) == 3 and len(b.shape) == 3:
        # WHAT: 3차원 텐서의 경우 배치 행렬 곱을 수행합니다.
        # WHY: 트랜스포머의 어텐션 연산 등에서 배치를 유지한 채 내적하기 위함입니다.
        # HOW: 내부의 bmm 함수로 위임합니다.
        return bmm(a, b)
        
    if len(a.shape) > 2 and len(b.shape) == 2:
        # WHAT: a가 3차원 이상이고 b가 2차원인 경우 평탄화(Flatten) 기반 곱셈을 합니다.
        # WHY: 완전 연결 계층(Linear)에서 배치+시퀀스 차원을 유지한 채 가중치 연산을 처리하기 위함입니다.
        # HOW: a를 2차원으로 눌렀다가(reshape) 곱셈 후 다시 원래 차원으로 되돌립니다.
        orig_shape = a.shape
        flat_size = 1
        for s in orig_shape[:-1]:
            flat_size *= s
        a_2d = a.reshape((flat_size, orig_shape[-1]))
        out_2d = MatmulFunction.apply(a_2d, b)
        return out_2d.reshape(orig_shape[:-1] + (b.shape[-1],))
        
    return MatmulFunction.apply(a, b)


# WHAT: 2차원 행렬의 전치(Transpose) 연산을 수행하는 클래스입니다.
# WHY: 행렬의 행과 열을 뒤바꿔 행렬 곱 등에서 호환되는 차원을 맞추기 위함입니다.
# HOW: 순전파 시 np.transpose를 사용하고, 역전파 시 기울기를 다시 전치시킵니다.
class TransposeFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        if len(x.shape) != 2:
            raise AMEVAForgeShapeError(
                f"Transpose requires a 2D tensor, got shape {x.shape}"
            )
        M, N = x.shape
        if _should_use_gpu(x):
            return Tensor(shape=(N, M), dtype="float32", device="gpu",
                          op='transpose', parents=(x,), op_params=[int(M), int(N)])
        else:
            data = _require_cpu_data(x, "x")
            res = np.transpose(data)
            return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        return (transpose(grad_output),)


# WHAT: 전치 연산 편의 함수입니다.
# WHY: 코드 작성 시 직관적인 transpose() 호출을 허용하기 위함입니다.
# HOW: TransposeFunction.apply를 실행합니다.
def transpose(x: Tensor) -> Tensor:
    return TransposeFunction.apply(x)


# WHAT: 렐루(ReLU, Rectified Linear Unit) 활성화 함수를 구현한 클래스입니다.
# WHY: 비선형성을 제공하며, 0 미만의 값을 버림으로써 그래디언트 소실을 방지하기 위함입니다.
# HOW: 순전파 시 max(0, x)를, 역전파 시 0보다 큰 입력 위치에만 1의 기울기를 곱합니다.
class ReLUFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        if _should_use_gpu(x):
            return Tensor(shape=x.shape, dtype="float32", device="gpu",
                          op='relu', parents=(x,))
        else:
            data = _require_cpu_data(x, "x")
            # WHAT: 입력 배열의 요소 중 0보다 작은 값을 0으로 변환합니다.
            # WHY: ReLU 수식을 적용하기 위함입니다.
            # HOW: np.maximum을 사용하여 0과 비교합니다.
            res = np.maximum(data, 0.0)
            return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        if x.device == "cpu":
            data_x = _require_cpu_data(x, "x")
            data_g = _require_cpu_data(grad_output, "grad_output")
            # WHAT: 원본 입력 텐서가 0보다 컸던 곳에만 그래디언트를 통과시키는 연산입니다.
            # WHY: ReLU 미분값이 x>0일 때 1, 그렇지 않을 때 0이기 때문입니다.
            # HOW: 조건문(data_x > 0)으로 마스크를 만들고 그래디언트와 요소별 곱셈을 합니다.
            grad = data_g * (data_x > 0).astype(np.float32)
            return (Tensor(shape=x.shape, dtype="float32", device="cpu", data=grad),)
        else:
            return (Tensor(shape=x.shape, dtype="float32", device="gpu",
                           op='relu_backward', parents=(x, grad_output)),)


# WHAT: ReLU 활성화 편의 함수입니다.
# WHY: 직관적인 함수 호출을 제공하기 위함입니다.
# HOW: ReLUFunction.apply를 통해 텐서를 전달합니다.
def relu(x: Tensor) -> Tensor:
    return ReLUFunction.apply(x)


# WHAT: 뺄셈 연산을 지원하는 자동 미분 클래스입니다.
# WHY: 텐서 간의 차이를 구하고 역전파 시 미분값을 적절히 분배하기 위함입니다.
# HOW: a - b를 계산하고, backward 시 a에는 grad_output을, b에는 -grad_output을 줍니다.
class SubFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, 'sub')
        out_shape = _broadcast_shapes(a.shape, b.shape)
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        if _should_use_gpu(a, b):
            a_numel = 1
            for s in a.shape:
                a_numel *= s
            b_numel = 1
            for s in b.shape:
                b_numel *= s
            return Tensor(shape=out_shape, dtype='float32', device='gpu', op='sub', parents=(a, b), op_params=[a_numel, b_numel])
        else:
            return Tensor(shape=out_shape, dtype='float32', device='cpu', data=_require_cpu_data(a, "a") - _require_cpu_data(b, "b"))
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # WHAT: 뺄셈의 역전파로, a 방향으로는 양의 기울기를, b 방향으로는 음의 기울기를 전달합니다.
        # WHY: d(a-b)/da = 1, d(a-b)/db = -1 이기 때문입니다.
        # HOW: neg(grad_output)를 사용하여 b의 기울기를 반전시킨 뒤 언브로드캐스트합니다.
        return _unbroadcast(grad_output, ctx.a_shape), _unbroadcast(neg(grad_output), ctx.b_shape)


# WHAT: 뺄셈 연산 편의 함수입니다.
# WHY: 쉽게 뺄셈을 사용할 수 있게 하기 위함입니다.
# HOW: SubFunction.apply를 호출합니다.
def sub(a: Tensor, b: Tensor) -> Tensor:
    return SubFunction.apply(a, b)


# WHAT: 부호 반전(Negation) 연산을 지원하는 자동 미분 클래스입니다.
# WHY: 텐서 요소들의 부호를 바꾸고, 그에 따른 미분 연산을 처리하기 위함입니다.
# HOW: x의 부호를 뒤집고, backward 시에도 기울기 부호를 뒤집어 넘깁니다.
class NegFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        if _should_use_gpu(x):
            return Tensor(shape=x.shape, dtype='float32', device='gpu', op='neg', parents=(x,))
        else:
            return Tensor(shape=x.shape, dtype='float32', device='cpu', data=-_require_cpu_data(x, "x"))
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        # WHAT: 부호 반전의 역전파 연산입니다.
        # WHY: f(x) = -x 미분값이 -1이기 때문에 기울기의 부호가 바뀌어야 합니다.
        # HOW: neg 함수를 재귀적으로 호출하여 반환합니다.
        return (neg(grad_output),)


# WHAT: 부호 반전 편의 함수입니다.
# WHY: 직관적인 호출(-x)을 지원하는 백엔드 함수로 쓰기 위함입니다.
# HOW: NegFunction.apply를 실행합니다.
def neg(x: Tensor) -> Tensor:
    return NegFunction.apply(x)


# WHAT: 나눗셈 연산을 지원하는 자동 미분 클래스입니다.
# WHY: 두 텐서 간의 나누기를 계산하고 몫의 미분법(Quotient Rule)을 구현하기 위함입니다.
# HOW: numpy 나눗셈을 수행하고, backward 시 미분 공식을 적용해 a와 b로 기울기를 분배합니다.
class DivFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, 'div')
        out_shape = _broadcast_shapes(a.shape, b.shape)
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        if _should_use_gpu(a, b):
            return Tensor(shape=out_shape, dtype='float32', device='gpu', op='div', parents=(a, b))
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            # VUL-003: debug mode에서만 zero-division 경고
            if _debug_mode and np.any(data_b == 0):
                import warnings
                warnings.warn(
                    "[AMEVA debug] Division by zero detected. "
                    "Result contains inf/nan. Use set_debug_mode(False) to suppress.",
                    RuntimeWarning, stacklevel=4
                )
            return Tensor(shape=out_shape, dtype='float32', device='cpu', data=data_a / data_b)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        # WHAT: 분자(a) 방향으로의 기울기(dA)를 계산합니다.
        # WHY: f = a/b 일 때 df/da = 1/b 이기 때문입니다.
        # HOW: grad_output을 b로 나눕니다.
        grad_a = div(grad_output, b)
        
        # WHAT: 분모(b) 방향으로의 기울기(dB)를 계산합니다.
        # WHY: df/db = -a / (b^2) 이기 때문입니다.
        # HOW: -grad_output * a / b^2 수식을 적용합니다.
        grad_b = neg(mul(div(grad_output, mul(b, b)), a))  # -grad * a / b^2
        return _unbroadcast(grad_a, ctx.a_shape), _unbroadcast(grad_b, ctx.b_shape)


# WHAT: 나눗셈 연산 편의 함수입니다.
# WHY: 외부에서 텐서 나눗셈을 쉽게 사용할 수 있도록 하기 위함입니다.
# HOW: DivFunction.apply를 호출합니다.
def div(a: Tensor, b: Tensor) -> Tensor:
    return DivFunction.apply(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# 편의 함수
# ─────────────────────────────────────────────────────────────────────────────

# WHAT: 텐서를 넘파이 배열로 동기 변환하는 함수입니다.
# WHY: 외부 라이브러리(matplotlib, sklearn 등)와의 호환성 및 디버깅을 위해 배열 데이터를 추출하기 위함입니다.
# HOW: 텐서의 numpy() 메서드를 호출합니다.
def to_numpy(x: Tensor) -> np.ndarray:
    return x.numpy()


# WHAT: 텐서를 넘파이 배열로 비동기 변환하는 함수입니다.
# WHY: GPU에서 CPU로 메모리를 복사할 때 메인 스레드를 블로킹하지 않고 다른 작업을 병렬로 처리하기 위함입니다.
# HOW: 텐서의 numpy_async() 코루틴을 await하여 비동기 대기합니다.
async def to_numpy_async(x: Tensor) -> np.ndarray:
    return await x.numpy_async()


# WHAT: 텐서의 자원(메모리 등)을 수동으로 해제하는 함수입니다.
# WHY: 파이썬 가비지 컬렉터를 기다리지 않고 즉시 GPU 메모리나 대용량 CPU 메모리를 반환하여 OOM을 방지하기 위함입니다.
# HOW: 텐서 내부의 dispose() 메서드를 호출합니다.
def dispose(x: Tensor) -> None:
    x.dispose()


# WHAT: 두 텐서의 형상이 주어졌을 때, 넘파이 스타일의 브로드캐스팅 규칙이 적용된 결과 형상을 계산하는 유틸리티 함수입니다.
# WHY: 크기가 다른 배열 간의 연산(예: 덧셈, 곱셈)이 가능한지 확인하고, 최종 결과물의 크기를 미리 알기 위함입니다.
# HOW: 두 형상의 길이를 맞추기 위해 왼쪽에 1을 패딩한 후, 각 차원별로 크기를 비교하여 호환되면 큰 값을 선택합니다.
def _broadcast_shapes(a_shape, b_shape):
    # WHAT: 두 형상 중 더 긴 길이를 구합니다.
    # WHY: 짧은 쪽의 형상을 긴 쪽에 맞춰 패딩하기 위함입니다.
    # HOW: max 함수를 사용합니다.
    ndim = max(len(a_shape), len(b_shape))
    
    # WHAT: 왼쪽 차원에 1을 채워넣어 길이를 맞춥니다.
    # WHY: 스칼라 또는 차원이 적은 텐서를 높은 차원의 텐서와 비교하기 위함입니다.
    # HOW: (1,) 튜플을 부족한 만큼 곱해서 이어붙입니다.
    a_padded = (1,) * (ndim - len(a_shape)) + a_shape
    b_padded = (1,) * (ndim - len(b_shape)) + b_shape
    
    result = []
    # WHAT: 패딩된 형상의 각 차원을 순회하며 브로드캐스트 가능 여부를 확인합니다.
    # WHY: 브로드캐스팅 규칙(차원이 같거나 어느 한쪽이 1이어야 함)을 검증하기 위함입니다.
    # HOW: zip을 이용해 요소별로 비교합니다.
    for ad, bd in zip(a_padded, b_padded):
        if ad == bd:
            result.append(ad)
        elif ad == 1:
            result.append(bd)
        elif bd == 1:
            result.append(ad)
        else:
            raise AMEVAForgeShapeError(f"Cannot broadcast {a_shape} and {b_shape}")
    return tuple(result)


# WHAT: 브로드캐스팅으로 인해 확장되었던 그래디언트 텐서를 원래 크기로 되돌리는(축소하는) 함수입니다.
# WHY: 역전파 시 각 입력 파라미터는 자신의 원래 모양과 똑같은 크기의 미분값을 받아야 하기 때문입니다.
# HOW: 대상 형상과 비교하여 1이었던 차원은 합산(sum)을 수행하고 필요 없는 차원은 제거합니다.
def _unbroadcast(grad, target_shape):
    if grad.shape == target_shape:
        # WHAT: 형상이 동일하면 그대로 반환합니다.
        # WHY: 추가적인 연산 낭비를 막기 위함입니다.
        # HOW: 조건문 검사 후 곧바로 grad를 리턴합니다.
        return grad
        
    # WHAT: 확장된 차원들의 위치(인덱스)를 찾아냅니다.
    # WHY: 어느 축(axis)을 기준으로 합산(sum)하여 원래 모양으로 찌그러뜨릴지 알아내기 위함입니다.
    # HOW: shape 길이를 비교해 ndim_diff를 구하고 1로 패딩해 비교합니다.
    ndim_diff = len(grad.shape) - len(target_shape)
    padded = (1,) * ndim_diff + target_shape
    
    if grad.device == 'cpu':
        # WHAT: CPU 환경에서의 언브로드캐스트 처리입니다.
        # WHY: 넘파이는 여러 축을 한 번에 합산할 수 있는 기능이 있기 때문입니다.
        # HOW: 축(axes)을 리스트에 모아 np.sum(axis=...)를 호출한 뒤 원래 차원으로 복구(reshape)합니다.
        data = _require_cpu_data(grad, 'grad')
        axes = []
        for i, (g, t) in enumerate(zip(grad.shape, padded)):
            if t == 1 and g != 1:
                axes.append(i)
        for i in range(ndim_diff):
            axes.append(i) if i not in axes else None
        axes = sorted(set(axes))
        
        result = np.sum(data, axis=tuple(axes), keepdims=True)
        result = result.reshape(target_shape)
        return Tensor(shape=target_shape, dtype='float32', device='cpu', data=result)
    else:
        # WHAT: GPU 환경에서의 언브로드캐스트 처리입니다.
        # WHY: GPU 커널 함수는 대개 한 번에 하나의 축만 감소(reduction)시키는 함수를 제공하기 때문입니다.
        # HOW: 차이 나는 축들을 뒤에서부터 순차적으로 sum_axis 함수를 호출하여 하나씩 줄여나갑니다.
        result = grad
        axes = []
        for i, (g, t) in enumerate(zip(grad.shape, padded)):
            if t == 1 and g != 1:
                axes.append(i)
        for i in range(ndim_diff):
            if i not in axes:
                axes.append(i)
        axes = sorted(set(axes))
        
        # WHAT: 인덱스 변화를 막기 위해 뒤에서부터 합산합니다.
        # WHY: 앞쪽 차원을 먼저 합치면 뒤쪽 축들의 인덱스가 하나씩 밀려 오류가 발생하기 때문입니다.
        # HOW: reversed(axes)를 순회하며 sum_axis를 호출합니다.
        for ax in reversed(axes):
            result = sum_axis(result, axis=ax)
            
        return reshape(result, target_shape) if result.shape != target_shape else result

# WHAT: 모든 요소의 합(Sum)을 구하는 축소 연산(Reduction) 클래스입니다.
# WHY: 손실값 누적 계산이나 정규화 등 텐서의 전체 합이 필요할 때 사용하기 위함입니다.
# HOW: 순전파에서는 배열의 모든 요소를 더해 스칼라 모양의 텐서를 만들고, 역전파에서는 동일한 기울기를 모든 위치에 분배합니다.
class SumFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # WHAT: 원래 입력 텐서의 모양을 저장합니다.
        # WHY: 합산 후 스칼라로 크기가 줄어들기 때문에, 역전파 시 원래 모양으로 복구(브로드캐스트)하기 위함입니다.
        # HOW: ctx 객체 속성으로 x.shape를 할당합니다.
        ctx.input_shape = x.shape
        if _should_use_gpu(x):
            return Tensor(shape=(), dtype='float32', device='gpu', op='sum', parents=(x,))
        else:
            return Tensor(shape=(), dtype='float32', device='cpu',
                         data=np.array(np.sum(_require_cpu_data(x, 'x')), dtype=np.float32))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # WHAT: 합산의 미분은 모든 원소에 대해 1이므로, 역전파된 기울기를 입력과 같은 크기로 뿌려줍니다.
        # WHY: f = x1 + x2 + ... 이면 df/dxi = 1 이기 때문입니다.
        # HOW: ones 배열을 생성하여 grad_output과 곱합니다.
        return (mul(grad_output, ones(x.shape, device=x.device)),)

# WHAT: 합산 연산 편의 함수입니다.
# WHY: 외부에서 텐서 전체 요소의 합을 손쉽게 계산하기 위함입니다.
# HOW: SumFunction.apply를 실행합니다.
def sum_op(x): return SumFunction.apply(x)


# WHAT: 모든 요소의 평균(Mean)을 구하는 축소 연산 클래스입니다.
# WHY: 손실 함수(MSE 등)나 배치 정규화 등에서 요소들의 평균값이 필요하기 때문입니다.
# HOW: 요소 개수를 구하고 numpy mean을 사용하거나, 합산을 개수로 나눕니다.
class MeanFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # WHAT: 텐서 안의 전체 원소 개수를 계산합니다.
        # WHY: GPU 등에서 평균을 '합산 / 개수'로 대체하기 위함과 역전파 시 나누기 위해 필요하기 때문입니다.
        # HOW: shape의 각 차원 크기를 곱해나갑니다.
        n = 1
        for d in x.shape:
            n *= d
        ctx.numel = n
        
        if _should_use_gpu(x):
            # WHAT: GPU에서는 no_grad() 격리를 통해 임시 Autograd 노드 생성을 차단하고 sum과 div를 조합합니다.
            # WHY: 중복된 커널 구현을 피하고 Autograd 컨텍스트 오염을 원천 차단하기 위함입니다.
            with no_grad():
                s = sum_op(x)
                res = div(s, tensor(np.array([float(n)], dtype=np.float32), device=x.device))
            return res
        else:
            return Tensor(shape=(), dtype='float32', device='cpu',
                         data=np.array(np.mean(_require_cpu_data(x, 'x')), dtype=np.float32))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        n = ctx.numel
        # WHAT: 평균의 역전파로, 흘러들어온 기울기를 원소 개수 N으로 나누어 모든 위치에 분배합니다.
        # WHY: 평균 식 f = (x1+...+xn)/N 의 미분값은 1/N 이기 때문입니다.
        # HOW: 1.0/n 값으로 채워진 텐서를 만들고 grad_output에 곱합니다.
        return (mul(grad_output, full(x.shape, 1.0/n, device=x.device)),)

# WHAT: 평균 연산 편의 함수입니다.
# WHY: 외부에서 텐서 전체 평균을 직관적으로 호출하기 위함입니다.
# HOW: MeanFunction.apply를 실행합니다.
def mean_op(x): return MeanFunction.apply(x)


# WHAT: 지수(Exponential) 연산을 수행하는 클래스입니다.
# WHY: 소프트맥스(Softmax)나 활성화 함수에서 자연상수 e의 거듭제곱을 계산하기 위함입니다.
# HOW: np.exp를 수행하고, 미분 시 자신(exp(x))을 다시 곱합니다.
class ExpFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if _should_use_gpu(x):
            result = Tensor(shape=x.shape, dtype='float32', device='gpu', op='exp', parents=(x,))
        else:
            result = Tensor(shape=x.shape, dtype='float32', device='cpu',
                          data=np.exp(_require_cpu_data(x, 'x')))
        # WHAT: 연산 결과(exp(x))를 역전파용으로 저장합니다.
        # WHY: 지수 함수의 미분은 자기 자신과 같으므로(d(e^x)/dx = e^x), 원본 x 대신 결과값을 저장해 연산량을 줄이기 위함입니다.
        # HOW: ctx에 result를 저장합니다.
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        exp_x, = ctx.saved_tensors
        # WHAT: 지수 함수의 역전파입니다.
        # WHY: 체인 룰에 의해 흘러들어온 기울기에 exp(x)를 곱해야 하기 때문입니다.
        # HOW: 저장해둔 출력 텐서 exp_x와 grad_output을 곱합니다.
        return (mul(grad_output, exp_x),)

# WHAT: 지수 함수 편의 호출 인터페이스입니다.
# WHY: 외부에서 텐서의 exp 연산을 수행할 수 있게 하기 위함입니다.
# HOW: ExpFunction.apply를 실행합니다.
def exp_op(x): return ExpFunction.apply(x)


# WHAT: 자연로그(Logarithm) 연산을 수행하는 클래스입니다.
# WHY: 크로스 엔트로피 손실 등에서 확률 분포의 정보량을 수치적으로 다루기 위함입니다.
# HOW: np.log를 사용해 계산하고, 미분 시 1/x을 적용합니다.
class LogFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if _should_use_gpu(x):
            return Tensor(shape=x.shape, dtype='float32', device='gpu', op='log', parents=(x,))
        else:
            data = _require_cpu_data(x, 'x')
            # VUL-004: debug mode에서만 non-positive 입력 경고
            if _debug_mode and np.any(data <= 0):
                import warnings
                warnings.warn(
                    "[AMEVA debug] log() received non-positive input. "
                    "Result contains -inf/nan. Use stabilized log_softmax in loss functions.",
                    RuntimeWarning, stacklevel=4
                )
            return Tensor(shape=x.shape, dtype='float32', device='cpu',
                         data=np.log(data))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # WHAT: 자연로그의 역전파 연산입니다.
        # WHY: log(x)의 도함수가 1/x 이기 때문입니다.
        # HOW: grad_output을 원본 입력 텐서 x로 나눕니다.
        return (div(grad_output, x),)

# WHAT: 자연로그 편의 함수입니다.
# WHY: 직관적인 호출을 제공하기 위함입니다.
# HOW: LogFunction.apply를 실행합니다.
def log_op(x): return LogFunction.apply(x)


# WHAT: 시그모이드(Sigmoid) 활성화 함수를 구현한 연산 클래스입니다.
# WHY: 텐서의 값을 0과 1 사이로 압축하여 이진 분류의 확률이나 게이트(gate) 값으로 변환하기 위함입니다.
# HOW: 1 / (1 + exp(-x)) 공식을 적용합니다.
class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if _should_use_gpu(x):
            result = Tensor(shape=x.shape, dtype='float32', device='gpu', op='sigmoid', parents=(x,))
        else:
            # WHAT: CPU 환경에서의 시그모이드 계산 수식입니다.
            # WHY: x의 값을 정규화하기 위함입니다.
            # HOW: np.exp를 이용해 수식을 전개합니다.
            data = 1.0 / (1.0 + np.exp(-_require_cpu_data(x, 'x')))
            result = Tensor(shape=x.shape, dtype='float32', device='cpu', data=data)
            
        # WHAT: 계산된 시그모이드 결과값을 컨텍스트에 저장합니다.
        # WHY: 도함수 계산 시 x 원본보다 시그모이드 출력값을 이용하는 것이 연산에 효율적이기 때문입니다.
        # HOW: result 자체를 저장합니다.
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        sig, = ctx.saved_tensors
        # WHAT: 시그모이드 함수의 미분식인 sig * (1 - sig)를 계산합니다.
        # WHY: f(x) = sigmoid(x) 일 때 df/dx = f(x)(1 - f(x)) 이기 때문입니다.
        # HOW: 1 - sig 텐서를 생성한 후 sig와 곱하고, 마지막으로 흘러온 기울기(grad_output)를 곱합니다.
        one_minus = sub(ones(sig.shape, device=sig.device), sig)
        return (mul(grad_output, mul(sig, one_minus)),)

# WHAT: 시그모이드 편의 함수입니다.
# WHY: 외부에서 텐서의 시그모이드를 쉽게 계산하기 위함입니다.
# HOW: SigmoidFunction.apply를 실행합니다.
def sigmoid(x): return SigmoidFunction.apply(x)


# WHAT: 하이퍼볼릭 탄젠트(Tanh) 활성화 함수 연산 클래스입니다.
# WHY: 텐서의 값을 -1과 1 사이로 매핑하여 신경망 은닉층의 비선형성을 제공하기 위함입니다.
# HOW: 순전파 시 np.tanh를 사용하고 역전파 시 1 - tanh^2 수식을 적용합니다.
class TanhFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if _should_use_gpu(x):
            result = Tensor(shape=x.shape, dtype='float32', device='gpu', op='tanh', parents=(x,))
        else:
            result = Tensor(shape=x.shape, dtype='float32', device='cpu',
                          data=np.tanh(_require_cpu_data(x, 'x')))
        # WHAT: tanh 결과를 컨텍스트에 저장합니다.
        # WHY: 미분 공식에서 원본 x보다 결과값을 활용(1 - tanh(x)^2)하는 편이 효율적이기 때문입니다.
        # HOW: ctx에 result를 저장합니다.
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        tanh_x, = ctx.saved_tensors
        # WHAT: 하이퍼볼릭 탄젠트의 역전파 연산입니다.
        # WHY: tanh'(x) = 1 - tanh^2(x) 이기 때문입니다.
        # HOW: 1로 채워진 텐서(ones)에서 tanh_x 제곱을 빼고 기울기를 곱합니다.
        one_minus_sq = sub(ones(tanh_x.shape, device=tanh_x.device), mul(tanh_x, tanh_x))
        return (mul(grad_output, one_minus_sq),)

# WHAT: 하이퍼볼릭 탄젠트 편의 함수입니다.
# WHY: 외부에서 쉽게 호출하기 위함입니다.
# HOW: TanhFunction.apply를 실행합니다.
def tanh_op(x): return TanhFunction.apply(x)


# WHAT: 텐서의 모양(Shape)을 변경하는 연산 클래스입니다.
# WHY: 메모리 내 데이터 순서를 유지한 채 차원 구조만 바꿔 호환성을 맞추기 위함입니다.
# HOW: numpy reshape를 사용하거나 GPU의 경우 메타데이터 갱신 명령(op='reshape')을 보냅니다.
class ReshapeFunction(Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        # WHAT: 원래 차원 형태를 저장합니다.
        # WHY: 역전파 시 그래디언트를 원래 모양으로 되돌려 보내야 하기 때문입니다.
        # HOW: 컨텍스트 속성에 x.shape를 기록합니다.
        ctx.original_shape = x.shape
        if x.device == 'gpu':
            # WHAT: GPU 기반 리쉐이프(Reshape) 처리입니다.
            # WHY: VRAM 내 데이터 이동 없이 메타데이터만 갱신해 비용을 최소화하기 위함입니다.
            # HOW: op_params로 새로운 모양을 전달합니다.
            return Tensor(shape=new_shape, dtype=x.dtype, device='gpu', op='reshape', parents=(x,),
                         op_params=list(new_shape))
        else:
            data = _require_cpu_data(x, 'x').reshape(new_shape)
            return Tensor(shape=new_shape, dtype='float32', device='cpu', data=data)
    
    @staticmethod
    def backward(ctx, grad_output):
        # WHAT: 모양 변경의 역전파입니다.
        # WHY: 미분값들도 연산 이전의 노드 형태와 같아야 하므로 원래 모양으로 되돌립니다.
        # HOW: reshape 유틸리티 함수를 다시 호출합니다.
        return (reshape(grad_output, ctx.original_shape),)

# WHAT: 리쉐이프 연산 편의 함수입니다.
# WHY: 리스트 등의 인자를 튜플로 정규화한 뒤 클래스에 전달하기 위함입니다.
# HOW: new_shape가 리스트면 튜플로 변환 후 ReshapeFunction.apply를 호출합니다.
def reshape(x, new_shape):
    if isinstance(new_shape, list):
        new_shape = tuple(new_shape)
    return ReshapeFunction.apply(x, new_shape)


# WHAT: 특정 축(Axis)을 따라 요소들의 합을 구하는 축소 연산 클래스입니다.
# WHY: 다차원 텐서에서 특정 차원(예: 배치 차원이나 클래스 차원)을 기준으로 합계를 계산하기 위함입니다.
# HOW: 넘파이 sum 연산에 axis 인자를 사용하거나, GPU 커널 계산용 매개변수(stride 등)를 도출하여 수행합니다.
class SumAxisFunction(Function):
    @staticmethod
    def forward(ctx, x, axis):
        ctx.save_for_backward(x)
        # WHAT: 대상 축과 원래 형태를 저장합니다. 음수 축(-1 등)은 양수 랭크 인덱스로 정규화합니다.
        # WHY: 역전파 시 줄어든 차원을 다시 복구(unsqueeze)하고 VRAM 스트라이드를 정확히 계산하기 위함입니다.
        # HOW: norm_axis = axis if axis >= 0 else axis + rank
        rank = len(x.shape)
        norm_axis = axis if axis >= 0 else axis + rank
        if norm_axis < 0 or norm_axis >= rank:
            raise AMEVAForgeShapeError(f"Invalid axis {axis} for tensor with rank {rank}")

        ctx.axis = norm_axis
        ctx.input_shape = x.shape
        
        if x.device == 'gpu':
            # WHAT: 결과 텐서의 모양(shape)을 계산합니다.
            # WHY: 축소된 차원이 제거된 새로운 shape 튜플을 만들기 위함입니다.
            # HOW: 리스트 변환 후 해당 인덱스를 지우고 다시 튜플로 만듭니다.
            new_shape = list(x.shape)
            del new_shape[norm_axis]
            new_shape = tuple(new_shape) if new_shape else ()
            
            # WHAT: GPU 커널이 사용할 다차원 합산 파라미터(stride)를 계산합니다.
            # WHY: 1차원 선형 배열 형태인 VRAM 데이터를 특정 차원 간격으로 순회하며 합쳐야 하기 때문입니다.
            # HOW: 축 바깥쪽 크기(outer_size)와 축 안쪽 간격(inner_stride)을 구하여 op_params로 넘깁니다.
            outer_size = 1
            for i in range(norm_axis):
                outer_size *= x.shape[i]
            reduction_size = x.shape[norm_axis]
            inner_stride = 1
            for i in range(norm_axis + 1, rank):
                inner_stride *= x.shape[i]
                
            return Tensor(shape=new_shape, dtype='float32', device='gpu',
                         op='sum_axis', parents=(x,), op_params=[outer_size, reduction_size, inner_stride])
        else:
            data = np.sum(_require_cpu_data(x, 'x'), axis=norm_axis)
            return Tensor(shape=data.shape, dtype='float32', device='cpu', data=data)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if grad_output.device == 'cpu':
            # WHAT: CPU 경로에서 합산의 역전파를 수행합니다.
            # WHY: 줄어든 차원으로 들어온 미분값을 원래 크기의 텐서에 복사(타일링)하기 위함입니다.
            # HOW: 축 자리에 1을 끼워넣고(expand_dims) 전체 모양에 맞춰 브로드캐스팅(broadcast_to)합니다.
            data = _require_cpu_data(grad_output, 'grad')
            expanded = np.expand_dims(data, axis=ctx.axis)
            tiled = np.broadcast_to(expanded, ctx.input_shape).copy()
            return (Tensor(shape=ctx.input_shape, dtype='float32', device='cpu', data=tiled),)
        else:
            # WHAT: VUL-013 Fix 적용된 GPU 경로의 역전파 처리입니다.
            # WHY: GPU는 넘파이 브로드캐스팅 메서드가 없으므로, 명시적으로 reshape와 ones 곱셈을 사용해야 하기 때문입니다.
            # HOW: 원래 축 위치에 1을 삽입한 뒤, ones로 채워진 텐서와 곱해 브로드캐스트 효과를 냅니다.
            axis = ctx.axis
            input_shape = ctx.input_shape
            expand_shape = list(input_shape)
            expand_shape[axis] = 1
            grad_reshaped = reshape(grad_output, tuple(expand_shape))
            broadcast_ones = ones(input_shape, device='gpu')
            return (mul(broadcast_ones, grad_reshaped),)

# WHAT: 축 기반 합산 연산 편의 함수입니다.
# WHY: 외부에서 axis 지정과 함께 합산을 편리하게 수행하기 위함입니다.
# HOW: SumAxisFunction.apply를 호출합니다. 기본 축은 0입니다.
def sum_axis(x, axis=0): return SumAxisFunction.apply(x, axis)

# WHAT: 표준 정규 분포(평균 0, 표준편차 1)를 따르는 난수 텐서를 생성합니다.
# WHY: 신경망 가중치 초기화 등에서 널리 쓰이는 정규 분포 데이터가 필요하기 때문입니다.
# HOW: np.random.randn을 호출한 후 지정된 디바이스에 맞게 Tensor 객체를 반환합니다.
def randn(shape, device='cpu', dtype='float32', requires_grad=False):
    # WHAT: 난수로 채워진 numpy 배열입니다.
    # WHY: 텐서의 백엔드 데이터로 사용하기 위함입니다.
    # HOW: 언패킹된 shape를 인자로 넘겨 난수를 생성합니다.
    arr = np.random.randn(*shape).astype(np.float32)
    if device == 'cpu':
        return Tensor(shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, data=arr)
    else:
        return Tensor(shape=shape, dtype=dtype, device='gpu', requires_grad=requires_grad, data=arr, op='upload')

# WHAT: 텐서의 특정 위치(dim)에 크기가 1인 새로운 차원을 삽입합니다.
# WHY: 배치 차원을 추가하거나 브로드캐스팅 조건을 맞추기 위함입니다.
# HOW: 형상을 리스트로 변환해 1을 삽입(insert)한 후 reshape 함수를 호출합니다.
def unsqueeze(x: Tensor, dim: int) -> Tensor:
    shape = list(x.shape)
    # WHAT: 음수 인덱스 처리입니다.
    # WHY: 뒤에서부터 차원을 지정하는 파이썬 관례를 지원하기 위함입니다.
    # HOW: -1이면 맨 뒤에 추가되도록 길이에 1을 더해 보정합니다.
    if dim < 0:
        dim += len(shape) + 1
    shape.insert(dim, 1)
    return reshape(x, tuple(shape))


# WHAT: 텐서에서 크기가 1인 차원을 제거합니다.
# WHY: 불필요한 차원을 줄여 원래 데이터 형태로 다루기 위함입니다.
# HOW: dim이 지정되면 해당 차원이 1일 때만 제거하고, 없으면 모든 크기 1 차원을 날린 후 reshape합니다.
def squeeze(x: Tensor, dim: Optional[int] = None) -> Tensor:
    shape = list(x.shape)
    if dim is not None:
        if dim < 0:
            dim += len(shape)
        # WHAT: 지정된 차원이 1인지 확인합니다.
        # WHY: 크기가 1이 아닌 차원을 제거하면 데이터 크기가 달라져 오류가 발생하기 때문입니다.
        # HOW: 조건문 검사 후 pop을 수행합니다.
        if shape[dim] == 1:
            shape.pop(dim)
    else:
        # WHAT: 리스트 컴프리헨션으로 크기가 1인 모든 차원을 걸러냅니다.
        # WHY: 특정 차원 지정 없이 전부 압축하기 위함입니다.
        # HOW: s != 1 조건만 모아 새로운 리스트를 만듭니다.
        shape = [s for s in shape if s != 1]
    return reshape(x, tuple(shape))

# WHAT: 텐서의 특정 차원 범위를 하나의 1차원으로 평탄화(Flatten)합니다.
# WHY: 합성곱층(CNN)의 4차원 출력을 완전연결층(Linear)의 2차원 입력으로 넘길 때 등 형상을 펴야 하기 때문입니다.
# HOW: 시작과 끝 차원 사이의 크기를 전부 곱하여 단일 차원으로 합친 뒤 reshape 합니다.
def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    shape = list(x.shape)
    if end_dim < 0:
        end_dim += len(shape)
    if start_dim < 0:
        start_dim += len(shape)
    if start_dim > end_dim:
        return x
        
    new_shape = shape[:start_dim]
    # WHAT: 합쳐질 차원들의 요소 개수를 누적할 변수입니다.
    # WHY: 여러 차원의 크기를 모두 곱해야 평탄화된 크기를 알 수 있기 때문입니다.
    # HOW: 순회를 돌며 곱셉 누적을 합니다.
    flat_size = 1
    for s in shape[start_dim:end_dim+1]:
        flat_size *= s
    new_shape.append(flat_size)
    new_shape.extend(shape[end_dim+1:])
    return reshape(x, tuple(new_shape))

# WHAT: 텐서의 차원 순서를 재배치(Permute)하는 연산 클래스입니다.
# WHY: (N, C, H, W)를 (N, H, W, C)로 바꾸는 등 이미지나 채널 차원을 다룰 때 필요하기 때문입니다.
# HOW: 주어진 차원 인덱스 순서대로 np.transpose를 수행하며, 역전파 시에는 역순열(inverse permutation)을 적용합니다.
class PermuteFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, dims: tuple) -> Tensor:
        rank = len(x.shape)
        if len(dims) != rank:
            raise AMEVAForgeShapeError(f"permute dims {dims} length does not match tensor rank {rank}")
            
        # WHAT: 음수 차원 인덱스를 양수로 정규화(Normalize)합니다.
        # WHY: 사용자 편의를 위해 파이썬스러운 음수 인덱싱을 지원하기 위함입니다.
        # HOW: 음수면 rank를 더해 양수 범위 안착을 확인합니다.
        normalized_dims = []
        for d in dims:
            if not isinstance(d, int):
                raise TypeError(f"permute dims must be integers, got {type(d)}")
            nd = d + rank if d < 0 else d
            if nd < 0 or nd >= rank:
                raise IndexError(f"Dimension out of range (expected to be in range of [-{rank}, {rank-1}], but got {d})")
            normalized_dims.append(nd)
            
        normalized_dims = tuple(normalized_dims)
        if len(set(normalized_dims)) != rank:
            raise ValueError(f"permute dims {dims} contains duplicate dimensions")
            
        ctx.save_for_backward(x)
        # WHAT: 정규화된 차원 순서를 컨텍스트에 저장합니다.
        # WHY: 역전파 시 이 순서를 뒤집어주는 행렬을 만들어야 하기 때문입니다.
        # HOW: ctx.dims 속성으로 기록합니다.
        ctx.dims = normalized_dims
        
        # WHAT: 새로운 형상을 계산합니다.
        # WHY: 리턴될 텐서의 shape 정보를 구성하기 위함입니다.
        # HOW: 제너레이터 표현식으로 원래 shape의 요소들을 새 순서대로 가져옵니다.
        out_shape = tuple(x.shape[i] for i in normalized_dims)
        
        if x.device == 'gpu':
            return Tensor(
                shape=out_shape,
                dtype=x.dtype,
                device='gpu',
                requires_grad=False,
                op='permute',
                parents=(x,),
                op_params=list(normalized_dims)
            )
            
        data = _require_cpu_data(x, "x")
        res = np.transpose(data, normalized_dims)
        return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        (x,) = ctx.saved_tensors
        # WHAT: 역순열(Inverse Permutation) 인덱스를 계산합니다.
        # WHY: 섞인 차원을 다시 원래대로 돌려놓아야 이전 노드로 기울기가 올바르게 가기 때문입니다.
        # HOW: 값이 위치 인덱스가 되고 인덱스가 값이 되는 배열 inv_dims를 생성합니다.
        inv_dims = [0] * len(ctx.dims)
        for i, d in enumerate(ctx.dims):
            inv_dims[d] = i
        return (permute(grad_output, tuple(inv_dims)),)


# WHAT: 순열 재배치 편의 함수입니다.
# WHY: 외부에서 쉽게 호출하기 위함입니다.
# HOW: PermuteFunction.apply를 실행합니다.
def permute(x: Tensor, dims: tuple) -> Tensor:
    return PermuteFunction.apply(x, dims)

# WHAT: 텐서 내의 모든 원소 중 최댓값(Max)을 찾아 스칼라로 반환하는 클래스입니다.
# WHY: 통계 추출이나 소프트맥스의 수치적 안정성을 위해 가장 큰 값이 필요하기 때문입니다.
# HOW: 순전파는 np.max를 쓰고 역전파는 최댓값이 있었던 위치에만 그래디언트를 줍니다.
class MaxFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        data = _require_cpu_data(x, "x")
        # WHAT: 전체 요소 중 가장 큰 값입니다.
        # WHY: 역전파 때 이 값이 위치했던 곳을 찾아야 하기 때문입니다.
        # HOW: np.max 결과값을 컨텍스트에 저장합니다.
        m = np.max(data)
        ctx.max_val = m
        return Tensor(shape=(), dtype="float32", device="cpu", data=np.array(m, dtype=np.float32))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        data = _require_cpu_data(x, "x")
        grad = _require_cpu_data(grad_output, "grad")
        
        # WHAT: 최댓값이 존재했던 위치를 나타내는 불리언 마스크 배열입니다.
        # WHY: 미분값은 최댓값을 배출한 원본 노드에게만 전달되어야 하기 때문입니다(max 함수의 특징).
        # HOW: data == ctx.max_val 비교를 통해 1과 0으로 만듭니다.
        mask = (data == ctx.max_val).astype(np.float32)
        sum_mask = np.sum(mask)
        # WHAT: 중복된 최댓값이 있을 경우 기울기를 나눠 갖는 처리입니다.
        # WHY: 특정 값이 여러 번 나타나면 전체 미분합이 커지는 것을 막기 위함입니다.
        # HOW: 마스크 합계로 나눠 평균화시킵니다.
        if sum_mask > 0:
            mask = mask / sum_mask
            
        res_grad = grad * mask
        return (Tensor(shape=x.shape, dtype="float32", device="cpu", data=res_grad),)

# WHAT: 전역 최댓값 편의 함수입니다.
# WHY: 사용자가 쉽게 텐서 전체 최댓값을 구하게 하기 위함입니다.
# HOW: MaxFunction.apply를 호출합니다.
def max_op(x: Tensor) -> Tensor:
    return MaxFunction.apply(x)

# WHAT: 특정 축을 기준으로 최댓값을 구하는 축소 연산 클래스입니다.
# WHY: 클래스별 확률 계산(예: Max Pooling, Softmax 최댓값 제거) 등에서 특정 차원의 대표값을 뽑기 위함입니다.
# HOW: np.max(axis=...)를 사용하고, 미분 시 해당 위치를 찾아 마스크를 생성합니다.
class MaxAxisFunction(Function):
    @staticmethod
    def forward(ctx, x, axis):
        ctx.save_for_backward(x)
        rank = len(x.shape)
        norm_axis = axis if axis >= 0 else axis + rank
        if norm_axis < 0 or norm_axis >= rank:
            raise AMEVAForgeShapeError(f"Invalid axis {axis} for tensor with rank {rank}")

        ctx.axis = norm_axis
        ctx.input_shape = x.shape
        
        if x.device == 'gpu':
            new_shape = list(x.shape)
            del new_shape[norm_axis]
            new_shape = tuple(new_shape) if new_shape else ()
            
            outer_size = 1
            for i in range(norm_axis):
                outer_size *= x.shape[i]
            reduction_size = x.shape[norm_axis]
            inner_stride = 1
            for i in range(norm_axis + 1, rank):
                inner_stride *= x.shape[i]
                
            return Tensor(shape=new_shape, dtype='float32', device='gpu',
                         op='max_axis', parents=(x,), op_params=[outer_size, reduction_size, inner_stride])
        else:
            data = _require_cpu_data(x, "x")
            m = np.max(data, axis=norm_axis)
            ctx.max_val = m
            return Tensor(shape=m.shape, dtype="float32", device="cpu", data=m)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        axis = ctx.axis
        rank = len(x.shape)
        
        if x.device == "gpu":
            outer_size = 1
            for i in range(axis):
                outer_size *= x.shape[i]
            reduction_size = x.shape[axis]
            inner_stride = 1
            for i in range(axis + 1, rank):
                inner_stride *= x.shape[i]
                
            return (
                Tensor(
                    shape=x.shape,
                    dtype="float32",
                    device="gpu",
                    op="max_axis_backward",
                    parents=(x, grad_output),
                    op_params=[outer_size, reduction_size, inner_stride],
                ),
            )
        
        data = _require_cpu_data(x, "x")
        grad = _require_cpu_data(grad_output, "grad")
        
        m_exp = np.expand_dims(ctx.max_val, axis=ctx.axis) if hasattr(ctx, 'max_val') else np.expand_dims(np.max(data, axis=ctx.axis), axis=ctx.axis)
        grad_exp = np.expand_dims(grad, axis=ctx.axis)
        
        mask = (data == m_exp).astype(np.float32)
        sum_mask = np.sum(mask, axis=ctx.axis, keepdims=True)
        mask = np.divide(mask, sum_mask, out=np.zeros_like(mask), where=sum_mask != 0)
        
        res_grad = grad_exp * mask
        return (Tensor(shape=x.shape, dtype="float32", device="cpu", data=res_grad),)

# WHAT: 특정 축 기반 최댓값 편의 함수입니다.
# WHY: 외부에서 쉽게 호출하게 하기 위함입니다.
# HOW: MaxAxisFunction.apply를 호출합니다.
def max_axis(x: Tensor, axis: int) -> Tensor:
    return MaxAxisFunction.apply(x, axis)

# WHAT: 특정 축 기반 평균값 계산 함수입니다.
# WHY: 배치 정규화 등에서 차원별 평균을 구하기 위함입니다.
# HOW: sum_axis 결과를 해당 축의 요소 개수로 나눕니다.
def mean_axis(x: Tensor, axis: int) -> Tensor:
    s = sum_axis(x, axis)
    n = x.shape[axis]
    return div(s, tensor(np.array(n, dtype=np.float32), device=x.device))

# WHAT: 분산(Variance)을 계산하는 함수입니다.
# WHY: 데이터의 산포도를 구하여 표준화나 통계 분석에 사용하기 위함입니다.
# HOW: E[(x - E[x])^2] 공식을 따르며, 표본 분산(unbiased) 여부에 따라 자유도를 조절합니다.
def var(x: Tensor, axis=None, unbiased=True) -> Tensor:
    if axis is None:
        # WHAT: 텐서 전체에 대한 분산 계산입니다.
        # WHY: 축 지정이 없으면 모든 요소의 분산을 구하기 때문입니다.
        # HOW: 전체 평균(mean_op)을 빼고 제곱한 뒤 전체를 합산하고 나눕니다.
        m = mean_op(x)
        diff = sub(x, m)
        diff_sq = mul(diff, diff)
        s = sum_op(diff_sq)
        n = x.numel()
        denom = n - 1 if unbiased and n > 1 else n
        return div(s, tensor(np.array(denom, dtype=np.float32), device=x.device))
    else:
        # WHAT: 특정 축에 대한 분산 계산입니다.
        # WHY: 특징(Feature)별 혹은 채널별 분산을 구하기 위함입니다.
        # HOW: mean_axis 결과를 원래 차원으로 unsqueeze한 뒤 계산합니다.
        m = mean_axis(x, axis)
        m_unsq = unsqueeze(m, axis)
        diff = sub(x, m_unsq)
        diff_sq = mul(diff, diff)
        s = sum_axis(diff_sq, axis)
        n = x.shape[axis]
        denom = n - 1 if unbiased and n > 1 else n
        return div(s, tensor(np.array(denom, dtype=np.float32), device=x.device))

# WHAT: 제곱근(Square Root) 연산을 지원하는 자동 미분 클래스입니다.
# WHY: 표준편차 계산이나 RMSProp과 같은 옵티마이저 등에서 수치 안정성을 도모하기 위함입니다.
# HOW: 순전파 시 np.sqrt를 적용하고, 미분 시 1 / (2 * sqrt(x)) 공식을 적용합니다.
class SqrtFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if _should_use_gpu(x):
            # GPU sqrt via mathematical identity: exp(0.5 * log(x))
            half = full(x.shape, 0.5, device='gpu')
            return exp_op(mul(log_op(x), half))
        else:
            ctx.save_for_backward(x)
            data = _require_cpu_data(x, "x")
            res = np.sqrt(data)
            return Tensor(shape=x.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        two = full(x.shape, 2.0, device=x.device)
        return (div(grad_output, mul(two, sqrt(x))),)

# WHAT: 제곱근 연산 편의 함수입니다.
# WHY: 텐서의 제곱근을 쉽게 구하기 위함입니다.
# HOW: SqrtFunction.apply를 실행합니다.
def sqrt(x: Tensor) -> Tensor:
    return SqrtFunction.apply(x)

# WHAT: 표준편차(Standard Deviation) 계산 함수입니다.
# WHY: 데이터의 흩어짐을 원래 단위로 파악하기 위함입니다.
# HOW: 분산(var) 함수의 결과에 제곱근(sqrt)을 씌워 리턴합니다.
def std(x: Tensor, axis=None, unbiased=True) -> Tensor:
    return sqrt(var(x, axis=axis, unbiased=unbiased))

# WHAT: 두 텐서를 특정 축(dim)을 기준으로 병합(Concatenate)하는 연산 클래스입니다.
# WHY: 여러 배치 데이터나 특징 맵을 하나로 이어붙이기 위함입니다.
# HOW: 순전파는 np.concatenate를 쓰고, 역전파는 병합된 기울기를 다시 slice하여 나눠 줍니다.
class CatFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor, dim: int = 0) -> Tensor:
        ctx.save_for_backward(a, b)
        ctx.dim = dim
        _ensure_same_device(a, b, "cat")
        
        shape_a = list(a.shape)
        shape_b = list(b.shape)
        # WHAT: 음수 차원(dim)에 대한 정규화입니다.
        # WHY: 맨 뒤 차원을 -1로 지정할 수 있게 지원하기 위함입니다.
        # HOW: 음수면 배열 길이를 더합니다.
        if dim < 0:
            dim += len(shape_a)
            
        out_shape = list(shape_a)
        out_shape[dim] += shape_b[dim]
        out_shape = tuple(out_shape)
        
        if _should_use_gpu(a, b):
            stride = 1
            for s in shape_a[dim+1:]:
                stride *= s
            return Tensor(shape=out_shape, dtype="float32", device="gpu",
                          op='cat', parents=(a, b), op_params=[int(shape_a[dim]), int(shape_b[dim]), stride])
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            res = np.concatenate((data_a, data_b), axis=dim)
            return Tensor(shape=out_shape, dtype="float32", device="cpu", data=res)
            
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        dim = ctx.dim
        
        if grad_output.device == 'gpu':
            raise AMEVAForgeDeviceError(
                "Cat backward on GPU tensors is not supported in Release 1. "
                "Execute model on CPU or transfer tensors to CPU before backward."
            )
        g = _require_cpu_data(grad_output, 'grad_output')
            
        # WHAT: 병합되었던 차원을 원래 a와 b의 크기로 쪼개기 위한 슬라이스 객체 생성입니다.
        # WHY: 역전파 시 각 입력 크기만큼 기울기를 나눠 주어야 하기 때문입니다.
        # HOW: slice()를 사용하여 a 부분과 b 부분 인덱스를 정의합니다.
        slc_a = [slice(None)] * len(g.shape)
        slc_a[dim] = slice(0, a.shape[dim])
        slc_b = [slice(None)] * len(g.shape)
        slc_b[dim] = slice(a.shape[dim], None)
        
        ga = g[tuple(slc_a)]
        gb = g[tuple(slc_b)]
        
        # WHAT: 분할된 배열을 메모리 연속(contiguous) 배열로 만듭니다.
        # WHY: 슬라이싱된 뷰(view)가 C 기반의 다음 연산에서 에러를 뿜지 않게 하기 위함입니다.
        # HOW: np.ascontiguousarray 함수를 통과시킵니다.
        ga = np.ascontiguousarray(ga)
        gb = np.ascontiguousarray(gb)
        
        return (Tensor(shape=a.shape, dtype="float32", device="cpu", data=ga),
                Tensor(shape=b.shape, dtype="float32", device="cpu", data=gb))

# WHAT: 리스트에 담긴 텐서들을 순차적으로 병합(Cat)하는 편의 함수입니다.
# WHY: 두 개뿐만 아니라 N개의 텐서를 쉽게 합치기 위함입니다.
# HOW: 리스트를 순회하며 CatFunction.apply를 누적 호출(reduce)합니다.
def cat(tensors: list, dim: int = 0) -> Tensor:
    if len(tensors) < 1:
        raise ValueError("cat requires at least 1 tensor")
    if len(tensors) == 1:
        return tensors[0]
    res = tensors[0]
    for t in tensors[1:]:
        res = CatFunction.apply(res, t, dim)
    return res

# WHAT: 조건 텐서에 따라 참이면 x, 거짓이면 y 요소를 선택하는 연산.
# WHY: 마스킹된 데이터 추출이나 조건부 활성화 함수(LeakyReLU 등)를 간결하게 구현하기 위함입니다.
# HOW: 순전파는 np.where를, 역전파도 where를 재사용해 조건에 맞게 기울기를 분배합니다.
class WhereFunction(Function):
    @staticmethod
    def forward(ctx: Context, condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(condition, x, y)
        if condition.device != x.device or x.device != y.device:
            raise AMEVAForgeDeviceError("where requires all tensors to be on the same device")
        
        out_shape = x.shape
        if _should_use_gpu(x, y):
            return Tensor(shape=out_shape, dtype="float32", device="gpu", op='where', parents=(condition, x, y))
        else:
            c = _require_cpu_data(condition, "condition")
            data_x = _require_cpu_data(x, "x")
            data_y = _require_cpu_data(y, "y")
            # WHAT: c > 0 조건에 따라 요소를 선택합니다.
            # WHY: condition이 불리언(1, 0)을 담은 float 텐서이기 때문입니다.
            # HOW: np.where 함수를 씁니다.
            res = np.where(c > 0, data_x, data_y)
            return Tensor(shape=out_shape, dtype="float32", device="cpu", data=res)
            
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, Tensor, Tensor]:
        condition, x, y = ctx.saved_tensors
        # WHAT: condition 텐서로는 역전파를 하지 않으므로 None, x와 y는 마스크에 따라 0 또는 grad_output을 받습니다.
        # WHY: 선택받은 쪽만 기울기를 가져가고 선택받지 못한 쪽의 기울기는 0이기 때문입니다.
        # HOW: zero_grad 텐서를 만들고 재귀적으로 where 연산을 호출합니다.
        zero_grad = zeros_like(grad_output)
        grad_x = where(condition, grad_output, zero_grad)
        grad_y = where(condition, zero_grad, grad_output)
        return (None, grad_x, grad_y)

# WHAT: 조건부 선택 편의 함수입니다.
# WHY: 외부에서 텐서 마스킹을 손쉽게 수행하기 위함입니다.
# HOW: WhereFunction.apply를 호출합니다.
def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return WhereFunction.apply(condition, x, y)

# WHAT: 텐서의 바깥 테두리에 특정 값(기본 0)을 덧대는(Padding) 패딩 연산 클래스입니다.
# WHY: 합성곱층(Conv2d) 연산 시 이미지 가장자리의 정보를 보존하거나 크기를 유지하기 위함입니다.
# HOW: 파이토치와 유사하게 pad 튜플을 받아 앞뒤(좌우/상하)로 패딩을 삽입합니다.
class PadFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, pad: Tuple[int, ...], mode='constant', value=0.0) -> Tensor:
        ctx.save_for_backward(x)
        ctx.pad = pad
        ctx.mode = mode
        ctx.value = value
        
        out_shape = list(x.shape)
        rank = len(x.shape)
        # WHAT: 각 차원별 패딩 크기를 앞뒤(before, after) 쌍으로 계산합니다.
        # WHY: 뒤에서부터 차례대로(pad 튜플이 우측 차원부터 명시됨) 적용해야 하기 때문입니다.
        # HOW: 역순으로 pad 배열을 읽어 pad_pairs 리스트 앞쪽에 insert합니다.
        pad_pairs = []
        for i in range(rank):
            pad_before = pad[-(i * 2 + 2)] if len(pad) >= (i * 2 + 2) else 0
            pad_after = pad[-(i * 2 + 1)] if len(pad) >= (i * 2 + 1) else 0
            pad_pairs.insert(0, (pad_before, pad_after))
            out_shape[i] += pad_before + pad_after
            
        out_shape = tuple(out_shape)
        if _should_use_gpu(x):
            # WHAT: GPU 커널이 다차원 배열을 계산할 수 있도록 보폭(strides)을 구합니다.
            # WHY: VRAM은 1차원이므로 다차원 인덱스를 선형 오프셋으로 변환해야 하기 때문입니다.
            # HOW: 차원들을 누적 곱하여 stride를 도출하는 헬퍼를 씁니다.
            def get_strides(s):
                st = [1]*len(s)
                for i in range(len(s)-2, -1, -1):
                    st[i] = st[i+1]*s[i+1]
                return st
            in_strides = get_strides(x.shape)
            out_strides = get_strides(out_shape)
            pad_before_arr = [p[0] for p in pad_pairs]
            
            # WHAT: GPU에 보낼 op_params 메타데이터 배열을 조립합니다.
            # WHY: C++ 기반 백엔드가 구조체 없이 정수/실수 배열만으로 파라미터를 파싱하기 때문입니다.
            # HOW: 리스트 평탄화를 수행하고 8차원 고정 크기로 0을 패딩해 맞춥니다.
            op_params = [
                0, rank, value, 0,
                *(in_strides + [0]*(8-rank)),
                *(out_strides + [0]*(8-rank)),
                *(pad_before_arr + [0]*(8-rank)),
                *(list(x.shape) + [0]*(8-rank))
            ]
            return Tensor(shape=out_shape, dtype=x.dtype, device='gpu', op='pad', parents=(x,), op_params=op_params)
        else:
            data = _require_cpu_data(x, "x")
            res = np.pad(data, pad_pairs, mode=mode, constant_values=value)
            return Tensor(shape=out_shape, dtype=x.dtype, device='cpu', data=res)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        if grad_output.device == 'gpu':
            # WHAT: GPU에서 패딩 역전파 시도 시 에러입니다.
            # WHY: 패딩의 미분은 잘려나가는 부분(슬라이스)인데, GPU 슬라이스 커널이 없기 때문입니다.
            # HOW: 강제로 에러를 냅니다.
            from .errors import AMEVAForgeDeviceError
            raise AMEVAForgeDeviceError("GPU pad backward requires a native slice kernel.")
            
        # WHAT: 패딩된 부분은 기울기가 0이 되므로 중앙 원본 영역의 미분값만 잘라(slice)옵니다.
        # WHY: 패딩은 상수로 추가된 값이므로 입력 데이터에 대한 미분(기여)이 없기 때문입니다.
        # HOW: pad_before부터 원래 shape 크기만큼 슬라이싱합니다.
        slices = []
        rank = len(x.shape)
        for i in range(rank):
            pad_before = ctx.pad[-(i * 2 + 2)] if len(ctx.pad) >= (i * 2 + 2) else 0
            slc = slice(pad_before, pad_before + x.shape[i])
            slices.append(slc)
        return (grad_output[tuple(slices)],)

# WHAT: 패딩 편의 함수입니다.
# WHY: 외부에서 쉽게 호출하도록 하기 위함입니다.
# HOW: PadFunction.apply를 호출합니다.
def pad(x: Tensor, pad: Tuple[int, ...], mode='constant', value=0.0) -> Tensor:
    return PadFunction.apply(x, pad, mode, value)

# WHAT: 인덱스 텐서를 기반으로 특정 축에서 데이터를 수집(Gather)하는 클래스입니다.
# WHY: 임베딩 룩업(Embedding Lookup)이나 특정 위치의 값들을 모아 새로운 텐서를 만들기 위함입니다.
# HOW: CPU는 np.take_along_axis를, GPU는 메타데이터와 stride를 계산해 커널 매개변수로 전달합니다.
class GatherFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, dim: int, index: Tensor) -> Tensor:
        ctx.save_for_backward(x, index)
        ctx.dim = dim
        _ensure_same_device(x, index, "gather")
        if _should_use_gpu(x, index):
            # WHAT: 다차원 인덱싱을 위한 각 텐서의 보폭(strides) 계산입니다.
            # WHY: GPU 메모리는 1차원 선형 공간이므로 몇 칸을 건너뛰어야 다음 차원으로 가는지 알아야 하기 때문입니다.
            # HOW: get_strides 헬퍼 함수를 이용해 계산합니다.
            def get_strides(s):
                st = [1]*len(s)
                for i in range(len(s)-2, -1, -1):
                    st[i] = st[i+1]*s[i+1]
                return st
            x_strides = get_strides(x.shape)
            out_strides = get_strides(index.shape)
            rank = len(x.shape)
            
            numel = 1
            for d in index.shape:
                numel *= d
            op_params = [
                numel, dim, rank, 0,
                *(x_strides + [0]*(8-rank)),
                *(out_strides + [0]*(8-rank)),
                *(list(x.shape) + [0]*(8-rank))
            ]
            return Tensor(shape=index.shape, dtype=x.dtype, device='gpu', op='gather', parents=(x, index), op_params=op_params)
        else:
            data = _require_cpu_data(x, "x")
            idx = _require_cpu_data(index, "index").astype(int)
            res = np.take_along_axis(data, idx, axis=dim)
            return Tensor(shape=index.shape, dtype=x.dtype, device='cpu', data=res)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        x, index = ctx.saved_tensors
        if x.device == 'gpu':
            raise AMEVAForgeUnsupportedOperationError(
                "GPU gather backward requires atomic scatter_add for duplicate index correctness. "
                "Release 1 supports GPU gather forward, but duplicate-safe GPU backward is planned for Release 2. "
                "Transfer tensor to CPU before backward if gather gradient calculation is required."
            )
        grad_x = scatter(zeros_like(x), ctx.dim, index, grad_output)
        return (grad_x, None)

# WHAT: 수집(Gather) 편의 함수입니다.
# WHY: 쉽게 임베딩이나 데이터 추출 연산을 호출하기 위함입니다.
# HOW: GatherFunction.apply를 실행합니다.
def gather(x: Tensor, dim: int, index: Tensor) -> Tensor:
    return GatherFunction.apply(x, dim, index)

# WHAT: 지정된 인덱스 위치에 소스 텐서의 값을 뿌려주는(Scatter) 연산 클래스입니다.
# WHY: 역전파나 원핫 인코딩(One-hot encoding) 등 특정 텐서 위치에 값을 삽입/업데이트할 때 필요하기 때문입니다.
# HOW: 순전파는 np.put_along_axis를 이용하고 역전파는 gather를 활용해 다시 뽑아옵니다.
class ScatterFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
        ctx.save_for_backward(x, index, src)
        ctx.dim = dim
        _ensure_same_device(x, index, "scatter")
        _ensure_same_device(x, src, "scatter")
        if _should_use_gpu(x, index) and src.device == 'gpu':
            def get_strides(s):
                st = [1]*len(s)
                for i in range(len(s)-2, -1, -1):
                    st[i] = st[i+1]*s[i+1]
                return st
            x_strides = get_strides(x.shape)
            idx_strides = get_strides(index.shape)
            rank = len(x.shape)
            numel = 1
            for d in index.shape: numel *= d
            op_params = [
                numel, dim, rank, 0,
                *(x_strides + [0]*(8-rank)),
                *(idx_strides + [0]*(8-rank))
            ]
            # WHAT: 세 개의 부모(parents)를 가지는 텐서를 생성합니다.
            # WHY: 업데이트할 원본(x), 위치 정보(index), 삽입할 값(src)이 모두 연산 그래프에 추적되어야 하기 때문입니다.
            # HOW: parents=(index, src, x)로 인자를 전달합니다.
            return Tensor(shape=x.shape, dtype=x.dtype, device='gpu', op='scatter', parents=(index, src, x), op_params=op_params)
        else:
            data = _require_cpu_data(x, "x").copy()
            idx = _require_cpu_data(index, "index").astype(int)
            src_data = _require_cpu_data(src, "src")
            # WHAT: 지정된 축과 인덱스를 기반으로 값을 덮어씁니다.
            # WHY: 스캐터 연산의 목적을 달성하기 위함입니다.
            # HOW: np.put_along_axis를 활용해 원본을 변형합니다.
            np.put_along_axis(data, idx, src_data, axis=dim)
            return Tensor(shape=x.shape, dtype=x.dtype, device='cpu', data=data)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, Tensor]:
        x, index, src = ctx.saved_tensors
        # WHAT: 산포(Scatter)의 역전파입니다. src와 x 양방향으로 기울기를 나눕니다.
        # WHY: 덮어씌워진 위치는 src가 기울기를 온전히 가져가고, 나머지 위치는 원래 x가 기울기를 가져가야 하기 때문입니다.
        # HOW: src 쪽은 gather로 가져오고, x 쪽은 scatter로 0을 덮어씌워 해당 위치의 기울기를 제거합니다.
        grad_src = gather(grad_output, ctx.dim, index)
        grad_x = scatter(grad_output, ctx.dim, index, zeros_like(src))
        return (grad_x, None, grad_src)

# WHAT: 산포(Scatter) 편의 함수입니다.
# WHY: 외부에서 텐서의 특정 위치를 쉽게 업데이트하기 위함입니다.
# HOW: ScatterFunction.apply를 호출합니다. Release 1 GPU는 assign semantics를 기본으로 지원합니다.
def scatter(x: Tensor, dim: int, index: Tensor, src: Tensor, reduce: str = "assign") -> Tensor:
    if x.device == "gpu" and reduce != "assign":
        raise AMEVAForgeUnsupportedOperationError(
            "GPU scatter with reduction is not supported in Release 1. Use assign semantics."
        )
    return ScatterFunction.apply(x, dim, index, src)

# WHAT: 텐서 슬라이싱(Slicing) 연산을 지원하는 클래스입니다.
# WHY: 파이썬의 대괄호 인덱싱(t[0:5])을 통해 텐서의 일부 영역만 추출하거나 미분을 추적하기 위함입니다.
# HOW: 순전파는 numpy 슬라이싱을 래핑하고, 역전파는 슬라이싱된 위치에만 np.add.at을 통해 미분값을 누적(scatter-add)합니다.
class SliceFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, key) -> Tensor:
        ctx.save_for_backward(x)
        import numpy as np
        
        # WHAT: 슬라이싱 키(key)를 복사하여 컨텍스트에 저장합니다.
        # WHY: 역전파 시 정확히 같은 위치에 미분값을 더해주어야 하기 때문입니다.
        # HOW: key의 타입(배열, 튜플 등)에 따라 깊은 복사를 수행합니다.
        if isinstance(key, np.ndarray):
            ctx.key = key.copy()
        elif isinstance(key, tuple):
            ctx.key = tuple(k.copy() if isinstance(k, np.ndarray) else k for k in key)
        else:
            ctx.key = key
            
        if x.device == 'gpu':
            # WHAT: GPU 기반 슬라이싱 에러 처리입니다.
            # WHY: 현재 GPU 커널에는 복잡한 다차원 슬라이싱 로직이 포팅되어 있지 않기 때문입니다.
            # HOW: AMEVAForgeDeviceError를 발생시킵니다.
            raise AMEVAForgeDeviceError(
                "GPU slicing is not implemented yet. "
                "A native GPU slicing kernel is required."
            )
            
        data = _require_cpu_data(x, "x")
        # WHAT: 데이터를 슬라이싱합니다.
        # WHY: 사용자가 요청한 범위의 배열을 얻기 위함입니다.
        # HOW: numpy 인덱싱을 그대로 활용합니다.
        res = data[key]
        res_array = np.asarray(res, dtype=np.float32)
        
        return Tensor(shape=res_array.shape, dtype=x.dtype, device='cpu', data=res_array)
        
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        if x.device == 'gpu':
            raise AMEVAForgeDeviceError(
                "GPU slicing backward is not implemented yet. "
                "A native GPU scatter-add kernel is required."
            )
        
        import numpy as np
        # WHAT: 원본 크기와 동일한 영(0) 텐서를 생성합니다.
        # WHY: 역전파된 부분 기울기(grad_data)를 원래 슬라이싱 위치에만 채워넣기 위함입니다.
        # HOW: np.zeros를 사용합니다.
        grad_x = np.zeros(x.shape, dtype=np.float32)
        grad_data = _require_cpu_data(grad_output, "grad_output")
        
        try:
            # WHAT: 역전파된 기울기를 누적합(Scatter-Add) 방식으로 반영합니다.
            # WHY: 슬라이싱된 뷰가 중복된 인덱스를 가질 경우(예: 팬시 인덱싱) 일반 대입(grad_x[key] = grad_data)을 쓰면 덮어씌워지기 때문입니다.
            # HOW: np.add.at을 사용하여 값을 차곡차곡 더합니다.
            np.add.at(grad_x, ctx.key, grad_data)
        except (IndexError, TypeError, ValueError) as exc:
            raise AMEVAForgeShapeError(f"Slice backward failed for key {ctx.key!r}: {exc}") from exc
            
        return (Tensor(shape=x.shape, dtype=x.dtype, device='cpu', data=grad_x),)

# WHAT: 텐서 슬라이싱 편의 함수입니다.
# WHY: Tensor.__getitem__에서 백엔드로 호출하기 위함입니다.
# HOW: SliceFunction.apply를 실행합니다.
def slice_op(x: Tensor, key) -> Tensor:
    return SliceFunction.apply(x, key)

# WHAT: 2차원 합성곱(Convolution 2D) 연산 클래스입니다.
# WHY: 이미지 등 공간 정보를 가진 텐서에 필터(커널)를 적용하여 특징 맵을 추출하기 위함입니다.
# HOW: CPU는 im2col 방식을 루프로 구현하고, GPU는 im2col 커널 후 matmul을 조합하여 계산합니다.
class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, weight: Tensor, bias: Optional[Tensor], stride: int, padding: int) -> Tensor:
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        
        N, C, H, W = x.shape
        C_out, C_in, K_h, K_w = weight.shape
        if C != C_in:
            raise AMEVAForgeShapeError(f"Input channels {C} does not match weight channels {C_in}")
            
        # WHAT: 합성곱 이후의 출력 형태(Height, Width)를 계산합니다.
        # WHY: 패딩과 스트라이드를 고려한 출력 텐서를 생성하기 위함입니다.
        # HOW: 일반적인 Conv2D 출력 크기 공식을 적용합니다.
        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1
        
        if x.device == "gpu" and (x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)):
            raise AMEVAForgeUnsupportedOperationError(
                "GPU Conv2d backward is not supported in Release 1. "
                "Use CPU Conv2d training or mark tensors requires_grad=False for GPU inference."
            )
        
        if _should_use_gpu(x, weight):
            # WHAT: GPU 경로에서 이미지를 열(Column)로 전개하는 im2col 연산을 수행합니다.
            # WHY: 합성곱 연산을 행렬 곱셈(Matmul)으로 치환하여 GPU 병렬 처리 효율을 극대화하기 위함입니다.
            # HOW: op="im2col"로 텐서를 띄웁니다.
            x_col = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                           op="im2col", parents=(x,), op_params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            
            # WHAT: 4차원 가중치 텐서를 2차원으로 평탄화(reshape)하고 전치(permute)합니다.
            # WHY: x_col과의 행렬 곱셈을 맞추기 위함입니다.
            # HOW: reshape 후 permute를 호출합니다.
            weight_reshaped = weight.reshape((C_out, C * K_h * K_w))
            weight_t = permute(weight_reshaped, (1, 0))
            
            out_2d = Tensor(shape=(N * H_out * W_out, C_out), dtype=x.dtype, device="gpu", requires_grad=False,
                            op="matmul", parents=(x_col, weight_t), op_params=[N * H_out * W_out, C_out, C * K_h * K_w])
                            
            out = permute(out_2d.reshape((N, H_out, W_out, C_out)), (0, 3, 1, 2))
            if bias is not None:
                bias_reshaped = bias.reshape((1, C_out, 1, 1))
                out = out + bias_reshaped
            return out
        else:
            # WHAT: CPU 경로에서 im2col 전개를 루프로 수행합니다.
            # WHY: GPU 커널이 없을 때 Numpy만으로 합성곱을 계산해야 하기 때문입니다.
            # HOW: 다중 루프를 돌며 patch를 추출해 x_col에 할당합니다.
            x_data = _require_cpu_data(x)
            weight_data = _require_cpu_data(weight)
            
            x_col = np.zeros((N, H_out * W_out, C * K_h * K_w), dtype=np.float32)
            for n in range(N):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride - padding
                        w_start = w_out * stride - padding
                        patch = np.zeros((C, K_h, K_w), dtype=np.float32)
                        for c in range(C):
                            for k_h in range(K_h):
                                for k_w in range(K_w):
                                    h_in = h_start + k_h
                                    w_in = w_start + k_w
                                    if 0 <= h_in < H and 0 <= w_in < W:
                                        patch[c, k_h, k_w] = x_data[n, c, h_in, w_in]
                        x_col[n, h_out * W_out + w_out, :] = patch.flatten()
            
            weight_reshaped = weight_data.reshape((C_out, C * K_h * K_w))
            out_data = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
            # WHAT: 전개된 x_col 행렬과 가중치 행렬을 곱합니다.
            # WHY: 특징 맵(Feature Map)을 생성하기 위함입니다.
            # HOW: 넘파이 행렬 곱(@)을 수행하고 원래 이미지 모양으로 복원합니다.
            for n in range(N):
                out_2d = x_col[n] @ weight_reshaped.T
                out_data[n] = out_2d.T.reshape((C_out, H_out, W_out))
                
            if bias is not None:
                bias_data = _require_cpu_data(bias)
                out_data += bias_data.reshape((1, C_out, 1, 1))
                
            return Tensor(shape=(N, C_out, H_out, W_out), dtype=x.dtype, device="cpu", requires_grad=False, data=out_data)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        
        N, C, H, W = x.shape
        C_out, C_in, K_h, K_w = weight.shape
        H_out = grad_output.shape[2]
        W_out = grad_output.shape[3]
        
        grad_bias = None
        # WHAT: 편향(bias) 텐서의 기울기를 계산합니다.
        # WHY: 편향은 모든 공간 픽셀과 배치에 대해 동일하게 더해졌으므로, 흘러온 기울기를 채널(C_out)만 남기고 싹 다 더해야 하기 때문입니다.
        # HOW: 축 3, 2, 0 차례대로 sum_axis를 호출합니다.
        if bias is not None and bias.requires_grad:
            g = sum_axis(grad_output, 3)
            g = sum_axis(g, 2)
            g = sum_axis(g, 0)
            grad_bias = g.reshape(bias.shape)
            
        if _should_use_gpu(x, weight):
            # WHAT: GPU 기반 합성곱 역전파 처리입니다.
            # WHY: 가중치 미분과 입력 미분을 행렬 연산과 col2im 연산으로 가속하기 위함입니다.
            # HOW: grad_output을 2차원으로 눌러서 matmul 후, col2im 커널을 통해 이미지를 복원시킵니다.
            grad_out_2d = permute(grad_output, (0, 2, 3, 1)).reshape((N * H_out * W_out, C_out))
            x_col = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                           op="im2col", parents=(x,), op_params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            
            x_col_t = permute(x_col, (1, 0))
            grad_weight_2d = Tensor(shape=(C * K_h * K_w, C_out), dtype=x.dtype, device="gpu", requires_grad=False,
                                    op="matmul", parents=(x_col_t, grad_out_2d), op_params=[C * K_h * K_w, C_out, N * H_out * W_out])
            grad_weight = permute(grad_weight_2d, (1, 0)).reshape(weight.shape)
            
            weight_reshaped = weight.reshape((C_out, C * K_h * K_w))
            grad_x_col_2d = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                                   op="matmul", parents=(grad_out_2d, weight_reshaped), op_params=[N * H_out * W_out, C * K_h * K_w, C_out])
            
            grad_x = Tensor(shape=(N, C, H, W), dtype=x.dtype, device="gpu", requires_grad=False,
                            op="col2im", parents=(grad_x_col_2d,), op_params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            
            return grad_x, grad_weight, grad_bias
        else:
            # WHAT: CPU 기반 합성곱 역전파 처리입니다.
            # WHY: GPU 가속을 사용할 수 없을 때 넘파이만으로 가중치와 입력의 기울기를 구하기 위함입니다.
            # HOW: 앞서 순전파와 동일하게 입력을 im2col로 전개하고, 루프를 돌며 grad_out과 행렬 곱을 수행합니다.
            x_data = _require_cpu_data(x)
            weight_data = _require_cpu_data(weight)
            grad_out_data = _require_cpu_data(grad_output)
            
            x_col = np.zeros((N, H_out * W_out, C * K_h * K_w), dtype=np.float32)
            # WHAT: CPU im2col 재계산 루프입니다.
            # WHY: x_col을 역전파용으로 저장하지 않았기 때문에 메모리 절약을 위해 여기서 다시 계산합니다.
            # HOW: 배치(N), 세로(H_out), 가로(W_out)를 순회하며 패치(patch)를 추출합니다.
            for n in range(N):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride - padding
                        w_start = w_out * stride - padding
                        patch = np.zeros((C, K_h, K_w), dtype=np.float32)
                        for c in range(C):
                            for k_h in range(K_h):
                                for k_w in range(K_w):
                                    h_in = h_start + k_h
                                    w_in = w_start + k_w
                                    if 0 <= h_in < H and 0 <= w_in < W:
                                        patch[c, k_h, k_w] = x_data[n, c, h_in, w_in]
                        x_col[n, h_out * W_out + w_out, :] = patch.flatten()
            
            grad_weight_data = np.zeros_like(weight_data)
            grad_x_data = np.zeros_like(x_data)
            weight_reshaped = weight_data.reshape((C_out, C * K_h * K_w))
            
            # WHAT: 출력의 기울기를 행렬 곱셈을 위해 평탄화(reshape)합니다.
            # WHY: im2col 형태의 입력과 내적(dot product)하여 가중치와 입력 기울기를 도출하기 위함입니다.
            # HOW: 전치(transpose) 후 reshape 합니다.
            grad_out_2d = grad_out_data.transpose(0, 2, 3, 1).reshape(N, H_out * W_out, C_out)
            
            grad_x_col = np.zeros_like(x_col)
            for n in range(N):
                # WHAT: 가중치(weight)에 대한 기울기를 누적 계산합니다.
                # WHY: dL/dW = x^T * dL/dY 공식을 따릅니다.
                # HOW: x_col 전치행렬과 grad_out_2d를 곱해 원래 가중치 모양으로 더합니다.
                gw = x_col[n].T @ grad_out_2d[n]
                grad_weight_data += gw.T.reshape(weight.shape)
                
                # WHAT: 입력(x) 텐서의 열(col) 형태 기울기를 계산합니다.
                # WHY: dL/dX_col = dL/dY * W 공식을 따릅니다.
                # HOW: grad_out_2d와 weight_reshaped를 곱합니다.
                gxc = grad_out_2d[n] @ weight_reshaped
                grad_x_col[n] = gxc
                
                # WHAT: col2im 과정을 수동으로 루프를 돌며 수행합니다.
                # WHY: 평탄화되었던 기울기를 다시 2차원 공간 좌표계(C, H, W)로 누적합하기 위함입니다.
                # HOW: grad_x_col의 각 패치를 원래 이미지 인덱스 위치(h_in, w_in)에 더합니다.
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        patch = grad_x_col[n, h_out * W_out + w_out].reshape(C, K_h, K_w)
                        h_start = h_out * stride - padding
                        w_start = w_out * stride - padding
                        for c in range(C):
                            for k_h in range(K_h):
                                for k_w in range(K_w):
                                    h_in = h_start + k_h
                                    w_in = w_start + k_w
                                    if 0 <= h_in < H and 0 <= w_in < W:
                                        grad_x_data[n, c, h_in, w_in] += patch[c, k_h, k_w]
                                        
            grad_x = Tensor(shape=x.shape, dtype=x.dtype, device="cpu", requires_grad=False, data=grad_x_data)
            grad_weight = Tensor(shape=weight.shape, dtype=weight.dtype, device="cpu", requires_grad=False, data=grad_weight_data)
            
            return grad_x, grad_weight, grad_bias

# WHAT: Conv2D 연산 편의 함수입니다.
# WHY: 사용자가 nn 모듈 등에서 텐서에 2D 합성곱을 손쉽게 호출하기 위함입니다.
# HOW: Conv2dFunction.apply를 실행합니다.
def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1, padding: int = 0) -> Tensor:
    return Conv2dFunction.apply(x, weight, bias, stride, padding)


# WHAT: 2차원 공간 영역에서의 최대 풀링(Max Pooling 2D) 연산 클래스입니다.
# WHY: 합성곱 신경망에서 특징 맵의 해상도를 줄이면서 중요한 특징(최댓값)만 남기기 위함입니다.
# HOW: 커널 크기만큼 패치를 이동하며 최대값을 찾고(forward), 역전파 시 해당 위치에만 기울기를 넘깁니다(backward).
class MaxPool2dFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, kernel_size, stride=None, padding=0):
        # WHAT: 스트라이드가 없을 경우 커널 사이즈와 동일하게 맞춥니다.
        # WHY: 기본적으로 풀링 영역이 겹치지 않게(non-overlapping) 이동하도록 하기 위함입니다.
        # HOW: None 체크 후 덮어씁니다.
        if stride is None: stride = kernel_size
        ctx.save_for_backward(x)
        ctx.kH = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        ctx.kW = kernel_size[1] if isinstance(kernel_size, (list, tuple)) else kernel_size
        ctx.sH = stride[0] if isinstance(stride, (list, tuple)) else stride
        ctx.sW = stride[1] if isinstance(stride, (list, tuple)) else stride
        ctx.pH = padding[0] if isinstance(padding, (list, tuple)) else padding
        ctx.pW = padding[1] if isinstance(padding, (list, tuple)) else padding
        
        B, C, in_h, in_w = x.shape
        out_h = (in_h + 2 * ctx.pH - ctx.kH) // ctx.sH + 1
        out_w = (in_w + 2 * ctx.pW - ctx.kW) // ctx.sW + 1
        
        if x.device == 'gpu' and x.requires_grad:
            raise AMEVAForgeUnsupportedOperationError(
                "GPU MaxPool2d backward is not supported in Release 1. "
                "GPU pooling is inference-only in this release."
            )
        
        if x.device == 'gpu':
            # WHAT: GPU 기반 풀링 처리입니다.
            # WHY: 풀링 연산을 커널로 위임해 속도를 높이기 위함입니다.
            # HOW: op='maxpool2d'로 텐서를 띄우고 필수 파라미터들을 op_params에 넣습니다.
            op_params = [B, C, in_h, in_w, out_h, out_w, ctx.kH, ctx.kW, ctx.sH, ctx.sW, ctx.pH, ctx.pW]
            return Tensor(shape=(B, C, out_h, out_w), dtype='float32', device='gpu', op='maxpool2d', parents=(x,), op_params=op_params)
        else:
            # WHAT: CPU 기반 풀링 계산입니다.
            # WHY: 넘파이를 활용해 수동으로 풀링 결과를 계산하기 위함입니다.
            # HOW: -inf로 가장자리 패딩을 넣고, 루프를 돌며 np.max를 뽑아냅니다.
            data = _require_cpu_data(x, "x")
            padded = np.pad(data, ((0,0), (0,0), (ctx.pH, ctx.pH), (ctx.pW, ctx.pW)), constant_values=-np.inf)
            out = np.zeros((B, C, out_h, out_w), dtype=np.float32)
            for h in range(out_h):
                for w in range(out_w):
                    h_start, w_start = h * ctx.sH, w * ctx.sW
                    out[:, :, h, w] = np.max(padded[:, :, h_start:h_start+ctx.kH, w_start:w_start+ctx.kW], axis=(2, 3))
            return Tensor(shape=(B, C, out_h, out_w), dtype='float32', device='cpu', data=out)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        if grad_output.device == 'gpu':
            raise AMEVAForgeDeviceError(
                "MaxPool2d backward is not supported on GPU tensors in the synchronous autograd engine in Release 1. "
                "Execute model on CPU or use Release 1 GPU-supported operators."
            )
        x, = ctx.saved_tensors
        grad_out_np = _require_cpu_data(grad_output, 'grad_output')
        x_np = _require_cpu_data(x, 'x')
        B, C, in_h, in_w = x_np.shape
        out_h = (in_h + 2 * ctx.pH - ctx.kH) // ctx.sH + 1
        out_w = (in_w + 2 * ctx.pW - ctx.kW) // ctx.sW + 1
        
        # WHAT: 패딩이 반영된 원본 입력 모양의 영행렬(기울기 누적용)을 만듭니다.
        # WHY: 원본 텐서가 패딩되었을 때 위치를 맞추기 위해서입니다.
        # HOW: np.pad로 패딩한 뒤 np.zeros_like로 형태를 본뜹니다.
        padded = np.pad(x_np, ((0,0), (0,0), (ctx.pH, ctx.pH), (ctx.pW, ctx.pW)), constant_values=-np.inf)
        grad_padded = np.zeros_like(padded)
        
        # WHAT: 최대값이 있던 위치(마스크)에만 기울기를 부여합니다.
        # WHY: 맥스 풀링은 미분 시 값을 배출한 원래 뉴런에게만 그 책임을 묻기 때문입니다.
        # HOW: (window == max_val) 조건으로 불리언 마스크를 만들고 평균(중복 시)한 뒤 grad_out_np를 더합니다.
        for b in range(B):
            for c in range(C):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start, w_start = h * ctx.sH, w * ctx.sW
                        window = padded[b, c, h_start:h_start+ctx.kH, w_start:w_start+ctx.kW]
                        max_val = np.max(window)
                        mask = (window == max_val)
                        sum_mask = np.sum(mask)
                        if sum_mask > 0:
                            mask = mask / sum_mask
                        grad_padded[b, c, h_start:h_start+ctx.kH, w_start:w_start+ctx.kW] += mask * grad_out_np[b, c, h, w]
                        
        # WHAT: 테두리 패딩 부분으로 넘어간 기울기를 잘라내 버립니다.
        # WHY: 원래 x 텐서의 실제 데이터 영역이 아니기 때문입니다.
        # HOW: 파이썬 슬라이싱 문법(pH:-pH)을 이용해 중앙값만 취합니다.
        if ctx.pH > 0 or ctx.pW > 0:
            grad_x_np = grad_padded[:, :, ctx.pH:-ctx.pH if ctx.pH > 0 else None, ctx.pW:-ctx.pW if ctx.pW > 0 else None]
        else:
            grad_x_np = grad_padded
            
        if x.device == 'gpu':
            return (tensor(grad_x_np, device='gpu'),)
        else:
            return (Tensor(shape=x.shape, dtype='float32', device='cpu', data=grad_x_np),)

def max_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    return MaxPool2dFunction.apply(x, kernel_size, stride, padding)

# WHAT: 2차원 공간 영역에서의 평균 풀링(Average Pooling 2D) 연산 클래스입니다.
# WHY: 특징 맵의 해상도를 낮추면서 각 패치의 평균을 취해 전반적인(Global/Local) 특성을 요약하기 위함입니다.
# HOW: 맥스 풀링과 유사하게 윈도우를 슬라이딩하되 np.sum 연산 후 넓이(kH * kW)로 나눕니다. 역전파 시에는 미분값을 똑같이 분배합니다.
class AvgPool2dFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, kernel_size, stride=None, padding=0):
        if stride is None: stride = kernel_size
        ctx.save_for_backward(x)
        ctx.kH = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        ctx.kW = kernel_size[1] if isinstance(kernel_size, (list, tuple)) else kernel_size
        ctx.sH = stride[0] if isinstance(stride, (list, tuple)) else stride
        ctx.sW = stride[1] if isinstance(stride, (list, tuple)) else stride
        ctx.pH = padding[0] if isinstance(padding, (list, tuple)) else padding
        ctx.pW = padding[1] if isinstance(padding, (list, tuple)) else padding
        
        B, C, in_h, in_w = x.shape
        out_h = (in_h + 2 * ctx.pH - ctx.kH) // ctx.sH + 1
        out_w = (in_w + 2 * ctx.pW - ctx.kW) // ctx.sW + 1
        
        if x.device == 'gpu' and x.requires_grad:
            raise AMEVAForgeUnsupportedOperationError(
                "GPU AvgPool2d backward is not supported in Release 1. "
                "GPU pooling is inference-only in this release."
            )
        
        if x.device == 'gpu':
            op_params = [B, C, in_h, in_w, out_h, out_w, ctx.kH, ctx.kW, ctx.sH, ctx.sW, ctx.pH, ctx.pW]
            return Tensor(shape=(B, C, out_h, out_w), dtype='float32', device='gpu', op='avgpool2d', parents=(x,), op_params=op_params)
        else:
            data = _require_cpu_data(x, "x")
            # WHAT: 평균 풀링에서는 -inf가 아닌 0으로 패딩을 채웁니다.
            # WHY: 평균 계산 시 외곽 패딩 영역이 0으로 기여하게 만들기 위함입니다.
            # HOW: constant_values=0 으로 np.pad를 호출합니다.
            padded = np.pad(data, ((0,0), (0,0), (ctx.pH, ctx.pH), (ctx.pW, ctx.pW)), constant_values=0)
            out = np.zeros((B, C, out_h, out_w), dtype=np.float32)
            for h in range(out_h):
                for w in range(out_w):
                    h_start, w_start = h * ctx.sH, w * ctx.sW
                    # WHAT: 해당 패치의 모든 요소를 더하고 면적으로 나누어 평균을 구합니다.
                    # WHY: 이것이 평균 풀링의 정의이기 때문입니다.
                    # HOW: np.sum 후 (ctx.kH * ctx.kW)로 나눕니다.
                    out[:, :, h, w] = np.sum(padded[:, :, h_start:h_start+ctx.kH, w_start:w_start+ctx.kW], axis=(2, 3)) / (ctx.kH * ctx.kW)
            return Tensor(shape=(B, C, out_h, out_w), dtype='float32', device='cpu', data=out)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        if grad_output.device == 'gpu':
            raise AMEVAForgeDeviceError(
                "AvgPool2d backward is not supported on GPU tensors in the synchronous autograd engine in Release 1. "
                "Execute model on CPU or use Release 1 GPU-supported operators."
            )
        x, = ctx.saved_tensors
        grad_out_np = _require_cpu_data(grad_output, 'grad_output')
        x_np = _require_cpu_data(x, 'x')
        B, C, in_h, in_w = x_np.shape
        out_h = (in_h + 2 * ctx.pH - ctx.kH) // ctx.sH + 1
        out_w = (in_w + 2 * ctx.pW - ctx.kW) // ctx.sW + 1
        
        grad_padded = np.zeros((B, C, in_h + 2 * ctx.pH, in_w + 2 * ctx.pW), dtype=np.float32)
        grad_per_element = grad_out_np / (ctx.kH * ctx.kW)
        
        for h in range(out_h):
            for w in range(out_w):
                h_start, w_start = h * ctx.sH, w * ctx.sW
                # WHAT: 계산된 균등 기울기를 해당 윈도우 위치에 더합니다.
                # WHY: 브로드캐스팅을 통해 블록 전체에 값이 더해지게 하기 위함입니다.
                # HOW: 넘파이 슬라이싱 대입(+=)을 사용합니다.
                grad_padded[:, :, h_start:h_start+ctx.kH, w_start:w_start+ctx.kW] += grad_per_element[:, :, h:h+1, w:w+1]
                
        if ctx.pH > 0 or ctx.pW > 0:
            grad_x_np = grad_padded[:, :, ctx.pH:-ctx.pH if ctx.pH > 0 else None, ctx.pW:-ctx.pW if ctx.pW > 0 else None]
        else:
            grad_x_np = grad_padded
            
        if x.device == 'gpu':
            return (tensor(grad_x_np, device='gpu'),)
        else:
            return (Tensor(shape=x.shape, dtype='float32', device='cpu', data=grad_x_np),)

# WHAT: 평균 풀링(AvgPool2d) 편의 함수입니다.
# WHY: 쉽게 평균 풀링을 적용하기 위함입니다.
# HOW: AvgPool2dFunction.apply를 호출합니다.
def avg_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    return AvgPool2dFunction.apply(x, kernel_size, stride, padding)

# WHAT: Col2Im(Column to Image) 편의 함수입니다.
# WHY: 역전파 등에서 평탄화된 열벡터를 다시 2D 이미지 형태로 복원하기 위함입니다.
# HOW: Col2ImFunction.apply를 호출합니다. (Col2ImFunction 정의는 다른 곳에 존재하거나 별도 모듈에 있습니다)
def col2im(cols: Tensor, output_size: Tuple[int, int], kernel_size: int, stride: int = 1, padding: int = 0) -> Tensor:
    return Col2ImFunction.apply(cols, output_size, kernel_size, stride, padding)

# WHAT: 과적합 방지를 위한 드롭아웃(Dropout) 클래스입니다.
# WHY: 학습 중 신경망의 일부 뉴런을 무작위로 꺼서 특정 뉴런에 대한 의존도를 낮추기 위함입니다.
# HOW: 지정된 확률 p로 요소를 0으로 만들고, 남은 요소들은 1/(1-p)로 스케일링하여 기댓값을 유지합니다.
class DropoutFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, p: float, training: bool) -> Tensor:
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability p must be in [0, 1), but got {p}")
            
        # WHAT: 평가(Evaluation) 모드이거나 확률이 0인 경우의 처리입니다.
        # WHY: 추론 시에는 드롭아웃을 적용하지 않고 그대로 통과시켜야 하기 때문입니다.
        # HOW: 마스크를 None으로 설정하고 x의 데이터를 복제하거나 그대로 리턴합니다.
        if not training or p == 0.0:
            return x
        
        ctx.p = p
        if _should_use_gpu(x):
            # WHAT: GPU 기반 드롭아웃 처리입니다.
            # WHY: 커널 단에서 난수를 생성해 마스킹을 수행하기 위함입니다.
            # HOW: 호스트에서 난수 시드(seed)를 하나 뽑아 op_params로 넘겨주면, GPU 커널이 그 시드로 드롭아웃을 병렬 수행합니다.
            seed = float(np.random.rand())
            ctx.seed = seed
            out = Tensor(shape=x.shape, dtype="float32", device="gpu", op="dropout", parents=(x,), op_params=[seed, float(p)])
            return out
        else:
            # WHAT: CPU 기반 드롭아웃 처리입니다.
            # WHY: Numpy를 이용해 이항 분포 기반의 마스크를 생성하기 위함입니다.
            # HOW: np.random.binomial로 0 또는 1을 생성하고 (1.0 / (1.0 - p))로 보정합니다.
            data = _require_cpu_data(x, "x")
            mask = np.random.binomial(1, 1 - p, size=data.shape).astype(np.float32)
            res = data * mask * (1.0 / (1.0 - p))
            ctx.mask = mask
            return Tensor(shape=x.shape, dtype="float32", device="cpu", data=res)
            
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        # WHAT: 추론 모드였거나 적용되지 않았을 때의 역전파입니다.
        # WHY: 적용되지 않았다면 미분값도 그대로 흘려보내면 되기 때문입니다.
        # HOW: 그대로 리턴합니다.
        if getattr(ctx, 'mask', None) is None and not hasattr(ctx, 'seed'):
            return (grad_output,)
            
        if hasattr(ctx, 'seed'):
            # WHAT: GPU 드롭아웃 역전파입니다.
            # WHY: 동일한 시드(seed)를 사용해 동일한 마스크를 재생성하고 미분값을 통과시켜야 하기 때문입니다.
            # HOW: seed와 p를 파라미터로 넘겨 다시 드롭아웃 연산을 거치게 만듭니다.
            seed = ctx.seed
            p = ctx.p
            grad_in = Tensor(shape=grad_output.shape, dtype="float32", device="gpu", op="dropout", parents=(grad_output,), op_params=[seed, float(p)])
            return (grad_in,)
        else:
            # WHAT: CPU 드롭아웃 역전파입니다.
            # WHY: 순전파 때 저장해둔 불리언 마스크를 꺼내 미분값에 곱하기 위함입니다.
            # HOW: 데이터에 마스크를 곱하고 스케일링 상수도 동일하게 곱합니다.
            mask = ctx.mask
            p = ctx.p
            data = _require_cpu_data(grad_output, "grad")
            res = data * mask * (1.0 / (1.0 - p))
            return (Tensor(shape=grad_output.shape, dtype="float32", device="cpu", data=res),)

# WHAT: 드롭아웃(Dropout) 편의 함수입니다.
# WHY: 모듈이나 함수형 API에서 쉽게 사용할 수 있도록 하기 위함입니다.
# HOW: training=False 이거나 p=0.0 일 때는 Function.apply를 거치지 않고 순수 x를 반환하여 이전 autograd 그래프를 안전하게 보존합니다.
def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not (0.0 <= p < 1.0):
        raise ValueError(f"Dropout probability p must be in [0, 1), but got {p}")
    if not training or p == 0.0:
        return x
    return DropoutFunction.apply(x, p, training)

# WHAT: 임베딩(Embedding) 룩업을 수행하는 클래스입니다.
# WHY: 단어 인덱스 같은 정수 배열을 받아 밀집 벡터(Dense Vector) 공간의 실수 가중치로 변환하기 위함입니다.
# HOW: numpy 팬시 인덱싱(data_w[data_i])을 사용하고, 미분 시 np.add.at을 통해 추출된 위치에 기울기를 누적합니다.
class EmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, weight: Tensor, index: Tensor) -> Tensor:
        if weight.device == "gpu" or index.device == "gpu":
            raise AMEVAForgeUnsupportedOperationError(
                "GPU Embedding is not supported in Release 1. "
                "Embedding requires GPU scatter-add or atomic accumulation for correct backward."
            )
        ctx.save_for_backward(weight, index)
        data_w = _require_cpu_data(weight, "weight")
        data_i = _require_cpu_data(index, "index").astype(int)
        
        # WHAT: 정수 인덱스 배열에 해당하는 가중치 벡터들을 가져옵니다.
        # WHY: 그것이 임베딩 룩업의 본질이기 때문입니다.
        # HOW: data_w[data_i]로 가져옵니다.
        out_data = data_w[data_i]
        return Tensor(shape=out_data.shape, dtype="float32", device="cpu", data=out_data)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, type(None)]:
        if grad_output.device == "gpu":
            raise AMEVAForgeUnsupportedOperationError(
                "GPU Embedding backward is not supported in Release 1."
            )
        weight, index = ctx.saved_tensors
        data_i = _require_cpu_data(index, "index").astype(int)
        data_g = _require_cpu_data(grad_output, "grad_output")
        
        # WHAT: 원본 가중치 크기의 0 텐서를 만들고 미분값을 더해줍니다.
        # WHY: 여러 번 참조된 인덱스는 기울기가 합산되어야 가중치 업데이트가 제대로 이루어지기 때문입니다.
        # HOW: np.add.at 함수를 사용해 data_g를 data_i 위치에 안전하게 누적합니다.
        grad_w = np.zeros_like(_require_cpu_data(weight, "weight"))
        np.add.at(grad_w, data_i, data_g)
        return (Tensor(shape=weight.shape, dtype="float32", device="cpu", data=grad_w), None)

# WHAT: 임베딩 룩업 편의 함수입니다.
# WHY: 외부에서 룩업 연산을 쉽게 호출하기 위함입니다.
# HOW: EmbeddingFunction.apply를 호출합니다.
def embedding(weight: Tensor, index: Tensor) -> Tensor:
    return EmbeddingFunction.apply(weight, index)

# WHAT: 배치 행렬 곱(Batched Matrix Multiplication, BMM) 연산 클래스입니다.
# WHY: 트랜스포머(Transformer)의 어텐션 메커니즘 등에서 배치 단위로 (B, N, M)과 (B, M, P) 행렬 곱을 동시 수행하기 위함입니다.
# HOW: GPU는 batched_matmul 커널을, CPU는 np.matmul을 활용하며 역전파 시 상대방 전치행렬과 bmm을 재귀적으로 씁니다.
class BmmFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "bmm")
        
        # WHAT: 형상 검증입니다.
        # WHY: BMM은 3차원 텐서(Batch, Row, Col)만 허용하기 때문입니다.
        # HOW: len(shape)가 3인지, 그리고 내항 크기(M)와 배치 크기(B)가 일치하는지 assert합니다.
        if len(a.shape) != 3 or len(b.shape) != 3:
            raise AMEVAForgeShapeError("bmm requires 3D tensors")
        B, N, M = a.shape
        B2, M2, P = b.shape
        if B != B2 or M != M2:
            raise AMEVAForgeShapeError(f"bmm shape mismatch: {a.shape} and {b.shape}")

        if _should_use_gpu(a, b):
            # WHAT: GPU 배치 행렬 곱입니다.
            # WHY: 여러 배치 행렬 곱을 병렬로 연산하기 위함입니다.
            # HOW: op_params로 크기들을 전달하고 batched_matmul 커널을 부릅니다.
            return Tensor(shape=(B, N, P), dtype="float32", device="gpu",
                          op="batched_matmul", parents=(a, b), op_params=[int(B), int(N), int(P), int(M)])
        else:
            # WHAT: CPU 행렬 곱입니다.
            # WHY: Numpy는 3차원 이상 배열끼리 np.matmul을 시도할 때 첫 차원들을 자동으로 배치 축으로 인식해 행렬 곱을 수행해주기 때문입니다.
            # HOW: np.matmul(a, b)를 호출합니다.
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            import numpy as np
            res = np.matmul(data_a, data_b)
            return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        # WHAT: BMM의 역전파 공식 적용입니다.
        # WHY: 일반 행렬 곱처럼 dL/dA = dL/dY * B^T 이고 dL/dB = A^T * dL/dY 이기 때문입니다(배치 축 유지).
        # HOW: permute로 (0, 2, 1) 축을 섞어 내부 행렬 부분만 전치시키고 다시 bmm을 재귀적으로 호출합니다.
        grad_a = bmm(grad_output, permute(b, (0, 2, 1)))
        grad_b = bmm(permute(a, (0, 2, 1)), grad_output)
        return grad_a, grad_b

# WHAT: 배치 행렬 곱 편의 함수입니다.
# WHY: 쉽게 3차원 텐서 간의 행렬 곱을 수행하기 위함입니다.
# HOW: BmmFunction.apply를 호출합니다.
def bmm(a: Tensor, b: Tensor) -> Tensor:
    return BmmFunction.apply(a, b)

```

---

## `packages/forge-py/src/forge/optim.py`

```python
"""
================================================================================
[AMEVA-Forge 역사 기록 (Historical Metadata)]
생성일 (Created): Wed Aug 12 12:14:52 2026 +0900
수정 내역 (Modified):
- Wed Aug 12 12:59:35 2026 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
- Wed Aug 12 12:23:09 2026 +0900: Docs: Build Apache-style docs and unify tests
- Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
================================================================================
"""

# WHAT: typing 모듈에서 List 타입을 임포트합니다.
# WHY: 파라미터 리스트의 타입을 명시하여 정적 분석과 코드 가독성을 높이기 위함입니다.
# HOW: 타입 힌트 어노테이션에 List를 사용합니다.
from typing import List, Optional
import math
import numpy as np
from .tensor import Tensor
from .errors import (
    AMEVAForgeDeviceError,
    AMEVAForgeShapeError,
    AMEVAForgeValidationError,
    AMEVAForgeUnsupportedOperationError,
)
from .ops import _require_cpu_data

# WHAT: 모든 최적화 알고리즘의 베이스 클래스인 Optimizer입니다.
# WHY: 다양한 옵티마이저(SGD, Adam 등)가 공통으로 가질 속성과 메서드 인터페이스를 정의하기 위함입니다.
# HOW: 서브클래스에서 이 클래스를 상속받아 step 메서드를 구체화합니다.
class Optimizer:
    # WHAT: Optimizer 인스턴스를 초기화하는 메서드입니다.
    # WHY: 최적화할 파라미터 목록과 학습률을 객체 내부에 저장하기 위함입니다.
    # HOW: 전달받은 파라미터 리스트를 복사하여 저장하고 학습률을 설정합니다.
    def __init__(self, params: List[Tensor], lr: float = 0.01, strict_training: bool = False):
        # WHAT: 최적화 대상이 되는 텐서 파라미터들의 리스트입니다.
        # WHY: 원본 리스트가 외부에서 변경되는 것을 방지하고 안전하게 관리하기 위함입니다.
        # HOW: list() 함수를 통해 새로운 리스트 객체로 복사하여 저장합니다.
        self.params = list(params)
        
        # WHAT: 학습률(Learning Rate)입니다.
        # WHY: 각 파라미터 업데이트 시 그래디언트를 얼마나 반영할지 스텝 크기를 결정하기 위함입니다.
        # HOW: 속성으로 저장되어 step 연산에 곱해집니다.
        self.lr = lr

        # WHAT: 엄격 학습 모드(Strict Training Mode) 플래그입니다.
        # WHY: NaN/Inf 그래디언트 발생 시 즉시 Fail-Fast 예외를 발생시켜 학습 발산을 조기 차단하기 위함입니다.
        self.strict_training = strict_training
    
    # WHAT: 파라미터 업데이트를 수행하는 메서드 인터페이스입니다.
    # WHY: 각 옵티마이저마다 고유의 업데이트 규칙(규칙)을 적용하기 위함입니다.
    # HOW: 베이스 클래스에서는 NotImplementedError를 발생시키며, 하위 클래스에서 오버라이드해야 합니다.
    def step(self):
        raise NotImplementedError
    
    def _active_devices(self):
        return {
            p.device
            for p in self.params
            if p.grad is not None
        }

    def _validate_param_grad_pair(self, p: Tensor) -> None:
        """파라미터와 gradient의 장치, shape, dtype 계약을 검증한다."""
        if p.grad is None:
            return
        if p.device != p.grad.device:
            raise AMEVAForgeDeviceError(
                f"Parameter/gradient device mismatch: {p.device} != {p.grad.device}"
            )
        if tuple(p.shape) != tuple(p.grad.shape):
            raise AMEVAForgeShapeError(
                f"Parameter/gradient shape mismatch: {p.shape} != {p.grad.shape}"
            )
        if p.dtype != p.grad.dtype:
            raise AMEVAForgeDeviceError(
                f"Parameter/gradient dtype mismatch: {p.dtype} != {p.grad.dtype}"
            )

    # WHAT: 등록된 모든 파라미터의 그래디언트를 초기화(None)하는 메서드입니다.
    # WHY: 새로운 미니배치의 학습을 시작할 때 이전 배치의 누적된 그래디언트를 지우기 위함입니다.
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                if getattr(p.grad, 'device', None) == 'gpu' and getattr(p.grad, '_handle', None) is not None:
                    try:
                        p.grad.dispose()
                    except Exception:
                        pass
                p.grad = None


# WHAT: 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 옵티마이저 클래스입니다.
# WHY: 모멘텀(Momentum)이 적용될 수 있는 기본적인 기울기 하강 업데이트를 수행하기 위함입니다.
# HOW: Optimizer를 상속받아 step 메서드를 구현하고, 속도를 추적하는 velocity 배열을 관리합니다.
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, strict_training: bool = False):
        super().__init__(params, lr, strict_training=strict_training)
        self.momentum = momentum
        self.velocity = [None] * len(self.params)
    
    def step(self, strict: Optional[bool] = None):
        """
        CPU parameter 전용 동기 SGD step.

        GPU parameter는 readback이 비동기이므로 이 메서드에서 처리하지 않는다.
        GPU 학습에서는 반드시 `await optimizer.step_async()`를 사용한다.
        """
        use_strict = self.strict_training if strict is None else strict
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            self._validate_param_grad_pair(p)

            if p.device == "gpu":
                raise AMEVAForgeDeviceError(
                    "SGD.step() is CPU-only for GPU-backed parameters. "
                    "Use: await optimizer.step_async()"
                )

            grad_data = p.grad.numpy()
            if use_strict and not np.isfinite(grad_data).all():
                raise AMEVAForgeValidationError(
                    "Non-finite gradient (NaN/Inf) detected in strict training mode."
                )

            param_data = p.numpy()

            if self.momentum > 0.0:
                if self.velocity[i] is None:
                    self.velocity[i] = grad_data.copy()
                else:
                    self.velocity[i] = (
                        self.momentum * self.velocity[i] + grad_data
                    )
                update = self.velocity[i]
            else:
                update = grad_data

            p._data = (param_data - self.lr * update).astype(np.float32)
            p._version += 1
            p.grad = None

    async def step_async(self, strict: Optional[bool] = None):
        """
        CPU와 GPU parameter를 모두 처리하는 공식 비동기 SGD step.

        GPU parameter는 기존 AXPY WGSL을 통해 readback 없이 in-place 갱신한다.
        Release 1에서는 GPU momentum을 지원하지 않으며, 단일 스텝 내 혼합 장치를 허용하지 않는다.
        """
        use_strict = self.strict_training if strict is None else strict
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr}")

        active_devices = self._active_devices()
        if len(active_devices) > 1:
            raise AMEVAForgeDeviceError(
                "Mixed CPU/GPU parameters in one SGD step are not supported in Release 1. "
                "Use one optimizer per device."
            )

        if self.momentum > 0.0 and any(
            p.grad is not None and p.device == "gpu" for p in self.params
        ):
            raise AMEVAForgeDeviceError(
                "GPU momentum SGD is outside Release 1. "
                "Use momentum=0.0 or implement a GPU velocity tensor."
            )

        from .graph import GraphBuilder
        from .bridge import js_execute_graph

        builder = GraphBuilder()
        cpu_updates = []
        param_out_map = []
        param_entries = []

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            self._validate_param_grad_pair(p)

            if p.device == "cpu":
                grad_data = p.grad.numpy()
                if use_strict and not np.isfinite(grad_data).all():
                    raise AMEVAForgeValidationError(
                        "Non-finite gradient (NaN/Inf) detected in strict training mode."
                    )
                param_data = p.numpy()

                if self.momentum > 0.0:
                    if self.velocity[i] is None:
                        self.velocity[i] = grad_data.copy()
                    else:
                        self.velocity[i] = (
                            self.momentum * self.velocity[i] + grad_data
                        )
                    update = self.velocity[i]
                else:
                    update = grad_data

                cpu_updates.append((p, param_data, update))
                continue

            await p.realize()
            await p.grad.realize()

            if p._handle is None or p.grad._handle is None:
                raise AMEVAForgeDeviceError(
                    "GPU SGD requires realized parameter and gradient handles."
                )

            if use_strict:
                grad_check = await p.grad.numpy_async()
                if not np.isfinite(grad_check).all():
                    raise AMEVAForgeValidationError(
                        "Non-finite gradient (NaN/Inf) detected in strict training mode on GPU."
                    )

            num_elements = int(np.prod(p.shape, dtype=np.int64))
            if num_elements <= 0:
                raise AMEVAForgeShapeError(
                    f"GPU SGD does not support empty parameter: shape={p.shape}"
                )

            grad_id = builder.add_load(p.grad.shape, p.grad._handle)
            param_id = builder.add_load(p.shape, p._handle)
            param_entries.append((p, num_elements, grad_id, param_id))

        for p, num_elements, grad_id, param_id in param_entries:
            out_id = builder.add_op(
                "axpy",
                p.shape,
                [grad_id, param_id],
                [num_elements, float(self.lr)],
            )
            param_out_map.append((p, out_id))

        # CPU 계산은 검증이 끝난 뒤 원자적으로 반영한다.
        for p, param_data, update in cpu_updates:
            p._data = (param_data - self.lr * update).astype(np.float32)
            p._version += 1
            p.grad = None

        # GPU 그래프는 단일 일괄 FFI 호출로 실행하여 브리지 오버헤드를 최소화한다.
        if param_out_map:
            instructions, inputs = builder.compile()
            result = await js_execute_graph(instructions, inputs)

            for p, out_id in param_out_map:
                returned_handle = result.get(str(out_id)) or result.get(out_id)

                # axpy는 in-place 계약이므로 같은 parameter handle을 반환해야 한다.
                if returned_handle != p._handle:
                    raise AMEVAForgeDeviceError(
                        "AXPY contract violation: optimizer returned a different handle."
                    )

                p._version += 1
                if p.grad is not None and getattr(p.grad, 'device', None) == 'gpu':
                    try:
                        p.grad.dispose()
                    except Exception:
                        pass
                p.grad = None


# WHAT: Adam(Adaptive Moment Estimation) 옵티마이저 클래스입니다.
# WHY: 1차 모멘트(평균)와 2차 모멘트(분산)를 추정하여 각 파라미터마다 적응형(adaptive) 학습률을 적용하기 위함입니다.
# HOW: 그래디언트의 지수 이동 평균을 두 가지 형태로 누적하고 편향이 보정된(bias-corrected) 값으로 가중치를 업데이트합니다.
class Adam(Optimizer):
    # WHAT: Adam 인스턴스 초기화 메서드입니다.
    # WHY: Adam 알고리즘에 필요한 하이퍼파라미터(베타, 엡실론)를 설정하고 모멘트 저장 공간을 할당하기 위함입니다.
    # HOW: 상위 초기화 후 beta, eps 등을 저장하고 상태 변수 리스트를 생성합니다.
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        # WHAT: 1차 및 2차 모멘트 추정을 위한 감쇠율(decay rate)입니다.
        # WHY: 과거의 그래디언트 정보를 어느 정도 비율로 반영할지 결정하기 위함입니다.
        # HOW: 튜플에서 언패킹하여 각각 beta1, beta2로 저장합니다.
        self.beta1, self.beta2 = betas
        
        # WHAT: 분모가 0이 되는 것을 방지하는 작은 상수입니다.
        # WHY: 수치적 안정성을 보장하기 위함입니다.
        # HOW: 업데이트 식의 제곱근 항에 더해집니다.
        self.eps = eps
        
        # WHAT: 파라미터별 1차 모멘트(평균 추정치)를 저장하는 리스트입니다.
        # WHY: 각 방향별 모멘텀을 추적하기 위함입니다.
        # HOW: None으로 초기화된 리스트를 파라미터 개수만큼 생성합니다.
        self.m = [None] * len(self.params)
        
        # WHAT: 파라미터별 2차 모멘트(비중심 분산 추정치)를 저장하는 리스트입니다.
        # WHY: 그래디언트의 크기 변화에 따라 학습률을 조절하기 위함입니다.
        # HOW: None으로 초기화된 리스트를 파라미터 개수만큼 생성합니다.
        self.v = [None] * len(self.params)
        
        # WHAT: 최적화 스텝을 밟은 횟수(타임스텝)입니다.
        # WHY: 편향 보정(bias correction) 시 t승을 계산하기 위함입니다.
        # HOW: 0으로 시작하여 step()마다 1씩 증가합니다.
        self.t = 0
    
    # WHAT: Adam의 파라미터 업데이트를 1스텝 진행하는 메서드입니다.
    # WHY: 각 파라미터마다 적응형 학습률 수식을 계산하여 가중치를 최적화하기 위함입니다.
    # HOW: 타임스텝을 올리고 각 파라미터에 대해 m, v를 갱신 및 보정한 뒤 값을 차감합니다.
    def step(self):
        # WHAT: 타임스텝 증가 연산입니다.
        # WHY: 편향 보정을 정확히 계산하기 위해 현재 몇 번째 업데이트인지 기록하기 위함입니다.
        # HOW: self.t 값에 1을 더합니다.
        self.t += 1
        
        # WHAT: 파라미터들을 순회하며 Adam 수식을 개별적으로 적용하는 루프입니다.
        # WHY: 신경망 전체 가중치를 갱신하기 위함입니다.
        # HOW: enumerate를 통해 인덱스와 파라미터 객체를 순회합니다.
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            if p.device == "gpu":
                raise AMEVAForgeDeviceError(
                    "Adam optimizer does not support synchronous GPU step in Release 1. "
                    "Use SGD.step_async() for GPU models, or run Adam on CPU tensors."
                )
            else:
                g = _require_cpu_data(p.grad, "p.grad")
                param_data = _require_cpu_data(p, "p")
            
            if self.m[i] is None:
                # WHAT: 처음 업데이트 시 모멘트 배열을 0으로 초기화합니다.
                # WHY: 메모리가 아직 할당되지 않은 상태이므로 그래디언트와 같은 크기의 0 행렬을 만들기 위함입니다.
                # HOW: np.zeros_like 함수를 사용합니다.
                self.m[i] = np.zeros_like(g)
                self.v[i] = np.zeros_like(g)
            
            # WHAT: 1차 및 2차 모멘트의 지수 이동 평균을 업데이트합니다.
            # WHY: 현재 그래디언트의 방향과 크기를 과거 통계치에 반영하기 위함입니다.
            # HOW: beta 상수와 결합된 수식으로 m과 v를 갱신합니다.
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            
            # WHAT: 학습 초기에 모멘트가 0으로 편향되는 것을 방지하는 편향 보정 연산입니다.
            # WHY: 초기에 m과 v가 너무 작게 측정되어 학습이 비정상적으로 튀는 것을 막기 위함입니다.
            # HOW: 각각 1 - beta^t 로 나누어 m_hat, v_hat을 계산합니다.
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # WHAT: 최종적으로 파라미터를 업데이트하는 수식입니다.
            # WHY: 보정된 모멘트를 사용하여 적응적으로 스텝을 이동하기 위함입니다.
            # HOW: 파라미터 데이터에서 `lr * m_hat / (sqrt(v_hat) + eps)`를 차감합니다.
            param_data = param_data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # WHAT: 갱신된 데이터를 텐서에 덮어쓰고 기울기를 비웁니다.
            # WHY: 다음 스텝을 위해 상태를 최신화하기 위함입니다.
            # HOW: astype(np.float32)로 형변환 후 p._data에 할당하고 grad를 초기화합니다.
            p._data = param_data.astype(np.float32)
            p._version += 1
            p.grad = None

    async def step_async(self):
        for p in self.params:
            if p.device == "gpu":
                raise AMEVAForgeUnsupportedOperationError(
                    "Adam async GPU step is not supported in Release 1. "
                    "Use SGD for GPU training or move parameters to CPU."
                )
        self.step()


# WHAT: 파라미터들의 그래디언트 글로벌 L2 노름(Norm)을 제한(Clip)하는 함수입니다.
# WHY: RNN이나 깊은 신경망에서 그래디언트 폭발(Gradient Exploding) 문제를 방지하여 학습을 안정화하기 위함입니다.
# HOW: 모든 그래디언트의 제곱합을 구해 노름을 계산하고, max_norm을 넘으면 그 비율만큼 전체 기울기를 축소합니다.
def clip_grad_norm(parameters: List[Tensor], max_norm: float):
    # WHAT: 전체 그래디언트의 제곱합을 누적할 변수입니다.
    # WHY: L2 노름을 계산하기 위한 중간 합계를 저장하기 위함입니다.
    # HOW: 0.0으로 초기화한 뒤 순회하며 누적합니다.
    total_norm = 0.0
    
    # WHAT: 모든 파라미터를 순회하며 그래디언트 제곱합을 계산하는 루프입니다.
    # WHY: 전체 벡터 공간에서의 길이를 파악하기 위함입니다.
    # HOW: 각 파라미터의 grad를 배열로 변환해 요소별 제곱 후 더합니다.
    for p in parameters:
        if p.device == "gpu" or (p.grad is not None and p.grad.device == "gpu"):
            raise AMEVAForgeDeviceError(
                "clip_grad_norm is supported only for CPU parameters in Release 1. "
                "GPU-native gradient clipping requires asynchronous GPU reduction."
            )
        if p.grad is not None:
            g = _require_cpu_data(p.grad, "p.grad")
            total_norm += np.sum(g ** 2)
                
    # WHAT: 전체 분산 합계에 제곱근을 취해 최종 L2 노름을 구합니다.
    # WHY: 스케일링 계수를 계산하기 위한 실수값을 얻기 위함입니다.
    # HOW: np.sqrt 연산을 수행하고 float 타입으로 변환합니다.
    total_norm = float(np.sqrt(total_norm))
    
    # WHAT: 그래디언트를 스케일링할 비율(계수)입니다.
    # WHY: 전체 노름이 max_norm을 초과했을 때 그 비율만큼 줄이기 위함입니다.
    # HOW: max_norm을 total_norm으로 나누어 계산하며, 0 분할 방지를 위해 1e-6을 더합니다.
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1.0:
        # WHAT: 그래디언트가 제한치를 초과했을 때 실제 클리핑을 수행하는 루프입니다.
        # WHY: 모든 기울기의 방향은 유지한 채 크기만 비례적으로 줄이기 위함입니다.
        # HOW: 각 파라미터를 다시 순회하며 그래디언트 배열에 clip_coef를 곱해줍니다.
        for p in parameters:
            if p.grad is not None:
                g = _require_cpu_data(p.grad, "p.grad")
                p.grad._data = (g * clip_coef).astype(np.float32)

# WHAT: 개별 그래디언트 요소의 최댓값/최솟값을 직접 자르는(Value Clipping) 함수입니다.
# WHY: 매우 큰 특정 그래디언트 값이 전체 학습을 망치는 것을 방지하기 위함입니다.
# HOW: 각 그래디언트 요소를 [-clip_value, clip_value] 범위 내로 제한(clip)합니다.
def clip_grad_value(parameters: List[Tensor], clip_value: float):
    # WHAT: 모든 파라미터를 순회하는 루프입니다.
    # WHY: 각 텐서의 그래디언트에 클리핑을 적용하기 위함입니다.
    # HOW: for문을 통해 파라미터를 하나씩 꺼냅니다.
    for p in parameters:
        if p.device == "gpu" or (p.grad is not None and p.grad.device == "gpu"):
            raise AMEVAForgeDeviceError(
                "clip_grad_value is supported only for CPU parameters in Release 1."
            )
        if p.grad is not None:
            g = _require_cpu_data(p.grad, "p.grad")
            p.grad._data = np.clip(g, -clip_value, clip_value).astype(np.float32)

# WHAT: 정해진 에포크 주기마다 학습률을 단계적으로 감소시키는 스케줄러입니다.
# WHY: 학습 후반부에 학습률을 낮춰 더 세밀한 최적화 지점(Global Minimum)에 도달하게 하기 위함입니다.
# HOW: step이 호출될 때마다 카운트를 올리고 주기(step_size)에 도달하면 lr에 gamma를 곱합니다.
class StepLR:
    # WHAT: StepLR 스케줄러의 초기화 메서드입니다.
    # WHY: 제어할 옵티마이저와 감소 주기, 감소 비율을 설정하기 위함입니다.
    # HOW: 속성들을 객체 내부에 저장합니다.
    def __init__(self, optimizer, step_size, gamma=0.1):
        # WHAT: 대상이 되는 옵티마이저 인스턴스입니다.
        # WHY: 옵티마이저 내부에 저장된 lr 값을 직접 수정하기 위함입니다.
        # HOW: 참조를 저장합니다.
        self.optimizer = optimizer
        
        # WHAT: 학습률을 감소시킬 에포크 주기입니다.
        # WHY: 몇 에포크마다 감쇠시킬지 판단하는 기준이 되기 때문입니다.
        # HOW: 변수로 저장됩니다.
        self.step_size = step_size
        
        # WHAT: 학습률을 감소시킬 비율(감쇠율)입니다.
        # WHY: 기존 학습률에 곱해져 값을 줄이는 강도를 결정하기 위함입니다.
        # HOW: 보통 0.1 등의 값을 저장합니다.
        self.gamma = gamma
        
        # WHAT: 지금까지 진행된 에포크 수입니다.
        # WHY: 주기에 도달했는지 확인하는 카운터로 사용하기 위함입니다.
        # HOW: 0으로 시작합니다.
        self.last_epoch = 0
        
    # WHAT: 1에포크가 끝났을 때 스케줄러를 한 스텝 전진시키는 메서드입니다.
    # WHY: 조건을 검사하고 필요 시 학습률을 감소시키기 위함입니다.
    # HOW: last_epoch를 1 늘리고, step_size로 나누어 떨어지면 lr에 gamma를 곱합니다.
    def step(self, metrics=None):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

# WHAT: 코사인 곡선을 따라 학습률을 부드럽게 감소시키는 스케줄러입니다.
# WHY: 웜 리스타트(Warm restart) 효과나 부드러운 하강을 통해 로컬 미니멈을 효율적으로 탈출/수렴하기 위함입니다.
# HOW: 반주기(T_max) 동안 base_lr에서 eta_min까지 코사인 함수 모양으로 학습률을 조절합니다.
class CosineAnnealingLR:
    # WHAT: CosineAnnealingLR의 초기화 메서드입니다.
    # WHY: 주기, 최소 학습률 등 코사인 스케줄링을 위한 환경을 준비하기 위함입니다.
    # HOW: 입력받은 인자들을 속성으로 저장합니다.
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = 0
        # WHAT: 스케줄러 시작 시점의 기준 학습률입니다.
        # WHY: 코사인 수식에서 최대 진폭의 기준점으로 쓰기 위함입니다.
        # HOW: 옵티마이저의 현재 lr을 저장해둡니다.
        self.base_lr = optimizer.lr
        
    # WHAT: 에포크마다 코사인 수식에 따라 학습률을 업데이트하는 메서드입니다.
    # WHY: 곡선의 다음 지점에 해당하는 학습률 값을 반영하기 위함입니다.
    # HOW: math.cos를 사용해 반환된 값을 옵티마이저 lr에 대입합니다.
    def step(self, metrics=None):
        self.last_epoch += 1
        import math
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2

# WHAT: 검증(validation) 평가 지표(metrics)가 정체될 때 학습률을 낮추는 스케줄러입니다.
# WHY: 손실값이 더 이상 떨어지지 않는 고원(plateau)에 도달했을 때 미세 조정을 유도하기 위함입니다.
# HOW: 정해진 횟수(patience) 동안 최고 기록(best)이 갱신되지 않으면 lr을 factor배 축소시킵니다.
class ReduceLROnPlateau:
    # WHAT: ReduceLROnPlateau 초기화 메서드입니다.
    # WHY: 모니터링 방식(최소화/최대화), 감소 배율, 참을성 횟수를 설정하기 위함입니다.
    # HOW: 초기 인자들을 저장하고, best 점수와 정체 카운터를 초기화합니다.
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, min_lr=0):
        self.optimizer = optimizer
        
        # WHAT: 지표의 개선 방향입니다 ('min'은 감소, 'max'는 증가가 개선).
        # WHY: Loss는 낮아져야 좋고, Accuracy는 높아져야 좋으므로 유연성을 주기 위함입니다.
        # HOW: 문자열로 모드를 기록합니다.
        self.mode = mode
        
        # WHAT: 학습률 축소 비율입니다.
        # WHY: 정체 시 얼마나 학습률을 줄일지 강도를 결정하기 위함입니다.
        # HOW: 저장해두었다가 lr * factor에 사용합니다.
        self.factor = factor
        
        # WHAT: 참을성(유예 기간) 횟수입니다.
        # WHY: 일시적인 정체에 바로 학습률을 깎는 것을 방지하기 위함입니다.
        # HOW: 나쁜 에포크 카운트가 이를 초과할 때만 축소가 일어납니다.
        self.patience = patience
        
        # WHAT: 허용되는 최소 학습률의 하한선입니다.
        # WHY: 학습률이 너무 0에 가까워져 학습이 아예 멈추는 것을 막기 위함입니다.
        # HOW: 축소 후 이 값과 비교하여 큰 값을 취합니다(max).
        self.min_lr = min_lr
        
        # WHAT: 모니터링 중인 지표의 역대 최고 기록입니다.
        # WHY: 현재 지표가 이전보다 개선되었는지 판별하는 기준선으로 쓰기 위함입니다.
        # HOW: 초기엔 None으로 두고 첫 스텝에 설정합니다.
        self.best = None
        
        # WHAT: 지표가 개선되지 않은 연속 에포크 횟수입니다.
        # WHY: patience를 넘었는지 체크하기 위함입니다.
        # HOW: 개선되면 0으로 리셋, 아니면 1씩 증가합니다.
        self.num_bad_epochs = 0
        
    # WHAT: 매 에포크의 평가 지표를 받아 판단 후 학습률을 조절하는 메서드입니다.
    # WHY: 실시간 성능 추이를 기반으로 동적 스케줄링을 하기 위함입니다.
    # HOW: 지표 개선 여부를 확인하고 카운터를 관리하며, 조건 만족 시 옵티마이저 lr을 줄입니다.
    def step(self, metrics):
        if self.best is None:
            # WHAT: 초기 상태 설정입니다.
            # WHY: 비교할 대상이 없으므로 첫 평가 지표를 최고 기록으로 삼기 위함입니다.
            # HOW: best 변수에 metrics를 복사하고 리턴합니다.
            self.best = metrics
            return
            
        # WHAT: 지표가 이전 기록보다 나아졌는지를 나타내는 플래그 변수입니다.
        # WHY: 조건 분기를 통합하기 위함입니다.
        # HOW: mode에 따라 비교 연산을 다르게 수행합니다.
        is_better = False
        if self.mode == 'min' and metrics < self.best:
            is_better = True
        elif self.mode == 'max' and metrics > self.best:
            is_better = True
            
        if is_better:
            # WHAT: 개선된 경우의 상태 갱신입니다.
            # WHY: 최고 기록을 갱신하고 정체 상태를 초기화하기 위함입니다.
            # HOW: best를 교체하고 num_bad_epochs를 0으로 만듭니다.
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            # WHAT: 개선되지 않은 경우의 상태 갱신입니다.
            # WHY: 정체 기간을 카운트하기 위함입니다.
            # HOW: num_bad_epochs에 1을 더합니다.
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            # WHAT: 정체기가 한계를 초과했을 때 실제 학습률을 감소시키는 부분입니다.
            # WHY: 고원을 벗어나기 위해 더 세밀한 보폭을 적용하기 위함입니다.
            # HOW: 기존 학습률에 factor를 곱하고 하한선과 비교한 뒤 적용합니다.
            self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
            # WHAT: 감소 이후 정체 카운터를 리셋합니다.
            # WHY: 새로운 학습률 스케일에서 다시 patience만큼 기회를 주기 위함입니다.
            # HOW: 0으로 초기화합니다.
            self.num_bad_epochs = 0



```

---

## `packages/forge-py/src/forge/optim.pyi`

```typescript
from typing import Iterable, List
from .tensor import Tensor

class Optimizer:
    params: List[Tensor]
    def __init__(self, params: Iterable[Tensor]) -> None: ...
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...
    async def step_async(self) -> None: ...

class SGD(Optimizer):
    lr: float
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01) -> None: ...
    def step(self) -> None: ...
    async def step_async(self) -> None: ...

```

---

## `packages/forge-py/src/forge/serialization.py`

```python
"""
================================================================================
파일 이력 (Historical Metadata)
Created: 2026-08-12 12:59:35 +0900 (첫 커밋 기준)
Modified:
  - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
================================================================================
이 파일은 모델의 가중치(상태)를 파일 시스템에 직렬화하여 저장하고, 
저장된 상태를 다시 메모리로 불러오는 기능을 제공합니다.
"""
# numpy 배열을 다루기 위한 라이브러리 임포트 (데이터 직렬화 및 역직렬화에 사용)
import numpy as np
# 신경망 모델의 기본 클래스 임포트
from .nn import Module

def save_model(model: Module, path: str):
    """
    WHAT: 주어진 모델의 가중치를 파일로 저장하는 함수입니다.
    WHY: 학습이 완료된 모델을 나중에 다시 사용하거나 배포하기 위해 파일 시스템에 상태를 영구적으로 보존할 필요가 있기 때문입니다.
    HOW: 모델의 state_dict를 호출하여 파라미터 딕셔너리를 얻고, 이를 numpy 포맷(.npz)으로 변환하여 저장합니다.
    """
    # WHAT: 모델의 현재 상태(가중치 등)를 담은 딕셔너리입니다.
    # WHY: 파일로 직렬화할 데이터를 추출하기 위함입니다.
    # HOW: keep_vars=False로 설정하여 순수 데이터만 추출합니다.
    state_dict = model.state_dict(keep_vars=False)
    
    # WHAT: numpy 배열 형태의 데이터를 담을 빈 딕셔너리입니다.
    # WHY: np.savez 함수에 전달하기 위해서는 모든 값이 numpy 배열이어야 하기 때문입니다.
    # HOW: 빈 딕셔너리로 초기화한 후 반복문을 통해 값을 채웁니다.
    numpy_dict = {}
    
    # WHAT: 모델의 상태 딕셔너리 내 모든 키(k)와 값(v) 쌍을 순회하는 반복문입니다.
    # WHY: 텐서 형태의 값들을 numpy 배열로 변환하여 numpy_dict에 옮겨 담기 위함입니다.
    # HOW: items() 메서드를 호출하여 반환된 키-값 쌍에 대해 반복합니다.
    for k, v in state_dict.items():
        if hasattr(v, 'numpy'):
            # WHAT: 텐서를 numpy 배열로 변환하여 저장합니다.
            # WHY: Tensor 객체는 직접 저장할 수 없으므로 호환 가능한 형식으로 변환해야 합니다.
            # HOW: v.numpy()를 호출한 값을 키 k로 numpy_dict에 할당합니다.
            numpy_dict[k] = v.numpy()
        else:
            # WHAT: 텐서가 아닌 값을 그대로 저장합니다.
            # WHY: 이미 numpy 배열이거나 기본 자료형인 경우 추가 변환이 필요 없기 때문입니다.
            # HOW: 원본 값을 그대로 키 k로 numpy_dict에 할당합니다.
            numpy_dict[k] = v
            
    # WHAT: 딕셔너리의 내용을 파일로 저장합니다.
    # WHY: 지정된 경로에 데이터를 물리적으로 기록하기 위해서입니다.
    # HOW: np.savez 함수에 경로와 키워드 인자(**numpy_dict)를 전달하여 호출합니다.
    np.savez(path, **numpy_dict)

def load_model(model: Module, path: str):
    """
    WHAT: 저장된 파일로부터 가중치 데이터를 읽어와 모델에 덮어씌우는 함수입니다.
    WHY: 이전에 저장된 모델의 상태를 복구하여 추론(Inference)이나 추가 학습을 이어서 진행하기 위해 필요합니다.
    HOW: np.load로 데이터를 읽어들인 뒤, 포함된 모든 키-값 쌍을 state_dict 형태로 재구성하고, 
         model.load_state_dict를 통해 모델 내부로 데이터를 주입합니다.
    """
    # WHAT: 디스크에서 읽어들인 numpy 데이터를 담는 객체입니다.
    # WHY: 파일에 압축되어 저장된 텐서 값들에 접근하기 위함입니다.
    # HOW: np.load 함수를 사용하여 지정된 경로의 파일을 엽니다.
    data = np.load(path)
    
    # WHAT: 모델에 주입하기 위해 재생성된 상태 딕셔너리입니다.
    # WHY: 모델의 load_state_dict가 요구하는 딕셔너리 형식에 맞추기 위함입니다.
    # HOW: 딕셔너리 컴프리헨션을 사용하여 data 파일 내의 모든 키(files)를 순회하는 루프로 값을 복원합니다.
    state_dict = {k: data[k] for k in data.files}
    
    # WHAT: 모델의 파라미터를 업데이트합니다.
    # WHY: 복원된 상태 딕셔너리 데이터를 실제 모델에 적용하기 위해서입니다.
    # HOW: model.load_state_dict 메서드를 호출하여 상태를 주입합니다.
    model.load_state_dict(state_dict)

```

---

## `packages/forge-py/src/forge/tensor.py`

```python
"""
================================================================================
파일 이력 (Historical Metadata)
Created: 2026-08-12 12:14:52 +0900 (첫 커밋 기준)
Modified:
  - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
  - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
================================================================================
이 파일은 핵심 텐서 자료구조와 지연 평가(Lazy Evaluation), 자동 미분(Autograd)을 위한 그래프 구성 기능을 구현합니다.
"""
# 타입 힌트를 위한 타이핑 모듈 임포트
from typing import Any, Tuple, Optional, List
# 넘파이 배열 조작을 위한 임포트
import numpy as np
# 패키지 내부 커스텀 에러 클래스 임포트
from .errors import AMEVAForgeDisposedError, AMEVAForgeShapeError, AMEVAForgeDeviceError


def build_lazy_topo(root: 'Tensor') -> List['Tensor']:
    """
    WHAT: 레이지 텐서 연산 그래프의 위상 정렬(Topological Sort)을 수행하여 루트부터 리프 노드까지의 순서를 반환하는 함수입니다.
    WHY: 연산을 실행(Realize)할 때 의존성이 있는 부모 텐서들이 먼저 계산되어야 하므로 올바른 실행 순서를 보장하기 위함입니다.
    HOW: 재귀 호출로 인한 스택 오버플로우를 막기 위해 반복적 깊이 우선 탐색(Iterative DFS) 방식으로 구현되었습니다.
    """
    # WHAT: 최종 위상 정렬 결과를 담을 텐서 리스트입니다.
    # WHY: 부모 노드 방문이 모두 완료된 후(Post-order) 텐서를 순차적으로 담아 반환하기 위해서입니다.
    # HOW: 노드의 부모들을 모두 방문하고 나면 리스트의 끝에(append) 추가합니다.
    topo: List['Tensor'] = []
    
    # WHAT: 이미 방문한 노드의 메모리 주소(ID)를 기록하는 집합(Set)입니다.
    # WHY: 그래프 내에서 동일한 노드를 중복으로 방문하거나 무한 루프에 빠지는 것을 방지하기 위함입니다.
    # HOW: 각 노드를 스택에 넣을 때 그 ID를 이 집합에 추가하고 확인합니다.
    visited: set = set()
    
    # (node, parent_index) 스택: 현재 노드와 다음에 방문할 부모의 인덱스
    # WHAT: 순회를 제어하기 위한 명시적인 작업 스택입니다.
    # WHY: 시스템 재귀 호출 한도 초과 문제를 우회하고 더 깊은 모델도 처리할 수 있게 하기 위해 반복문 기반 스택을 씁니다.
    # HOW: 시작 노드(root)와 첫 부모 인덱스(0)를 튜플로 묶어 리스트에 넣습니다.
    stack: list = [(root, 0)]
    visited.add(id(root))

    # WHAT: 스택에 남아있는 작업이 없을 때까지 반복하는 메인 루프입니다.
    # WHY: 그래프에 연결된 모든 노드를 순회하여 빠짐없이 정렬된 리스트를 만들기 위해서입니다.
    # HOW: 스택이 비어있지 않으면 계속 루프를 실행합니다.
    while stack:
        # WHAT: 스택의 가장 위(마지막)에 있는 현재 텐서 노드와 방문할 부모 인덱스를 가져옵니다.
        # WHY: DFS 방식이므로 가장 최근에 추가된(깊이가 깊은) 노드부터 처리하기 위함입니다.
        # HOW: 스택 리스트의 마지막 요소([-1])를 튜플 언패킹합니다.
        node, idx = stack[-1]
        
        # WHAT: 현재 텐서의 부모 노드 튜플입니다.
        # WHY: 이미 GPU 버퍼가 실체화(realize)된 텐서는 리프(load) 노드로 취급하여 부모를 재탐색하지 않음으로써 그래프 낭비와 고아 텐서 참조를 방지합니다.
        # HOW: _handle이 None이 아닐 경우 빈 튜플로 처리합니다.
        if getattr(node, '_handle', None) is not None:
            parents = ()
        else:
            parents = getattr(node, '_parents', ())

        # WHAT: 아직 방문하지 않은 부모 노드가 남아있는지 검사하는 조건문입니다.
        # WHY: 모든 부모 노드를 먼저 처리한 뒤에야 현재 노드를 처리(위상 정렬 리스트 추가)할 수 있기 때문입니다.
        # HOW: 현재 인덱스(idx)가 전체 부모 개수보다 작은지 비교합니다.
        if idx < len(parents):
            # 다음 부모로 진행
            # WHAT: 현재 노드의 인덱스를 1 증가시켜 스택 상단에 갱신합니다.
            # WHY: 현재 부모 탐색을 마치고 돌아왔을 때 다음 부모를 이어서 탐색할 수 있게 상태를 저장합니다.
            # HOW: 스택의 마지막 요소를 새로운 튜플로 교체합니다.
            stack[-1] = (node, idx + 1)
            
            # WHAT: 현재 인덱스가 가리키는 부모 노드 객체입니다.
            # WHY: 이 부모 노드부터 다시 깊이 우선 탐색을 이어나가기 위해서입니다.
            # HOW: parents 튜플에서 idx로 접근합니다.
            p = parents[idx]
            
            # WHAT: 부모 노드의 고유 식별자(메모리 주소)입니다.
            # WHY: 해당 부모 노드가 이전에 이미 방문된 노드인지 중복 검사를 하기 위함입니다.
            # HOW: 내장 함수 id()를 사용합니다.
            pid = id(p)
            
            if pid not in visited:
                # WHAT: 부모 노드의 ID를 방문 완료 집합에 추가합니다.
                # WHY: 나중에 다른 경로를 통해 이 부모 노드에 도달하더라도 다시 처리하지 않기 위해서입니다.
                # HOW: set.add() 메서드를 사용합니다.
                visited.add(pid)
                
                # WHAT: 부모 노드를 스택에 추가하여 다음 반복 시에 처리되도록 합니다.
                # WHY: DFS 특성상 방금 발견한 새로운 노드를 가장 먼저 탐색해야 하기 때문입니다.
                # HOW: 부모 노드와 초기 인덱스 0을 스택에 푸시(append)합니다.
                stack.append((p, 0))
        else:
            # 모든 부모 방문 완료 → post-order 추가
            # WHAT: 부모를 모두 탐색 완료한 노드를 스택에서 제거합니다.
            # WHY: 이 노드에 대한 처리(위상 정렬 순서 확정)가 끝났기 때문입니다.
            # HOW: list.pop()을 호출합니다.
            stack.pop()
            
            # WHAT: 탐색을 마친 노드를 결과 리스트에 추가합니다.
            # WHY: 의존성이 없는 리프 노드부터 순서대로 추가되므로 위상 정렬 순서가 보장됩니다.
            # HOW: topo.append()를 사용합니다.
            topo.append(node)

    # WHAT: 완성된 위상 정렬 리스트를 반환합니다.
    # WHY: 이 리스트 순서대로 텐서 연산을 GPU/CPU에 제출하여 실행하기 위함입니다.
    # HOW: topo 리스트를 리턴합니다.
    return topo


# WHAT: GPU 리소스 해제가 필요한 텐서 핸들(문자열 등)을 임시로 모아두는 큐(집합)입니다.
# WHY: 단일 텐서마다 즉각적으로 리소스를 해제(dispose)하면 오버헤드가 크므로, 모아서 일괄 처리(Batch GC)하기 위함입니다.
# HOW: Python의 set 객체를 전역으로 생성해 중복 핸들 등록을 방지합니다.
_gc_queue: set = set()


_gc_failures: int = 0
_gc_next_retry_at: float = 0.0

def flush_gc(force: bool = False) -> None:
    """
    WHAT: 보류 중인(큐에 쌓인) 리소스 해제 요청들을 모아 JS/WebGPU 브릿지로 일괄 전달하여 처리하는 함수입니다.
    WHY: 성능 최적화를 위해 개별 해제 대신 Batch Dispose를 수행하며, 브릿지 일시 지연 시에도 핸들을 유실(drop)하지 않고 지수 백오프로 재시도하기 위함입니다.
    HOW: 큐에 항목이 있으면 백오프 타임을 체크한 후 js_dispose_batch를 호출하고, 배치 성공 시에만 큐에서 제거합니다.
    """
    global _gc_failures, _gc_next_retry_at
    import time
    import warnings
    
    if not _gc_queue:
        return
        
    now = time.monotonic()
    if not force and now < _gc_next_retry_at:
        return
        
    handles = list(_gc_queue)
    try:
        from .bridge import js_dispose_batch
        js_dispose_batch(handles)
        _gc_queue.difference_update(handles)
        _gc_failures = 0
        _gc_next_retry_at = 0.0
    except Exception as e:
        _gc_failures += 1
        delay = min(2.0 ** _gc_failures, 30.0)
        _gc_next_retry_at = now + delay
        warnings.warn(
            f"[AMEVA GC] disposeBatch failed; keeping {len(_gc_queue)} handles queued for retry in {delay:.1f}s: {e}",
            RuntimeWarning,
            stacklevel=2,
        )


class _HandleCell:
    """
    WHAT: 실제 핸들(문자열 등)을 감싸는 레퍼런스 셀 클래스입니다.
    WHY: C-01 Fix: weakref.finalize가 생성 시점의 handle(None)을 캡처하는 버그 방지를 위해,
         handle을 mutable container에 담아 finalize 시점에 항상 최신 값을 참조하도록 하기 위함입니다.
    HOW: 슬롯(__slots__)을 사용하여 메모리를 절약하고 속성을 단일화(handle)한 클래스를 정의합니다. Reference Cell 패턴 적용.
    """
    __slots__ = ('handle',)

    def __init__(self, handle: Optional[str]) -> None:
        """
        WHAT: HandleCell 객체의 생성자입니다.
        WHY: 객체 생성 시 초기 핸들 값을 저장하기 위해 필요합니다.
        HOW: 전달받은 인자 handle을 self.handle 멤버 변수에 할당합니다.
        """
        self.handle = handle


class Tensor:
    """
    WHAT: AMEVA-Forge의 핵심 데이터 구조인 텐서 클래스입니다.
    WHY: 다차원 배열 데이터를 다루고, 연산 기록을 추적하여 자동 미분 및 GPU 레이지 평가를 지원하기 위함입니다.
    HOW: 내부적으로 shape, dtype, device 등의 메타데이터를 저장하며, 데이터 또는 연산(AST)의 참조를 유지합니다.
    """
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: str,
        device: str,
        requires_grad: bool = False,
        handle: Optional[str] = None,
        data: Optional[np.ndarray] = None,
        op: Optional[str] = None,
        parents: tuple = (),
        op_params: Optional[list] = None
    ):
        """
        WHAT: 텐서 객체를 초기화하는 생성자입니다.
        WHY: 텐서의 형태, 타입, 디바이스 위치 및 연산 히스토리를 설정하기 위함입니다.
        HOW: 전달받은 인자들을 검증하고 멤버 변수들에 할당합니다.
        """
        # WHAT: 텐서의 차원 크기를 나타내는 튜플입니다.
        # WHY: 각 연산에서 형태(Shape) 검증 및 브로드캐스팅에 사용됩니다.
        # HOW: 입력받은 shape 튜플을 할당합니다.
        self.shape = shape

        # PY-H01 Fix: shape 타입 검증
        if not isinstance(shape, tuple):
            raise AMEVAForgeShapeError(f"shape must be a tuple, got {type(shape).__name__}")
        
        # WHAT: shape 튜플의 각 차원(d)을 순회하는 반복문입니다.
        # WHY: 모든 차원이 유효한 양의 정수인지 검증하기 위함입니다.
        # HOW: enumerate로 인덱스와 값을 가져와 int 타입 및 0 이상인지 체크합니다.
        for i, d in enumerate(shape):
            if not isinstance(d, int):
                raise AMEVAForgeShapeError(f"shape[{i}] must be int, got {type(d).__name__}: {d}")
            if d < 0:
                raise AMEVAForgeShapeError(f"shape[{i}] must be non-negative, got {d}")

        # PY-H02 Fix: 빈 텐서(0-element) 차단 — GPU에서 0바이트 버퍼 크래시 방지
        # WHAT: shape 내에 0이 포함되어 있는지 확인하는 제너레이터(암묵적 반복문)입니다.
        # WHY: WebGPU 등에서 0 크기의 버퍼 할당 시 크래시가 발생할 수 있기 때문입니다.
        # HOW: any()와 제너레이터 표현식을 사용하여 0이 있는지 검사합니다.
        if any(d == 0 for d in shape) and len(shape) > 0:
            raise AMEVAForgeShapeError(
                f"Zero-size dimensions are not supported: shape={shape}. "
                f"All dimensions must be positive."
            )

        # Rank 제한: TS MAX_SHAPE_DIM=8과 일치
        if len(shape) > 8:
            raise AMEVAForgeShapeError(
                f"Maximum tensor rank is 8, got {len(shape)}."
            )

        # WHAT: 텐서 데이터의 자료형(예: 'float32')입니다.
        # WHY: 메모리 할당 크기 및 통신 시 타입 일치를 위해 필요합니다.
        # HOW: 인자로 받은 dtype을 할당합니다.
        self.dtype = dtype
        
        # WHAT: 텐서가 저장될 장치('cpu' 또는 'gpu')입니다.
        # WHY: 데이터 저장소 및 연산 수행 장소를 결정하기 위함입니다.
        # HOW: 인자로 받은 device를 할당합니다.
        self.device = device
        
        # WHAT: 자동 미분 추적 여부입니다.
        # WHY: 역전파 그래프에 포함할지를 결정하기 위함입니다.
        # HOW: 불리언 값을 할당합니다.
        self.requires_grad = requires_grad
        
        # WHAT: 역전파를 통해 계산된 이 텐서의 기울기(Gradient)입니다.
        # WHY: 파라미터 업데이트를 위해 기울기를 저장해두어야 하기 때문입니다.
        # HOW: 초기값 None으로 설정됩니다.
        self.grad: Optional['Tensor'] = None

        # --- 내부 상태 ---
        # WHAT: 실제 텐서 핸들(GPU 등)을 간접 참조하기 위한 래퍼 객체입니다.
        # WHY: C-01: handle을 _HandleCell로 감싸 finalizer가 항상 최신 handle을 참조하도록 하기 위함입니다.
        # HOW: _HandleCell 객체를 생성하여 할당합니다.
        self._handle_cell = _HandleCell(handle)
        
        # WHAT: CPU에 저장된 텐서의 실제 데이터(numpy 배열)입니다.
        # WHY: CPU 기반 연산이나 읽어온 데이터를 캐싱하기 위함입니다.
        # HOW: 인자로 받은 data를 할당합니다.
        self._data = data
        
        # WHAT: 텐서 리소스가 명시적으로 해제되었는지 여부 플래그입니다.
        # WHY: 이미 해제된 텐서에 접근하는 것을 막아 안전성을 확보하기 위함입니다.
        # HOW: False로 초기화합니다.
        self._disposed = False
        self._version = 0

        # --- Autograd 상태 ---
        # WHAT: 역전파 시 활용될 컨텍스트 객체입니다.
        # WHY: 순전파(forward) 시에 역전파에 필요한 임시 값들을 저장하기 위함입니다.
        # HOW: None으로 초기화합니다.
        self._ctx: Optional[Any] = None
        
        # WHAT: 이 텐서를 만들어낸 부모 텐서들(레이지 그래프 탐색용)입니다.
        # WHY: 레이지 그래프를 위상 정렬로 탐색하기 위함입니다.
        # HOW: 인자로 받은 parents 튜플을 할당합니다.
        self._parents: tuple = parents
        
        # WHAT: 자동 미분(backward) 전용 부모 텐서 튜플입니다.
        # WHY: NC-05: autograd 그래프와 레이지 AST 그래프의 의존성을 분리하기 위함입니다.
        # HOW: 빈 튜플로 초기화합니다.
        self._grad_parents: tuple = ()
        
        # WHAT: 이 텐서를 생성한 연산(Operation)의 클래스 참조입니다.
        # WHY: 역전파 시 해당 클래스의 backward를 호출하기 위함입니다.
        # HOW: None으로 초기화합니다.
        self._op_cls: Optional[Any] = None

        # --- Lazy 그래프 메타데이터 ---
        # WHAT: 레이지 평가 시 이 텐서를 생성하기 위한 연산 이름입니다.
        # WHY: GPU 컴파일러가 어떤 연산을 수행해야 할지 알기 위함입니다.
        # HOW: 인자로 받은 op 문자열을 할당합니다.
        self._lazy_op = op
        
        # WHAT: 레이지 연산에 필요한 추가 파라미터 리스트입니다.
        # WHY: 차원(axis) 축 등 연산별 고유 옵션을 저장하기 위함입니다.
        # HOW: 인자로 받은 op_params를 할당합니다.
        self._lazy_params = op_params

        # WHAT: GPU 장치인 경우 텐서 파괴 시 가비지 컬렉션을 수행하는 코드 블록입니다.
        # WHY: 파이썬 객체 소멸 시 메모리 누수를 방지하고 GPU 리소스도 해제하기 위함입니다.
        # HOW: weakref.finalize를 사용해 콜백을 등록합니다.
        self._finalizer_registered = False
        if self.device == "gpu":
            import weakref
            weakref.finalize(self, Tensor._finalize_buffer, self._handle_cell)
            self._finalizer_registered = True

    @property
    def _handle(self) -> Optional[str]:
        # WHAT: 내부 래퍼(HandleCell)에서 실제 핸들 값을 가져옵니다.
        # WHY: 텐서의 고유 ID(GPU 상의 버퍼 포인터 등)를 확인하기 위함입니다.
        # HOW: _handle_cell의 handle 속성을 반환합니다.
        return self._handle_cell.handle

    @_handle.setter
    def _handle(self, value: Optional[str]) -> None:
        # WHAT: 내부 래퍼에 새로운 핸들 값을 설정합니다.
        # WHY: 연산 결과로 새로운 버퍼가 할당되었을 때 참조를 갱신하기 위함입니다.
        # HOW: _handle_cell의 handle 속성에 값을 대입합니다.
        self._handle_cell.handle = value

    @staticmethod
    def _finalize_buffer(cell: '_HandleCell') -> None:
        """
        WHAT: 가비지 컬렉터에 의해 호출되는 리소스 해제 콜백 함수입니다.
        WHY: 텐서 객체가 메모리에서 지워질 때 연결된 GPU 버퍼도 해제하여 메모리 누수를 방지하기 위함입니다.
        HOW: cell에 저장된 핸들 문자열을 _gc_queue에 추가하여 일괄 해제(Batch GC)를 준비합니다.
        """
        handle = cell.handle
        if handle is not None:
            _gc_queue.add(handle)
            cell.handle = None
            if len(_gc_queue) >= 16:
                flush_gc()

    def _check_disposed(self) -> None:
        # WHAT: 텐서가 이미 해제되었는지 검사하는 내부 함수입니다.
        # WHY: 해제된 메모리에 접근해 발생하는 크래시(Use-After-Free)를 방지하기 위함입니다.
        # HOW: _disposed 플래그가 True이면 예외를 발생시킵니다.
        if self._disposed:
            raise AMEVAForgeDisposedError("Cannot access a disposed Tensor.")

    async def realize(self) -> None:
        """
        WHAT: 레이지 평가(Lazy Evaluation) 그래프를 단일 FFI 호출로 GPU에 제출(Submit)하는 함수입니다.
        WHY: 연산들을 모았다가 한 번에 수행하여 커널 호출 오버헤드를 줄이고 최적화 기회를 얻기 위함입니다.
        HOW: 위상 정렬된 노드들을 순회하며 명령(instruction) 목록을 만들고 브릿지를 통해 JS로 전달합니다.
        """
        if len(_gc_queue) > 0:
            flush_gc()
        flush_gc()
        if self.device == "cpu" or self._handle is not None:
            return

        # WHAT: 현재 텐서를 계산하기 위한 위상 정렬된 노드 리스트입니다.
        # WHY: 의존성이 없는 순서대로 노드를 처리해야 에러 없이 그래프를 빌드할 수 있기 때문입니다.
        # HOW: build_lazy_topo 함수를 호출합니다.
        topo = build_lazy_topo(self)
        from .graph import GraphBuilder
        # WHAT: JS로 넘길 연산 명령어들을 모아주는 빌더 객체입니다.
        # WHY: 복잡한 연산 그래프를 브릿지가 이해할 수 있는 평탄화된 배열 포맷으로 변환하기 위함입니다.
        # HOW: GraphBuilder 클래스의 인스턴스를 생성합니다.
        builder = GraphBuilder()

        # WHAT: 파이썬 객체의 id를 빌더의 내부 노드 ID와 매핑하는 딕셔너리입니다.
        # WHY: 텐서 간의 참조 관계를 빌더 내의 정수형 ID 기반 참조로 변환하기 위함입니다.
        # HOW: 빈 딕셔너리를 생성한 뒤 반복문에서 값을 채웁니다.
        node_id_map: dict = {}
        
        # WHAT: 위상 정렬된 텐서 목록을 순회하여 연산 그래프를 빌드하는 반복문입니다.
        # WHY: 브릿지(C/WASM)로 넘길 명령 스트림을 순서대로 생성하기 위함입니다.
        # HOW: topo 리스트 내의 각 텐서(v)를 확인하여 분기 처리합니다.
        for v in topo:
            if v._handle is not None:
                # WHAT: 노드를 불러오기(load) 위한 정수형 식별자입니다.
                # WHY: 이미 할당된 텐서를 참조하여 연산을 수행하기 위함입니다.
                # HOW: builder.add_load를 호출하여 반환된 ID를 저장합니다.
                nid = builder.add_load(v.shape, v._handle)
                node_id_map[id(v)] = nid
            elif v._lazy_op == 'upload':
                nid = builder.add_upload(v.shape, v._data)
                node_id_map[id(v)] = nid
            else:
                # WHAT: 현재 연산의 입력으로 사용될 부모 노드들의 ID 리스트입니다.
                # WHY: builder가 의존성 있는 이전 명령들을 참조할 수 있게 하기 위함입니다.
                # HOW: 부모 텐서를 순회하며 매핑된 ID를 모읍니다.
                in_ids = []
                # WHAT: 부모 텐서들을 순회하는 반복문입니다.
                # WHY: 모든 입력의 식별자를 추출하기 위함입니다.
                # HOW: v._parents 튜플을 반복합니다.
                for p in v._parents:
                    if id(p) not in node_id_map:
                        raise AMEVAForgeDeviceError(
                            f"Lazy graph build failed: parent tensor (op={p._lazy_op!r}) "
                            f"is not in the computation graph. It may have been disposed."
                        )
                    in_ids.append(node_id_map[id(p)])
                nid = builder.add_op(v._lazy_op, v.shape, in_ids, v._lazy_params)
                node_id_map[id(v)] = nid

        # WHAT: JS로 보낼 연산 명령어 배열과 추가 입력 데이터입니다.
        # WHY: 런타임에서 그래프를 재구성하여 커널을 실행하기 위한 최종 데이터 형태이기 때문입니다.
        # HOW: builder.compile()을 호출해 튜플 형태로 받습니다.
        instructions, inputs = builder.compile()
        from .bridge import js_execute_graph

        # WHAT: JS 브릿지가 연산을 실행한 뒤 반환한 새로운 텐서 핸들(버퍼 ID)들입니다.
        # WHY: 생성된 텐서 결과를 파이썬 텐서 객체와 연결(binding)하기 위함입니다.
        # HOW: js_execute_graph 함수를 호출하여 딕셔너리로 반환받습니다.
        out_handles = await js_execute_graph(instructions, inputs)

        # WHAT: 생성된 핸들들을 각 텐서 객체에 주입하는 반복문입니다.
        # WHY: 레이지(지연) 상태였던 텐서들이 이제 실제 GPU 버퍼를 가리키게 하기 위함입니다.
        # HOW: topo를 다시 순회하며 out_handles 맵에서 식별자를 찾아 할당합니다.
        for v in topo:
            if v._handle is not None:
                continue
            nid = node_id_map[id(v)]
            # WHAT: JS가 반환한 개별 텐서의 실제 문자열 핸들입니다.
            # WHY: 파이썬 쪽 텐서 객체에 이 값을 심어주기 위함입니다.
            # HOW: 반환된 딕셔너리에서 nid 키로 조회합니다.
            h = out_handles.get(str(nid)) or out_handles.get(nid)
            if h is None:
                raise AMEVAForgeDeviceError(
                    f"Failed to retrieve valid tensor handle from JS for node {nid} "
                    f"(op={v._lazy_op!r}). The JS graph executor may have failed silently."
                )
            v._handle = h
            if v._lazy_op == 'upload':
                v._data = None
                
            v._parents = ()
            v._lazy_op = None

    def numpy(self) -> np.ndarray:
        """
        WHAT: CPU 텐서의 데이터를 동기적으로 반환하는 함수입니다.
        WHY: 텐서 안의 실제 값(수치)을 확인하거나 디버깅, 외부 라이브러리(numpy 기반)에 데이터를 넘기기 위함입니다.
        HOW: 장치가 'cpu'인지 확인한 뒤 저장된 내부 _data 배열을 반환합니다.
        """
        self._check_disposed()
        if self.device == "cpu":
            if self._data is None:
                raise AMEVAForgeDisposedError("CPU tensor data has been released.")
            return self._data
        else:
            raise AMEVAForgeDeviceError(
                "GPU tensor readback is asynchronous. Use: data = await tensor.numpy_async()"
            )

    async def numpy_async(self) -> np.ndarray:
        """
        WHAT: GPU 텐서 데이터를 비동기로 읽어오는 함수입니다.
        WHY: GPU에서 CPU로 메모리를 복사하는 작업은 메인 스레드를 블로킹할 수 있으므로 비동기적으로 처리하기 위함입니다.
        HOW: JS/WebGPU의 비동기 버퍼 맵핑 기능을 활용해 완료를 대기(await)한 후 데이터를 복사합니다.
        """
        self._check_disposed()
        if self.device == "cpu":
            if self._data is None:
                raise AMEVAForgeDisposedError("CPU tensor data has been released.")
            return self._data

        # 1. 레이지 그래프를 GPU에 제출 (동기 submit -> 이제 async)
        await self.realize()

        from .bridge import js_map_async, js_read_mapped_into
        # 2. GPU 큐 완료 대기 + staging 버퍼 맵핑
        await js_map_async(self._handle)

        # 3. WASM 힙에 직접 읽어들이기
        # WHAT: 데이터를 수신할 빈 numpy 배열입니다.
        # WHY: GPU로부터 복사해올 데이터를 담아둘 메모리 공간을 미리 준비하기 위함입니다.
        # HOW: np.empty를 사용하여 텐서와 동일한 크기(shape)의 float32 배열을 할당합니다.
        out = np.empty(self.shape, dtype=np.float32)
        js_read_mapped_into(self._handle, out)

        return out

    def to(self, device: str) -> 'Tensor':
        """
        WHAT: CPU Tensor를 GPU lazy-upload Tensor로 이동하거나 장치를 변경합니다.
        WHY: 텐서 데이터를 GPU 메모리로 업로드하기 위해 지연 업로드 노드를 생성합니다.
        HOW: 새로운 GPU 장치 Tensor 객체를 생성하여 반환합니다.
        """
        self._check_disposed()
        if device not in ("cpu", "gpu"):
            raise AMEVAForgeDeviceError(f"Unsupported device: {device}")
        if device == self.device:
            return self
        if device == "cpu":
            if self._handle is not None:
                raise AMEVAForgeDeviceError(
                    "GPU to CPU transfer is asynchronous. Use await tensor.numpy_async()."
                )
            if self._data is not None:
                return Tensor(
                    shape=self.shape,
                    dtype=self.dtype,
                    device="cpu",
                    requires_grad=self.requires_grad,
                    data=self._data.astype(np.float32, copy=True),
                )
            raise AMEVAForgeDeviceError("Cannot move unallocated GPU tensor to CPU synchronously")
        if self._data is None:
            raise AMEVAForgeDeviceError("CPU tensor has no uploadable data")

        return Tensor(
            shape=self.shape,
            dtype=self.dtype,
            device="gpu",
            requires_grad=self.requires_grad,
            data=self._data.astype(np.float32, copy=True),
            op="upload",
            parents=(),
            op_params=[],
        )

    def move_to_(self, device: str) -> 'Tensor':
        """
        WHAT: Tensor의 내부 저장소(Storage/Device)를 in-place로 대상 장치로 마이그레이션합니다.
        WHY: Module.to(device) 호출 시 Parameter 객체 참조(Identity)를 보존하여,
             Optimizer 생성 후 model.to('gpu')를 호출해도 가중치 업데이트 참조가 단절되지 않도록 하기 위함입니다.
        HOW: self.to(device)로 상태를 생성한 후, self의 내부 속성들을 in-place 덮어쓰고 moved의 소유권을 박탈합니다.
        """
        self._check_disposed()
        if device == self.device:
            return self

        moved = self.to(device)

        # 기존 GPU 핸들 정리
        if self.device == "gpu" and self._handle is not None and self._handle != moved._handle:
            try:
                self.dispose()
                self._disposed = False
            except Exception:
                pass

        # moved의 내부 상태를 self로 in-place 이전
        self._data = moved._data
        self._handle = moved._handle
        if hasattr(self, "_handle_cell"):
            self._handle_cell.handle = moved._handle

        self._lazy_op = getattr(moved, "_lazy_op", None)
        self._op = getattr(moved, "_op", None)
        self._parents = getattr(moved, "_parents", ())
        self._op_params = getattr(moved, "_op_params", None)
        self.shape = moved.shape
        self.dtype = moved.dtype
        self.device = moved.device
        self._version += 1

        if self.device == "gpu" and not getattr(self, "_finalizer_registered", False):
            import weakref
            weakref.finalize(self, Tensor._finalize_buffer, self._handle_cell)
            self._finalizer_registered = True

        # moved가 동일 핸들을 중복 해제하지 않도록 소유권 박탈
        moved._handle = None
        if hasattr(moved, "_handle_cell"):
            moved._handle_cell.handle = None
        moved._data = None

        return self

    def dispose(self) -> None:
        """
        WHAT: 텐서와 연결된 리소스(GPU 버퍼 및 내부 데이터)를 즉시 해제하는 함수입니다.
        WHY: 더 이상 사용하지 않는 메모리를 명시적으로 반환하여 VRAM 초과(OOM) 오류를 막기 위함입니다.
        HOW: 핸들을 GC 큐에 넣고 내부 데이터 참조와 그래프 연결을 모두 초기화(None/빈 튜플)합니다.
        """
        if self._disposed:
            return
        if self.device == "gpu" and self._handle is not None:
            _gc_queue.add(self._handle)
            self._handle = None
            flush_gc()

        self._data = None
        self._disposed = True
        self._lazy_op = None
        self._parents = ()
        self._grad_parents = ()
        self._ctx = None

    def backward(self, gradient: Optional['Tensor'] = None, retain_graph: bool = False) -> None:
        """
        WHAT: 역전파(Backpropagation)를 수행하여 이 텐서에 기여한 모든 부모 텐서들의 기울기(gradient)를 계산하는 함수입니다.
        WHY: 신경망을 학습시킬 때 손실 함수(Loss)로부터 파라미터 업데이트에 필요한 미분값을 구하기 위해서입니다.
        HOW: 위상 정렬의 역순으로 그래프를 탐색하며, 연산 클래스의 backward 함수에 체인 룰(Chain Rule)을 적용합니다.
        """
        self._check_disposed()
        if retain_graph:
            from .errors import AMEVAForgeUnsupportedOperationError
            raise AMEVAForgeUnsupportedOperationError(
                "retain_graph=True is outside Release 1"
            )
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require grad.")

        from .autograd import build_topological_sort
        from .ops import ones_like

        if gradient is None:
            if self.shape != () and self.shape != (1,):
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            else:
                gradient = ones_like(self)

        # WHAT: 각 텐서 노드의 ID를 키로 하여 해당 노드까지 누적된 기울기를 저장하는 맵입니다.
        # WHY: 중복된 경로를 통해 전달되는 기울기들을 하나로 모아(add) 기록하기 위함입니다.
        # HOW: 현재 노드(루트)에 대한 기울기를 초기화하여 딕셔너리에 넣습니다.
        grads: dict = {id(self): gradient}

        # WHAT: 순전파 시 생성된 그래프를 탐색하기 위한 위상 정렬된 노드 리스트입니다.
        # WHY: 계산 순서를 보장해야 부모 노드로 기울기를 올바르게 전달할 수 있기 때문입니다.
        # HOW: autograd 모듈의 함수를 호출합니다.
        topo = build_topological_sort(self)
        
        # WHAT: 정렬된 노드를 역방향으로 순회하며 역전파를 수행하는 반복문입니다.
        # WHY: 체인 룰에 따라 최종 출력(현재 텐서)에서부터 입력(부모) 방향으로 기울기를 흘려보내야 하기 때문입니다.
        # HOW: reversed() 함수를 통해 리스트의 순서를 뒤집어 반복합니다.
        for v in reversed(topo):
            if not getattr(v, 'requires_grad', False):
                continue
            if id(v) not in grads:
                continue

            # WHAT: 현재 노드(v)에 전달된 누적 기울기입니다.
            # WHY: 부모 노드들의 기울기를 구하기 위해 이 값을 곱해야(Chain Rule) 하기 때문입니다.
            # HOW: grads 맵에서 꺼내오고 메모리 해제를 위해 pop합니다.
            grad_out = grads.pop(id(v))

            if v._ctx is None or v._op_cls is None:
                if v.grad is None:
                    v.grad = grad_out
                else:
                    from .ops import add
                    v.grad = add(v.grad, grad_out)
                continue

            v._ctx.validate_saved_tensor_versions()
            grad_inputs = v._op_cls.backward(v._ctx, grad_out)
            v._ctx.saved_tensors = ()
            v._ctx.saved_versions = ()
            if not isinstance(grad_inputs, tuple):
                grad_inputs = (grad_inputs,)

            # WHAT: 계산된 부모 기울기들을 각 부모 노드에 전달하는 반복문입니다.
            # WHY: 부모 노드의 ID 맵(grads)에 값을 누적(add)하기 위함입니다.
            # HOW: _grad_parents 튜플과 grad_inputs를 zip으로 묶어 순회합니다.
            for parent, g in zip(v._grad_parents, grad_inputs):
                if not getattr(parent, 'requires_grad', False) or g is None:
                    continue
                # WHAT: 부모 노드의 메모리 주소(ID)입니다.
                # WHY: grads 딕셔너리에 접근하는 식별자로 사용하기 위함입니다.
                # HOW: id() 함수를 사용합니다.
                pid = id(parent)
                if pid in grads:
                    from .ops import add
                    grads[pid] = add(grads[pid], g)
                else:
                    grads[pid] = g
        grads.clear()



    def relu(self) -> 'Tensor':
        """
        WHAT: 현재 텐서에 ReLU(Rectified Linear Unit) 활성화 함수를 적용합니다.
        WHY: 신경망에 비선형성을 부여하여 복잡한 패턴을 학습할 수 있게 하기 위함입니다.
        HOW: ops 모듈의 relu 연산을 호출하여 결과를 반환합니다.
        """
        self._check_disposed()
        from .ops import relu
        return relu(self)

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        WHAT: 현재 텐서와 다른 텐서 간의 행렬 곱셈을 수행합니다.
        WHY: 선형 계층(Linear Layer) 등에서 가중치와의 곱 연산을 처리하기 위함입니다.
        HOW: ops 모듈의 matmul 함수를 호출하여 결과를 반환합니다.
        """
        self._check_disposed()
        from .ops import matmul
        return matmul(self, other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        WHAT: 파이썬의 `@` 연산자를 오버로딩하여 행렬 곱셈을 수행합니다.
        WHY: 사용자가 `a @ b`처럼 직관적으로 수학 기호를 사용할 수 있게 하기 위함입니다.
        HOW: 내부적으로 self.matmul()을 호출합니다.
        """
        return self.matmul(other)

    def __add__(self, other):
        """
        WHAT: 두 텐서 간의 덧셈 연산(`+`)을 수행합니다.
        WHY: 텐서들 간의 요소별 덧셈을 직관적으로 지원하기 위함입니다.
        HOW: 스칼라 값일 경우 () 크기의 스칼라 텐서로 변환하여 0-stride 브로드캐스팅으로 ops.add를 호출합니다.
        """
        self._check_disposed()
        from .ops import add, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return add(self, other)

    def __radd__(self, other):
        """
        WHAT: 우측 피연산자 기준의 덧셈 연산(`스칼라 + 텐서`)을 수행합니다.
        WHY: 파이썬 내장 스칼라 타입이 왼쪽에 올 때도 덧셈이 동작하도록 지원하기 위함입니다.
        HOW: 덧셈은 교환 법칙이 성립하므로 self.__add__(other)를 그대로 반환합니다.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        WHAT: 두 텐서 간의 뺄셈 연산(`-`)을 수행합니다.
        WHY: 텐서 요소들 간의 차이를 계산하기 위함입니다.
        HOW: 스칼라를 () 스칼라 텐서로 변환한 뒤 0-stride 브로드캐스팅으로 ops.sub를 호출합니다.
        """
        self._check_disposed()
        from .ops import sub, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return sub(self, other)

    def __rsub__(self, other):
        """
        WHAT: 우측 피연산자 기준의 뺄셈 연산(`스칼라 - 텐서`)을 수행합니다.
        WHY: 스칼라가 왼쪽에 올 경우 순서에 맞게 뺄셈을 적용하기 위함입니다.
        HOW: other를 () 스칼라 텐서로 변환한 뒤 other에서 self를 빼는 ops.sub를 호출합니다.
        """
        self._check_disposed()
        from .ops import sub, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return sub(other, self)

    def __mul__(self, other):
        """
        WHAT: 두 텐서 간의 요소별 곱셈 연산(`*`)을 수행합니다.
        WHY: 아다마르 곱(Hadamard Product)이나 스칼라 배율을 적용하기 위함입니다.
        HOW: 스칼라를 () 스칼라 텐서로 변환 후 0-stride 브로드캐스팅으로 ops.mul을 호출합니다.
        """
        self._check_disposed()
        from .ops import mul, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return mul(self, other)

    def __rmul__(self, other):
        """
        WHAT: 우측 피연산자 기준의 곱셈 연산(`스칼라 * 텐서`)을 수행합니다.
        WHY: 곱셈의 교환 법칙을 지원하여 스칼라가 왼쪽에 와도 처리되게 하기 위함입니다.
        HOW: self.__mul__(other)를 호출하여 결과를 반환합니다.
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        WHAT: 두 텐서 간의 나눗셈 연산(`/`)을 수행합니다.
        WHY: 텐서 요소별 나눗셈을 수식으로 간편하게 표현하기 위함입니다.
        HOW: 스칼라를 () 스칼라 텐서로 변환 후 0-stride 브로드캐스팅으로 ops.div를 호출합니다.
        """
        self._check_disposed()
        from .ops import div, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return div(self, other)

    def __rtruediv__(self, other):
        """
        WHAT: 우측 피연산자 기준의 나눗셈 연산(`스칼라 / 텐서`)을 수행합니다.
        WHY: 스칼라를 텐서의 각 요소로 나누는 연산을 지원하기 위함입니다.
        HOW: 스칼라를 () 스칼라 텐서로 바꾼 후 other를 self로 나누는 ops.div를 호출합니다.
        """
        self._check_disposed()
        from .ops import div, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return div(other, self)

    def __neg__(self):
        """
        WHAT: 텐서의 부호를 반전시키는 단항 연산(`-텐서`)을 수행합니다.
        WHY: 수식 내에서 값의 부호를 쉽게 바꾸기 위함입니다.
        HOW: ops.neg 함수를 호출하여 결과를 반환합니다.
        """
        self._check_disposed()
        from .ops import neg
        return neg(self)



    def sum(self):
        """
        WHAT: 텐서 내 모든 요소들의 합을 계산합니다.
        WHY: 손실(Loss) 합산이나 특정 차원의 데이터를 집계하기 위함입니다.
        HOW: ops.sum_op 함수를 호출하여 스칼라(또는 축소된 텐서)를 반환합니다.
        """
        self._check_disposed()
        from .ops import sum_op
        return sum_op(self)

    def mean(self):
        """
        WHAT: 텐서 내 모든 요소들의 평균을 계산합니다.
        WHY: MSE 손실 계산 등에서 전체 데이터의 평균적인 크기를 구하기 위함입니다.
        HOW: ops.mean_op 함수를 호출하여 결과를 반환합니다.
        """
        self._check_disposed()
        from .ops import mean_op
        return mean_op(self)

    def reshape(self, *shape):
        """
        WHAT: 텐서의 차원 형태(Shape)를 변경합니다.
        WHY: 데이터의 논리적 구조를 재배열하여 다른 연산(행렬곱 등)과 호환되게 하기 위함입니다.
        HOW: 새로운 shape 튜플을 구성하고 ops.reshape를 호출합니다.
        """
        self._check_disposed()
        from .ops import reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def view(self, *shape):
        """
        WHAT: reshape와 동일하게 텐서의 형태를 변경합니다.
        WHY: 기존 파이토치(PyTorch) 코드와의 호환성을 유지하여 사용 편의성을 높이기 위함입니다.
        HOW: 내부적으로 self.reshape를 호출합니다.
        """
        return self.reshape(*shape)

    def numel(self):
        """
        WHAT: 텐서가 포함하는 전체 원소(Element)의 개수를 반환합니다.
        WHY: 데이터의 총 크기를 파악하거나 평균 계산 시 나누는 값으로 사용하기 위함입니다.
        HOW: 텐서의 shape 튜플을 순회하며(loop) 각 차원의 크기를 곱합니다.
        """
        # WHAT: 원소 개수를 누적해서 곱할 변수입니다.
        # WHY: 1부터 시작하여 각 차원을 곱해가기 위함입니다.
        # HOW: 정수 1로 초기화합니다.
        n = 1
        # WHAT: shape의 각 차원 크기를 순회하는 반복문입니다.
        # WHY: 모든 차원의 크기를 곱하여 총 볼륨을 구하기 위함입니다.
        # HOW: self.shape를 순회하여 n에 곱합니다.
        for d in self.shape:
            n *= d
        return n

    @property
    def data(self):
        """
        WHAT: 현재 텐서 객체 자체를 반환합니다 (데이터 속성 접근용).
        WHY: 파이토치 스타일의 tensor.data 접근 패턴을 모방하여 기존 코드 호환성을 제공하기 위함입니다.
        HOW: self를 그대로 반환합니다.
        """
        return self

    @data.setter
    def data(self, new_tensor):
        """
        WHAT: 텐서의 내부 데이터와 상태를 다른 텐서의 것으로 교체(In-place replacement)합니다.
        WHY: 옵티마이저(Optimizer)가 파라미터를 업데이트할 때 새로운 텐서를 할당하지 않고 제자리에서 값을 바꾸기 위함입니다.
        HOW: 새 텐서의 데이터, 핸들, 형태를 복사하고, 기존의 자동 미분 그래프 연결(부모, 컨텍스트 등)을 모두 끊습니다.
        """
        # In-place replacement for optimizer updates
        self._data = new_tensor._data if hasattr(new_tensor, '_data') else None
        self.shape = new_tensor.shape
        self._handle = getattr(new_tensor, '_handle', None)
        self._parents = ()
        self._grad_parents = ()  # VUL-005 Fix: autograd 그래프 참조 해제
        self._ctx = None         # VUL-005 Fix: backward context 해제
        self._op_cls = None      # VUL-005 Fix: op class 해제
        self._grad_fn = None
        self.grad = None
        
    def exp(self):
        """
        WHAT: 텐서의 각 요소에 지수 함수(e^x)를 적용합니다.
        WHY: 소프트맥스(Softmax) 등의 활성화 함수나 로그 확정값을 원래 스케일로 복원하기 위함입니다.
        HOW: ops.exp_op를 호출합니다.
        """
        self._check_disposed()
        from .ops import exp_op
        return exp_op(self)
        
    def log(self):
        """
        WHAT: 텐서의 각 요소에 자연 로그(ln x)를 적용합니다.
        WHY: 크로스 엔트로피 손실(Cross Entropy Loss) 등 정보량 기반 수식 처리를 위함입니다.
        HOW: ops.log_op를 호출합니다.
        """
        self._check_disposed()
        from .ops import log_op
        return log_op(self)
        
    def sigmoid(self):
        """
        WHAT: 시그모이드(Sigmoid) 활성화 함수를 적용합니다.
        WHY: 출력값을 0과 1 사이의 확률값으로 변환하기 위함입니다.
        HOW: ops.sigmoid를 호출합니다.
        """
        self._check_disposed()
        from .ops import sigmoid
        return sigmoid(self)
        
    def tanh(self):
        """
        WHAT: 하이퍼볼릭 탄젠트(Tanh) 활성화 함수를 적용합니다.
        WHY: 출력값을 -1과 1 사이로 정규화하여 기울기 소실을 완화하기 위함입니다.
        HOW: ops.tanh_op를 호출합니다.
        """
        self._check_disposed()
        from .ops import tanh_op
        return tanh_op(self)

    def unsqueeze(self, dim: int):
        """
        WHAT: 지정한 차원(dim) 위치에 크기가 1인 새로운 차원을 삽입합니다.
        WHY: 브로드캐스팅(Broadcasting)을 위해 차원 수를 늘리거나 배치 차원을 추가하기 위함입니다.
        HOW: ops.unsqueeze를 호출합니다.
        """
        self._check_disposed()
        from .ops import unsqueeze
        return unsqueeze(self, dim)
        
    def squeeze(self, dim: Optional[int] = None):
        """
        WHAT: 크기가 1인 차원을 제거합니다. dim 지정 시 해당 차원만 제거합니다.
        WHY: 불필요한 차원을 축소시켜 데이터 구조를 단순화하거나 차원 수를 맞추기 위함입니다.
        HOW: ops.squeeze를 호출합니다.
        """
        self._check_disposed()
        from .ops import squeeze
        return squeeze(self, dim)
        
    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        """
        WHAT: 다차원 텐서를 1차원 배열(또는 지정된 차원 구간 축소)로 평탄화합니다.
        WHY: 합성곱(CNN) 층의 출력을 선형(Linear) 층에 전달하기 위해 1차원 벡터로 펴주어야 하기 때문입니다.
        HOW: ops.flatten을 호출합니다.
        """
        self._check_disposed()
        from .ops import flatten
        return flatten(self, start_dim, end_dim)
        
    def permute(self, dims: tuple):
        """
        WHAT: 텐서의 차원 순서를 지정된 배열(dims)대로 재배치합니다.
        WHY: 이미지 데이터의 채널 순서 변경(NHWC <-> NCHW)이나 행렬 전치 등을 수행하기 위함입니다.
        HOW: ops.permute를 호출합니다.
        """
        self._check_disposed()
        from .ops import permute
        return permute(self, dims)

    def max(self, axis=None):
        """
        WHAT: 텐서 내 요소들의 최댓값을 구합니다. 특정 축(axis)이 주어지면 해당 축을 따라 최댓값을 구합니다.
        WHY: 예측 클래스를 선택(Argmax 역할)하거나 풀링 연산을 수행하기 위함입니다.
        HOW: axis 여부에 따라 ops.max_op 또는 ops.max_axis를 호출합니다.
        """
        self._check_disposed()
        if axis is None:
            from .ops import max_op
            return max_op(self)
        else:
            from .ops import max_axis
            return max_axis(self, axis)

    def var(self, axis=None, unbiased=True):
        """
        WHAT: 텐서의 분산(Variance)을 계산합니다.
        WHY: 데이터의 분포 범위를 파악하거나 정규화(Normalization) 과정에서 쓰이기 때문입니다.
        HOW: ops.var를 호출합니다.
        """
        self._check_disposed()
        from .ops import var
        return var(self, axis, unbiased)
        
    def std(self, axis=None, unbiased=True):
        """
        WHAT: 텐서의 표준 편차(Standard Deviation)를 계산합니다.
        WHY: 데이터의 퍼짐 정도를 원래 단위로 확인하거나 표준화 기법에서 분모로 사용하기 위함입니다.
        HOW: ops.std를 호출합니다.
        """
        self._check_disposed()
        from .ops import std
        return std(self, axis, unbiased)

    def __getitem__(self, key):
        """
        WHAT: 인덱싱 또는 슬라이싱 문법(예: tensor[0:2])을 사용하여 텐서의 부분 배열을 추출합니다.
        WHY: 특정 데이터 샘플을 선택하거나 관심 영역(ROI)만 잘라내어 처리하기 위함입니다.
        HOW: 파이썬 특수 메서드를 오버로딩하여 ops.slice_op를 호출합니다.
        """
        self._check_disposed()
        from .ops import slice_op
        return slice_op(self, key)

    def __repr__(self) -> str:
        """
        WHAT: 텐서 객체를 문자열로 표현(출력)합니다.
        WHY: 디버깅이나 로그 확인 시 텐서의 크기, 타입, 디바이스 등 메타데이터를 쉽게 확인하기 위함입니다.
        HOW: 포맷팅된 문자열을 생성하여 반환합니다. 해제된 텐서는 별도로 표시합니다.
        """
        if self._disposed:
            return '<AMEVA Tensor (disposed)>'
        return (
            f'<AMEVA Tensor shape={self.shape}, dtype={self.dtype}, '
            f'device={self.device}, requires_grad={self.requires_grad}, '
            f'handle={self._handle}>'
        )

```

---

## `packages/forge-py/src/forge/tensor.pyi`

```typescript
from typing import Any, Sequence, Union, Optional, Tuple
import numpy as np

ArrayLike = Union[Sequence[Any], np.ndarray, float, int]

class Tensor:
    _handle: Optional[str]
    _shape: Tuple[int, ...]
    _dtype: str
    _device: str
    _requires_grad: bool
    _grad: Optional[Tensor]
    _data: Optional[np.ndarray]
    _version: int

    def __init__(
        self,
        data: Optional[ArrayLike] = None,
        shape: Optional[Sequence[int]] = None,
        dtype: str = "float32",
        device: str = "cpu",
        requires_grad: bool = False,
        handle: Optional[str] = None,
    ) -> None: ...

    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def requires_grad(self) -> bool: ...
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None: ...
    @property
    def grad(self) -> Optional[Tensor]: ...
    @grad.setter
    def grad(self, value: Optional[Tensor]) -> None: ...

    def to(self, device: str) -> Tensor: ...
    def backward(self, grad: Optional[Tensor] = None, retain_graph: bool = False) -> None: ...
    def realize(self) -> Tensor: ...
    def dispose(self) -> None: ...
    async def numpy_async(self) -> np.ndarray: ...
    def numpy(self) -> np.ndarray: ...

    def __add__(self, other: Union[Tensor, float, int]) -> Tensor: ...
    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor: ...
    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor: ...
    def __truediv__(self, other: Union[Tensor, float, int]) -> Tensor: ...
    def __neg__(self) -> Tensor: ...
    def __matmul__(self, other: Tensor) -> Tensor: ...
    def relu(self) -> Tensor: ...
    def sum(self) -> Tensor: ...
    def reshape(self, *shape: int) -> Tensor: ...
    def transpose(self, dim0: int = 0, dim1: int = 1) -> Tensor: ...

def tensor(
    data: Optional[ArrayLike] = None,
    shape: Optional[Sequence[int]] = None,
    dtype: str = "float32",
    device: str = "cpu",
    requires_grad: bool = False,
) -> Tensor: ...

```

---

## `packages/forge/src/bridge/pyodideBridge.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * 
 * pyodideBridge.ts — globalThis.amevaForge API 등록자
 *
 * H-02 연동: 단일 실행 경로(graphExecutor.ts)로 통합.
 *   executeGraph 시그니처: (instructionsJson: string, jsInputs: unknown) => Record
 *
 * M-06 연동: disposeBatch 추가 (bridge.py의 js_dispose_batch 지원)
 */

import { init, read, dispose, getTensorInfo, mapBufferAsync, readMappedInto, warmupKernels } from "../tensor/gpuCore";
import { executeGraph } from "../tensor/graphExecutor";
import { _globalRegistry } from "../tensor/tensorRegistry";
import { getQuotaSnapshot } from "../webgpu/quota";
import { getDevice, _safeLog } from "../webgpu/device";
import { clearStagingPool } from "../webgpu/buffers";
import { _globalUniformPool } from "../webgpu/uniformPool";
import { TensorHandle } from "../types";

/**
 * WHAT: 이 인터페이스는 전역 amevaForge 객체의 형태를 정의합니다.
 * WHY: 이 API의 목적은 파이오다이드(Pyodide) 환경의 파이썬 코드에서
 *      자바스크립트/웹어셈블리(WASM) 쪽의 GPU 핵심 기능과 그래프 실행 기능을 호출할 수 있도록 
 *      타입스크립트 브리지(bridge) 역할을 하는 것입니다.
 * HOW: 이 인터페이스를 통해 파이썬이 GPU 메모리 관리 및 연산 실행 관련 함수들에 접근하여
 *      WebGPU 자원을 다룰 수 있도록 구조화합니다.
 */
export interface AmevaTensorGlobalAPI {
  /** WHAT: GPU 코어 초기화 함수. WHY: WebGPU 디바이스를 준비하기 위해. HOW: WebGPU API를 호출해 설정. */
  init: typeof init;
  /** WHAT: 텐서 데이터 읽기 함수. WHY: GPU 메모리 데이터를 메인 메모리로 가져오기 위해. HOW: 비동기로 버퍼 매핑 후 데이터 복사. */
  read: typeof read;
  /** WHAT: 텐서 메모리 해제 함수. WHY: 사용이 끝난 GPU 자원을 반환하기 위해. HOW: WebGPU 버퍼의 destroy 메서드 호출. */
  dispose: typeof dispose;
  /** WHAT: 텐서 메타데이터 조회 함수. WHY: 텐서의 크기, 타입, 상태를 확인하기 위해. HOW: 내부 레지스트리에서 정보 조회. */
  getTensorInfo: typeof getTensorInfo;
  /** WHAT: 비동기 버퍼 매핑 함수. WHY: 데이터를 효율적으로 읽기 위해 매핑 상태로 만들기 위함. HOW: mapAsync를 호출. */
  mapBufferAsync: typeof mapBufferAsync;
  /** WHAT: 매핑된 버퍼를 특정 타입 배열로 읽어오는 함수. WHY: 복사 오버헤드 없이 직접 뷰를 가져오기 위함. HOW: getMappedRange 결과를 TypedArray로 변환. */
  readMappedInto: typeof readMappedInto;
  /** WHAT: 텐서 연산 그래프 실행 함수. WHY: 복잡한 연산들을 순차적으로 GPU에서 수행하기 위함. HOW: JSON 명령어 파싱 후 각 커널 실행. */
  executeGraph: typeof executeGraph;
  /** WHAT: 커널 웜업 함수. WHY: 런타임 성능을 안정화하기 위해 미리 셰이더를 컴파일하기 위함. HOW: 파이프라인을 미리 생성. */
  warmupKernels: typeof warmupKernels;
  
  /** 
   * WHAT: M-06 batch dispose — 여러 텐서 핸들 배열을 한 번에 해제.
   * WHY: 파이썬 쪽에서 여러 개의 텐서를 가비지 컬렉션할 때 단일 호출로 성능을 높이기 위해.
   * HOW: 전달된 배열을 순회하며 개별 dispose를 호출.
   */
  disposeBatch: (handles: TensorHandle[]) => void;
  getQuotaSnapshot: typeof getQuotaSnapshot;
  snapshotHandles: () => string[];
  flushGC: (options?: { strict?: boolean }) => Promise<{ ok: boolean; error?: string }>;
}

declare global {
  /**
   * WHAT: 전역 네임스페이스(globalThis)에 amevaForge 객체를 등록하기 위한 선언입니다.
   * WHY: 브라우저나 워커 환경 어디서든 전역 스코프에서 이 브리지 객체에 접근할 수 있게 하기 위해 존재합니다.
   * HOW: var 키워드를 통해 전역 타입 확장을 수행합니다.
   */
  var amevaForge: AmevaTensorGlobalAPI | undefined;
}

/**
 * WHAT: 여러 개의 텐서 핸들(TensorHandle)을 한 번에 일괄적으로 메모리에서 해제(dispose)합니다.
 * WHY: 파이썬(Pyodide) 환경에서 다수의 텐서 가비지 컬렉션을 효율적으로 처리하기 위해 존재합니다. (단일 호출로 오버헤드 감소)
 * HOW: 반복문(for...of)을 통해 각 핸들마다 GPU 메모리 해제를 시도하며, 이미 해제된 텐서의 에러는 조용히 무시하여 중단되지 않도록 처리합니다.
 * 
 * @param handles 해제할 텐서 핸들들의 배열
 */
function disposeBatch(handles: TensorHandle[]): void {
  /**
   * WHAT: 입력받은 핸들 배열을 순회하는 반복문입니다.
   * WHY: 각각의 텐서 리소스에 대해 개별적인 해제 절차가 필요하기 때문에 존재합니다.
   * HOW: for...of 구문을 사용하여 handles 배열의 각 원소(handle)를 하나씩 가져와 내부 블록을 실행합니다.
   */
  for (const handle of handles) {
    /** 
     * WHAT: 현재 순회 중인 텐서 핸들이 유효한 값(truthy)인지 확인하는 조건문입니다.
     * WHY: null, undefined 혹은 빈 문자열 같은 잘못된 핸들이 전달되어 불필요한 예외나 시스템 오류가 발생하는 것을 방지하기 위함입니다.
     * HOW: 자바스크립트의 truthy 평가를 통해 handle 값이 존재할 때만 내부의 해제 로직(try-catch 블록)을 수행하도록 제어합니다.
     */
    if (handle) {
      try { 
        dispose(handle); 
      } catch (e) { 
        _safeLog(`[pyodideBridge] disposeBatch handle "${handle}" failed: ${e}`);
      }
    }
  }
}

/**
 * WHAT: Pyodide가 자바스크립트 기능에 접근할 수 있도록 전역 `globalThis.amevaForge` 객체를 생성하고 등록합니다.
 * WHY: 파이썬 측 브리지 코드가 WASM을 거쳐 GPU 하드웨어 가속(WebGPU 등) 기능과 그래프 실행 로직을 사용할 수 있게 하는 엔트리 포인트가 필요하기 때문입니다.
 * HOW: 필요한 모든 내부 함수들을 모은 api 객체를 만들고 Object.freeze로 동결시킨 뒤, globalThis의 속성으로 할당하여 전역에서 접근 가능하게 만듭니다.
 * 
 * @returns 등록된 전역 API 객체
 */
export function registerPyodideBridge(): AmevaTensorGlobalAPI {
  /**
   * WHAT: 실제로 전역에 노출될 API 객체를 구성하는 변수입니다.
   * WHY: 각 기능(init, read 등)들을 하나의 통일된 인터페이스 객체로 모아서 파이썬 측에서 구조화된 방식으로 쉽게 접근할 수 있게 묶어주기 위함입니다.
   * HOW: AmevaTensorGlobalAPI 타입에 맞추어 내부 모듈에서 임포트한 함수들을 프로퍼티로 할당하여 객체 리터럴을 생성합니다.
   */
  const api: AmevaTensorGlobalAPI = {
    init,
    read,
    dispose,
    getTensorInfo,
    mapBufferAsync,
    readMappedInto,
    executeGraph,
    warmupKernels,
    disposeBatch,
    getQuotaSnapshot,
    snapshotHandles: () => _globalRegistry.snapshotHandles(),
    flushGC: async (options?: { strict?: boolean }) => {
      try {
        const dev = getDevice();
        await dev.queue.onSubmittedWorkDone();
        await _globalUniformPool.retireSubmitted(dev);
        clearStagingPool();
        _globalUniformPool.clear();
        return { ok: true };
      } catch (e: any) {
        _safeLog(`[pyodideBridge] flushGC work done error: ${e}`);
        if (options && options.strict) {
          throw e;
        }
        return { ok: false, error: String(e) };
      }
    },
  };

  Object.freeze(api); // F-014 Fix: API 객체 동결하여 외부 변조 방지
  globalThis.amevaForge = api;
  return api;
}

```

---

## `packages/forge/src/bridge/safeCopy.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * 
 * safeCopy.ts — Pyodide PyProxy → Float32Array 안전 변환
 *
 * H-05 Fix: ensureFloat32Array에서 불필요한 new Float32Array(jsView) deep copy 제거.
 *   Float32Array가 이미 WASM 힙을 가리키고 있으면 그대로 반환 (Zero-Copy).
 *   복사가 실제로 필요한 경우에만 cloneToFloat32Array()를 명시적으로 호출.
 */

/**
 * WHAT: Pyodide의 PyProxy 객체(또는 그와 유사한 구조의 객체)가 가지는 형태를 정의하는 인터페이스입니다.
 * WHY: 자바스크립트 측에서 파이썬 객체의 데이터를 가져오기 위한 `toJs()` 메서드의 존재를 명시하여 컴파일 타임에 타입 안정성을 높이기 위해 존재합니다.
 * HOW: 타입스크립트 인터페이스로 선언하여 `toJs` 속성이 반환값을 알 수 없는 함수임을 규정합니다.
 */
interface PyodideLikeProxy {
  /** WHAT: 파이썬 객체를 자바스크립트 객체로 변환하는 메서드입니다. WHY: 파이썬 데이터를 다루기 위해. HOW: 함수 호출을 통해 JS 값 반환. */
  toJs: () => unknown;
}

/**
 * WHAT: 주어진 객체가 `toJs` 메서드를 가진 PyodideLikeProxy 타입인지 런타임에 확인하는 타입 가드 함수입니다.
 * WHY: 객체의 유효성과 `toJs` 속성의 함수 여부를 동적으로 검사하여, 런타임 에러 없이 안전하게 PyProxy의 데이터를 자바스크립트 영역으로 추출할 수 있도록 보장하는 역할을 합니다.
 * HOW: typeof 연산자와 in 연산자를 사용하여 입력된 값이 객체이며 null이 아니고, 'toJs' 속성이 존재하며 그 타입이 'function'인지 논리식으로 평가하여 불리언 결과를 반환합니다.
 * 
 * @param input 검사할 임의의 데이터
 * @returns `toJs` 메서드가 존재하고 함수이면 true
 */
function hasToJs(input: unknown): input is PyodideLikeProxy {
  return (
    typeof input === "object" &&
    input !== null &&
    "toJs" in input &&
    typeof (input as { toJs?: unknown }).toJs === "function"
  );
}

/**
 * WHAT: 주어진 입력 데이터를 검증하고 안전하게 Float32Array 형태로 변환 혹은 반환하는 함수입니다. (H-05 Fix 적용)
 * WHY: 데이터의 중복 복사를 막아(Zero-Copy) 대용량 텐서 데이터 전송 시의 성능 저하를 방지하면서도, 데이터 타입의 일관성을 유지하기 위해 존재합니다.
 * HOW: 이미 Float32Array 형태이면 원본 그대로 반환합니다. PyProxy인 경우 toJs() 결과를 추출한 후 Float32Array이면 그대로 반환하고, ArrayBuffer라면 새로운 Float32Array 뷰로 래핑하여 반환합니다. 그 외에는 예외를 던집니다.
 *
 * @param input Pyodide Proxy 객체이거나 ArrayBuffer/Float32Array 형태의 데이터
 * @returns 확보된 Float32Array
 */
function isBufferDetached(buf: ArrayBufferLike): boolean {
  return (buf as any).detached === true || buf.byteLength === 0;
}

export type SafeCopyOptions = {
  retryDetached?: boolean;
  reacquire?: () => Float32Array;
};

export function ensureFloat32Array(
  input: unknown,
  options: SafeCopyOptions = {}
): Float32Array {
  if (input instanceof Float32Array) {
    if (!isBufferDetached(input.buffer)) {
      return input; // H-05: 복사 제거 — 이미 올바른 타입
    }
    if (options.retryDetached && options.reacquire) {
      const fresh = options.reacquire();
      if (!isBufferDetached(fresh.buffer)) {
        return fresh;
      }
    }
    throw new Error("WASM Memory Detached: ArrayBuffer has been detached by memory.grow.");
  }

  if (hasToJs(input)) {
    const jsView = input.toJs();
    
    if (jsView instanceof Float32Array) {
      if (!isBufferDetached(jsView.buffer)) {
        return jsView; // H-05: 복사 제거 — WASM 힙 뷰 그대로 반환
      }
      if (options.retryDetached && options.reacquire) {
        const fresh = options.reacquire();
        if (!isBufferDetached(fresh.buffer)) {
          return fresh;
        }
      }
      throw new Error("WASM Memory Detached: ArrayBuffer has been detached by memory.grow.");
    }
    
    if (jsView instanceof ArrayBuffer) {
      if (!isBufferDetached(jsView)) {
        return new Float32Array(jsView);
      }
      if (options.retryDetached && options.reacquire) {
        const fresh = options.reacquire();
        if (!isBufferDetached(fresh.buffer)) {
          return fresh;
        }
      }
      throw new Error("WASM Memory Detached: ArrayBuffer has been detached by memory.grow.");
    }
  }

  throw new Error(
    "Invalid input type: expected Float32Array or a Pyodide proxy coercible to Float32Array."
  );
}

/**
 * WHAT: 입력 데이터를 강제로 새로운 메모리 공간에 깊은 복사(Deep Copy)하여 반환하는 함수입니다. (명시적 deep copy 용도)
 * WHY: 원본 데이터(WASM 힙 등)가 삭제되거나 변경되어도 안전하게 데이터를 보존해야 할 때, 혹은 독립적인 데이터 소유권을 가지는 버퍼가 필요할 때 호출하기 위해 존재합니다. 일반 데이터 읽기에는 성능상 사용하지 않아야 합니다.
 * HOW: ensureFloat32Array를 호출하여 먼저 안전한 뷰를 확보한 뒤, new Float32Array(view)를 사용하여 동일한 요소들을 가지는 완전히 새로운 메모리 배열 인스턴스를 할당하여 반환합니다.
 * 
 * @param input 원본 데이터
 * @returns 독립된 메모리 공간을 가지는 복사된 Float32Array
 */
export function cloneToFloat32Array(input: unknown): Float32Array {
  /**
   * WHAT: 원본 데이터로부터 읽기 가능한 Float32Array 뷰를 안전하게 가져와 담아두는 변수입니다.
   * WHY: 복사를 수행하기 전, 원본 데이터가 어떤 형태이든(PyProxy, ArrayBuffer 등) 통일된 Float32Array 포맷으로 만들어놓기 위해서입니다.
   * HOW: ensureFloat32Array(input) 함수를 호출하여 반환값을 저장합니다.
   */
  const view = ensureFloat32Array(input);
  return new Float32Array(view);
}

```

---

## `packages/forge/src/devtools/inspector.ts`

```typescript
/**
 * AMEVA-Forge Lightweight In-Browser Visual Inspector & DevTools HUD
 * Real-time VRAM allocation tracking & Training loss curve visualization
 */

import { _globalQuotaManager, getQuotaSnapshot } from '../webgpu/quota';
import { _globalRegistry } from '../tensor/tensorRegistry';

export interface InspectorState {
  mounted: boolean;
  history: Array<{ step: number; loss: number }>;
}

let inspectorContainer: HTMLElement | null = null;
let canvasElement: HTMLCanvasElement | null = null;
let animationFrameId: number | null = null;
const lossHistory: Array<{ step: number; loss: number }> = [];

/**
 * Record a training step loss for live chart visualization
 */
export function recordStepLoss(step: number, loss: number): void {
  lossHistory.push({ step, loss });
  if (lossHistory.length > 200) {
    lossHistory.shift();
  }
}

/**
 * Clear recorded training history
 */
export function clearStepLossHistory(): void {
  lossHistory.length = 0;
}

/**
 * Render HUD loop
 */
function renderHUD(): void {
  if (!canvasElement) return;
  const ctx = canvasElement.getContext('2d');
  if (!ctx) return;

  const width = canvasElement.width;
  const height = canvasElement.height;

  // Background
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, width, height);

  // Header / Metrics
  const quota = getQuotaSnapshot();
  const handles = _globalRegistry.snapshotHandles();

  ctx.fillStyle = '#f8fafc';
  ctx.font = 'bold 12px monospace';
  ctx.fillText('⚡ AMEVA-Forge DevTools', 10, 20);

  ctx.font = '10px monospace';
  ctx.fillStyle = '#94a3b8';
  const vramKB = (quota.usedBytes / 1024).toFixed(1);
  const maxKB = (quota.maxBytes / (1024 * 1024)).toFixed(0);
  ctx.fillText(`VRAM: ${vramKB} KB / ${maxKB} MB | Handles: ${handles.length}`, 10, 36);

  // VRAM Bar
  const barWidth = width - 20;
  const barHeight = 6;
  ctx.fillStyle = '#334155';
  ctx.fillRect(10, 44, barWidth, barHeight);

  const usageRatio = Math.min(1.0, quota.usedBytes / Math.max(1, quota.maxBytes));
  ctx.fillStyle = usageRatio > 0.8 ? '#ef4444' : '#10b981';
  ctx.fillRect(10, 44, barWidth * usageRatio, barHeight);

  // Loss Curve Area
  const chartX = 10;
  const chartY = 60;
  const chartW = width - 20;
  const chartH = height - 70;

  ctx.fillStyle = '#1e293b';
  ctx.fillRect(chartX, chartY, chartW, chartH);

  if (lossHistory.length > 1) {
    const minLoss = Math.min(...lossHistory.map(h => h.loss));
    const maxLoss = Math.max(...lossHistory.map(h => h.loss), minLoss + 1e-4);

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    for (let i = 0; i < lossHistory.length; i++) {
      const x = chartX + (i / (lossHistory.length - 1)) * chartW;
      const normalizedY = (lossHistory[i].loss - minLoss) / (maxLoss - minLoss);
      const y = chartY + chartH - normalizedY * (chartH - 8) - 4;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    const latest = lossHistory[lossHistory.length - 1];
    ctx.fillStyle = '#38bdf8';
    ctx.font = '9px monospace';
    ctx.fillText(`Step ${latest.step}: Loss ${latest.loss.toFixed(4)}`, 14, chartY + 12);
  } else {
    ctx.fillStyle = '#64748b';
    ctx.font = '9px monospace';
    ctx.fillText('Awaiting training steps...', chartX + 10, chartY + chartH / 2);
  }

  animationFrameId = requestAnimationFrame(renderHUD);
}

/**
 * Mount floating DevTools HUD overlay into DOM
 */
export function mountInspector(targetParent?: HTMLElement): HTMLElement {
  if (inspectorContainer) {
    return inspectorContainer;
  }

  const container = document.createElement('div');
  container.id = 'ameva-forge-devtools';
  container.style.position = 'fixed';
  container.style.bottom = '16px';
  container.style.right = '16px';
  container.style.width = '280px';
  container.style.height = '180px';
  container.style.backgroundColor = '#0f172a';
  container.style.border = '1px solid #334155';
  container.style.borderRadius = '8px';
  container.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.4)';
  container.style.zIndex = '999999';
  container.style.overflow = 'hidden';
  container.style.fontFamily = 'monospace';

  const canvas = document.createElement('canvas');
  canvas.width = 280;
  canvas.height = 180;
  canvas.style.display = 'block';
  container.appendChild(canvas);

  const parent = targetParent || document.body;
  parent.appendChild(container);

  inspectorContainer = container;
  canvasElement = canvas;

  if (typeof requestAnimationFrame !== 'undefined') {
    animationFrameId = requestAnimationFrame(renderHUD);
  }

  return container;
}

/**
 * Unmount and destroy DevTools HUD
 */
export function unmountInspector(): void {
  if (animationFrameId !== null && typeof cancelAnimationFrame !== 'undefined') {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  if (inspectorContainer && inspectorContainer.parentNode) {
    inspectorContainer.parentNode.removeChild(inspectorContainer);
  }
  inspectorContainer = null;
  canvasElement = null;
}

```

---

## `packages/forge/src/errors.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */

/**
 * WHAT: AMEVA Forge 시스템 전체에서 발생하는 모든 커스텀 에러의 최상위 기본 클래스입니다.
 * WHY: 표준 Error 객체를 확장하여 스택 트레이스와 에러 이름을 올바르게 유지함으로써, 이 라이브러리 내부에서 발생하는 예외 상황을 쉽게 식별하고 포착(catch)할 수 있게 하기 위함입니다.
 * HOW: 자바스크립트의 내장 Error 클래스를 상속(extends)받아 구현됩니다.
 */
export class AMEVAForgeError extends Error {
  /**
   * WHAT: AMEVAForgeError 인스턴스를 생성하는 생성자입니다.
   * WHY: 에러 메시지를 초기화하고, 클래스의 인스턴스 타입 체크(instanceof)가 정상적으로 작동하도록 프로토타입 체인을 교정하기 위해 필요합니다.
   * HOW: 부모 생성자(super)를 호출한 후, this.name을 설정하고 Object.setPrototypeOf를 사용하여 프로토타입을 강제로 맞춰줍니다.
   * 
   * @param message 사용자에게 노출될 구체적인 에러 메시지 내용
   */
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * WHAT: 텐서의 형태(Shape)나 차원(Dimension)이 계산 또는 검증 중 맞지 않을 때 발생하는 에러 클래스입니다.
 * WHY: 연산의 수학적/구조적 조건이 위배되었음을 사용자나 상위 로직에 명확히 알리기 위해 존재합니다.
 * HOW: AMEVAForgeError를 상속받아 정의되어, Shape 관련된 구체적인 예외 상황을 나타내는 타입으로 활용됩니다.
 */
export class AMEVAForgeShapeError extends AMEVAForgeError {}

/**
 * WHAT: 텐서의 데이터 타입(DType)이 연산에서 지원하지 않거나 서로 충돌할 때 발생하는 에러 클래스입니다.
 * WHY: 잘못된 자료형 접근이나 호환되지 않는 텐서 연산을 조기에 차단하여 런타임 크래시를 방지하기 위해 사용됩니다.
 * HOW: AMEVAForgeError를 상속받아 DType 특화 예외를 표현합니다.
 */
export class AMEVAForgeDTypeError extends AMEVAForgeError {}

/**
 * WHAT: GPU 등 하드웨어 디바이스를 초기화하거나 통신하는 과정에서 발생하는 에러 클래스입니다.
 * WHY: 디바이스 손실(Device Lost)이나 잘못된 디바이스 상태 등 하드웨어 의존적인 실패 상황을 명확히 구분하여 처리하기 위해 필요합니다.
 * HOW: AMEVAForgeError를 상속받아 GPU/디바이스 레벨의 문제를 나타냅니다.
 */
export class AMEVAForgeDeviceError extends AMEVAForgeError {}

/**
 * WHAT: 이미 메모리에서 해제된(disposed) 텐서 자원에 접근하려고 시도할 때 발생하는 에러 클래스입니다.
 * WHY: 메모리 누수나 무효한 메모리 접근(Use-After-Free)을 방지하는 안전장치 역할을 하여, 잘못된 리소스 참조를 차단하기 위함입니다.
 * HOW: AMEVAForgeError를 상속받아 생명주기가 끝난 객체에 대한 접근 시 던져집니다.
 */
export class AMEVAForgeDisposedError extends AMEVAForgeError {}

/**
 * WHAT: 시스템이나 WebGPU에서 할당 가능한 메모리 할당량(Quota)이나 버퍼 크기를 초과했을 때 발생하는 에러 클래스입니다.
 * WHY: 제한된 VRAM이나 시스템 리소스 한계에 도달했음을 명확히 알리고, 메모리 할당 실패를 우아하게(gracefully) 처리하기 위해 존재합니다.
 * HOW: AMEVAForgeError를 상속받아 메모리 관련 한계 초과를 나타냅니다.
 */
export class AMEVAForgeQuotaExceededError extends AMEVAForgeError {}

/**
 * WHAT: 실행 중인 브라우저나 환경이 WebGPU API 자체를 지원하지 않을 때 발생하는 에러 클래스입니다.
 * WHY: 호환되지 않는 환경에서 실행을 시도할 때 발생시켜, 폴백(fallback) 메커니즘을 구동하거나 사용자에게 호환성 문제를 신속히 알리기 위해 사용됩니다.
 * HOW: AMEVAForgeError를 상속받아 WebGPU 초기화 실패 시 즉각적으로 던져집니다.
 */
export class AMEVAForgeWebGPUUnavailableError extends AMEVAForgeError {}

/**
 * WHAT: 보안 정책, 권한 부족, 혹은 검증되지 않은 셰이더/WASM 접근 등 보안 관련된 문제가 발생했을 때 던져지는 에러 클래스입니다.
 * WHY: 비정상적인 메모리 접근이나 권한을 벗어난 조작을 막아 시스템의 전반적인 안전성을 보장하기 위한 보호 계층으로 작용합니다.
 * HOW: AMEVAForgeError를 상속받아 보안 정책 위반 시 발동됩니다.
 */
export class AMEVAForgeSecurityError extends AMEVAForgeError {}

/**
 * WHAT: 현재 구현되지 않았거나 지원하지 않는 연산(Operation)을 실행하려고 할 때 발생하는 에러 클래스입니다.
 * WHY: 사용자가 유효하지 않은 그래프 노드나 현재 라이브러리에서 지원 범위를 벗어난 커널을 호출하는 것을 사전에 막아 오작동을 예방합니다.
 * HOW: AMEVAForgeError를 상속받아 구현되지 않은 기능 호출 시 발생합니다.
 */
export class AMEVAForgeUnsupportedOpError extends AMEVAForgeError {}

/**
 * WHAT: GPU validation error scope에서 감지된 오류 클래스입니다.
 * WHY: WebGPU pushErrorScope('validation') 결과를 typed exception으로 전달하기 위해 존재합니다.
 */
export class AMEVAForgeValidationError extends AMEVAForgeError {}

/**
 * WHAT: GPU out-of-memory error scope에서 감지된 오류 클래스입니다.
 * WHY: WebGPU pushErrorScope('out-of-memory') 결과를 typed exception으로 전달하기 위해 존재합니다.
 */
export class AMEVAForgeOutOfMemoryError extends AMEVAForgeError {}

/**
 * WHAT: GPU internal error scope에서 감지된 오류 클래스입니다.
 * WHY: WebGPU pushErrorScope('internal') 결과를 typed exception으로 전달하기 위해 존재합니다.
 */
export class AMEVAForgeInternalGPUError extends AMEVAForgeError {}

/**
 * WHAT: GPU 디바이스가 유실(device lost)되었을 때 발생하는 오류 클래스입니다.
 * WHY: 디바이스 유실 상황을 명확히 구분하여 재초기화 흐름을 유도하기 위해 존재합니다.
 */
export class AMEVAForgeDeviceLostError extends AMEVAForgeError {}

/**
 * WHAT: 이전 generation의 stale handle에 접근할 때 발생하는 오류 클래스입니다.
 * WHY: Device lost 후 재초기화된 환경에서 이전 텐서 접근을 차단하기 위해 존재합니다.
 */
export class AMEVAForgeStaleHandleError extends AMEVAForgeError {}

```

---

## `packages/forge/src/generated/opSchema.ts`

```typescript
/**
 * AUTO-GENERATED FILE. DO NOT MODIFY MANUALLY.
 * Generated from packages/forge/schema/release1-ops.json
 * Run `py -3 scripts/generate_release1_contracts.py` to regenerate.
 */

export interface OpParamDef {
  name: string;
  type: string;
}

export interface OpDef {
  inputs: number;
  params: OpParamDef[];
  output: string;
  dtypes: string[];
}

export const RELEASE1_OP_SCHEMA: Record<string, OpDef> = {
  "add": {
    "inputs": 2,
    "params": [],
    "output": "same-shape",
    "dtypes": [
      "float32"
    ]
  },
  "sub": {
    "inputs": 2,
    "params": [],
    "output": "same-shape",
    "dtypes": [
      "float32"
    ]
  },
  "mul": {
    "inputs": 2,
    "params": [],
    "output": "same-shape",
    "dtypes": [
      "float32"
    ]
  },
  "div": {
    "inputs": 2,
    "params": [],
    "output": "same-shape",
    "dtypes": [
      "float32"
    ]
  },
  "neg": {
    "inputs": 1,
    "params": [],
    "output": "same-shape",
    "dtypes": [
      "float32"
    ]
  },
  "matmul": {
    "inputs": 2,
    "params": [
      {
        "name": "M",
        "type": "positive-int"
      },
      {
        "name": "N",
        "type": "positive-int"
      },
      {
        "name": "K",
        "type": "positive-int"
      }
    ],
    "output": "matmul-2d",
    "dtypes": [
      "float32"
    ]
  },
  "transpose": {
    "inputs": 1,
    "params": [
      {
        "name": "M",
        "type": "positive-int"
      },
      {
        "name": "N",
        "type": "positive-int"
      }
    ],
    "output": "transpose-2d",
    "dtypes": [
      "float32"
    ]
  },
  "reshape": {
    "inputs": 1,
    "params": [
      {
        "name": "targetShape",
        "type": "shape-tuple"
      }
    ],
    "output": "reshape",
    "dtypes": [
      "float32"
    ]
  },
  "sum": {
    "inputs": 1,
    "params": [],
    "output": "scalar",
    "dtypes": [
      "float32"
    ]
  },
  "relu": {
    "inputs": 1,
    "params": [],
    "output": "same-shape",
    "dtypes": [
      "float32"
    ]
  },
  "relu_backward": {
    "inputs": 2,
    "params": [
      {
        "name": "numElements",
        "type": "positive-int"
      }
    ],
    "output": "same-shape",
    "dtypes": [
      "float32"
    ]
  },
  "mse_loss": {
    "inputs": 2,
    "params": [],
    "output": "scalar",
    "dtypes": [
      "float32"
    ]
  },
  "mse_loss_backward": {
    "inputs": 2,
    "params": [
      {
        "name": "numElements",
        "type": "positive-int"
      }
    ],
    "output": "same-shape",
    "dtypes": [
      "float32"
    ]
  },
  "axpy": {
    "inputs": 2,
    "params": [
      {
        "name": "numElements",
        "type": "positive-int"
      },
      {
        "name": "alpha",
        "type": "float32"
      }
    ],
    "output": "alias-input-0",
    "dtypes": [
      "float32"
    ]
  }
};

export type Release1OpName = keyof typeof RELEASE1_OP_SCHEMA;

```

---

## `packages/forge/src/index.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *   - 2026-08-17: Security Hardening: Enforce strict public API boundary, isolate raw GPUDevice
 * 
 * WHAT: 라이브러리의 외부 공개용(Public) API를 모두 한 곳으로 모아 내보내는 진입점(엔트리포인트) 파일입니다.
 * WHY: 패키지 사용자가 내부 디렉토리 구조를 일일이 알 필요 없이 일관된 단일 경로에서 모듈을 쉽게 임포트할 수 있도록 편의성을 제공하기 위함입니다.
 * HOW: 내부의 여러 모듈들에 정의된 클래스, 타입, 함수 등을 export 및 re-export 키워드를 활용하여 다시 바깥으로 통합 추출합니다.
 */

import { getDevice, initWebGPU, isAvailable, _resetDeviceForTesting } from "./webgpu/device";
import { _globalQuotaManager, QuotaManager, getQuotaSnapshot } from "./webgpu/quota";
import { AMEVAForgeValidationError } from "./errors";

declare var process: any;

export * from "./errors";
export * from "./types";

// Security Hardened WebGPU device exports (raw GPUDevice / Queue / Adapter NOT publicly exposed)
export { initWebGPU, isAvailable };
export { assertWasmRange } from "./webgpu/validateWasmRange";
export { QuotaManager, getQuotaSnapshot };
export { flushGC, clearStagingPool } from "./webgpu/buffers";
export * from "./webgpu/shaderGuard";

export * from "./tensor/validateShape";
export * from "./tensor/validateDType";
export * from "./tensor/dispatchShape";
export * from "./tensor/broadcastParams";
export * from "./tensor/gpuCore";
export { executeGraph, configureRuntime, getRuntimeConfig, type ForgeRuntimeConfig } from "./tensor/graphExecutor";

export * from "./bridge/safeCopy";
export * from "./bridge/pyodideBridge";
export * from "./devtools/inspector";

/**
 * WHAT: 테스트 환경(E2E / Jest)에서만 제어 가능한 결함 주입(Fault Injection) 훅입니다.
 * WHY: 프로덕션 환경에 raw GPUDevice를 노출하지 않으면서도 OOM, Validation, Device Lost 복구력을 엄격히 검증하기 위함입니다.
 */
export const __testing = (
  (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'test') ||
  (typeof globalThis !== 'undefined' && ((globalThis as any).__AMEVA_TEST_MODE__ || (globalThis as any).testReady))
) ? Object.freeze({
  destroyDevice: () => {
    try {
      getDevice().destroy();
    } catch (e) { console.warn(`[__testing] destroyDevice failed: ${e}`); }
    try {
      _resetDeviceForTesting();
    } catch (e) { console.warn(`[__testing] _resetDeviceForTesting failed: ${e}`); }
  },
  triggerValidationError: async () => {
    const dev = getDevice();
    dev.pushErrorScope('validation');
    try {
      dev.createBuffer({
        size: 1024,
        usage: 0 as GPUBufferUsageFlags, // Usage 0 is an unconditional WebGPU validation fault
      });
    } finally {
      const err = await dev.popErrorScope();
      if (err) {
        throw new AMEVAForgeValidationError(`GPU Validation Error: ${err.message}`);
      }
    }
  },
  setQuotaLimit: (maxBytes: number) => {
    _globalQuotaManager.setLimits(maxBytes, maxBytes);
  },
  getDeviceInternal: getDevice,
}) : undefined;

```

---

## `packages/forge/src/tensor/broadcastParams.ts`

```typescript
/**
 * 파일 생성일: 2026-08-18T14:42:00+09:00
 * 역할: 8D 다차원 스트라이드 브로드캐스팅 파라미터 계산 유틸리티
 * 목적: Direct TS API(gpuCore.ts)와 Graph Executor(graphExecutor.ts) 간의 112-Byte WGSL 유니폼 버퍼 계약 일치
 */

export interface BroadcastParams {
  dOut: number[];
  effSA: number[];
  effSB: number[];
}

/**
 * WHAT: 두 텐서의 형태(shapeA, shapeB)를 8차원으로 좌측 패딩(pad8)하고 유효 스트라이드를 계산합니다.
 * WHY: WGSL 셰이더(ADD, SUB, MUL, DIV)가 8차원 좌표 디코딩을 수행할 때 정확한 메모리 오프셋을 역산할 수 있도록 하기 위함입니다.
 * HOW: 8차원으로 정규화 후 역순 스트라이드를 계산하고, 크기가 1인 차원은 스트라이드를 0으로 매핑(브로드캐스팅)합니다.
 */
export function computeBroadcastParams(outShape: number[], shapeA: number[], shapeB: number[]): BroadcastParams {
  const pad8 = (s: number[]) => {
    const res = [1, 1, 1, 1, 1, 1, 1, 1];
    const diff = 8 - s.length;
    for (let i = 0; i < s.length; i++) {
      res[diff + i] = s[i];
    }
    return res;
  };
  const dOut = pad8(outShape);
  const dA = pad8(shapeA);
  const dB = pad8(shapeB);

  const calcStrides = (dims: number[]) => {
    const st = [1, 1, 1, 1, 1, 1, 1, 1];
    st[7] = 1;
    for (let i = 6; i >= 0; i--) {
      st[i] = st[i + 1] * dims[i + 1];
    }
    return st;
  };
  const baseSA = calcStrides(dA);
  const baseSB = calcStrides(dB);

  const effSA = dA.map((d, i) => d === 1 ? 0 : baseSA[i]);
  const effSB = dB.map((d, i) => d === 1 ? 0 : baseSB[i]);

  return { dOut, effSA, effSB };
}

```

---

## `packages/forge/src/tensor/dispatchShape.ts`

```typescript
/**
 * dispatchShape.ts - 2D WebGPU Workgroup Dispatch Calculator
 * 
 * WHAT: 1D 요소 수(numElements)를 WebGPU의 2D 디스패치 그리드(dispatchX, dispatchY)로 안전하게 분할하는 공용 유틸리티입니다.
 * WHY: WebGPU 디스패치 차원당 한도(65,535)를 초과하는 대용량 텐서(> 4.19M 원소)에서 연산이 절단되는 Silent Truncation 버그를 원천 차단합니다.
 * HOW: dispatchX = min(totalWorkgroups, maxPerDim), dispatchY = ceil(totalWorkgroups / maxPerDim)로 2D 그리드를 계산합니다.
 */
import { AMEVAForgeValidationError } from "../errors";

export type Dispatch2D = {
  dispatchX: number;
  dispatchY: number;
  workgroupsX: number;
  totalWorkgroups: number;
};

export function computeDispatch2D(
  numElements: number,
  workgroupSize: number = 64,
  maxPerDim: number = 65535
): Dispatch2D {
  if (!Number.isSafeInteger(numElements) || numElements <= 0) {
    throw new AMEVAForgeValidationError(`Invalid numElements: ${numElements}`);
  }

  const totalWorkgroups = Math.ceil(numElements / workgroupSize);
  const dispatchX = Math.min(totalWorkgroups, maxPerDim);
  const dispatchY = Math.ceil(totalWorkgroups / maxPerDim);

  if (dispatchY > maxPerDim) {
    throw new AMEVAForgeValidationError(
      `Dispatch too large: ${totalWorkgroups} workgroups exceeds 2D WebGPU limit (${maxPerDim}x${maxPerDim})`
    );
  }

  return {
    dispatchX,
    dispatchY,
    workgroupsX: dispatchX,
    totalWorkgroups,
  };
}

```

---

## `packages/forge/src/tensor/gpuCore.ts`

```typescript
/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:59:35+09:00: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 *   - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * gpuCore.ts — GPU 코어 API (초기화, 텐서 생명주기, 개별 op 디스패치)
 *
 * H-01 Fix: 모든 op에 _globalPipelineCache 적용 (셰이더 재컴파일 방지)
 * NH-03 Fix: quota를 maxStorageBufferBindingSize 기반으로 설정 (maxBufferSize는 단일 버퍼 크기 제한이지 VRAM 용량이 아님)
 * NH-01 Fix: 개별 op 함수들을 internal로 유지, pyodideBridge에서는 executeGraph만 노출
 * NH-07 Fix: shaderGuard.assertAllowedKernelName() 실제 호출
 * ARC-01 Fix: device.pushErrorScope로 OOM 감지 시도
 * L-01 Fix: dispatchKernel 헬퍼로 모든 op의 반복 코드 통합 (DRY)
 */

import { initWebGPU, getDevice, getAdapter, setDeviceLostCallback } from "../webgpu/device";
import {
  allocateBuffer,
  writeFloat32Array,
  readBufferToFloat32Array,
  freeBuffer,
  clearStagingPool,
  mapBufferAsync as _mapBufferAsync,
  readMappedInto as _readMappedInto,
} from "../webgpu/buffers";
import { _globalQuotaManager, AllocationToken } from "../webgpu/quota";
import { _globalRegistry } from "./tensorRegistry";
import { _globalPipelineCache } from "../webgpu/pipelineCache";
import { assertAllowedKernelName, registerKernelNames } from "../webgpu/shaderGuard";
import { TensorHandle, DType, TensorInfo } from "../types";
import { AMEVAForgeShapeError, AMEVAForgeDTypeError } from "../errors";
import { validateShape } from "./validateShape";
import { validateDType } from "./validateDType";
import { assertWasmRange } from "../webgpu/validateWasmRange";
import { computeBroadcastParams } from "./broadcastParams";
import { computeDispatch2D } from "./dispatchShape";

import { MATMUL_WGSL } from "./kernels/matmul.wgsl";
import { RELU_WGSL } from "./kernels/relu.wgsl";
import { ADD_WGSL } from "./kernels/add.wgsl";
import { TRANSPOSE_WGSL } from "./kernels/transpose.wgsl";
import { MUL_WGSL } from "./kernels/mul.wgsl";
import { RELU_BACKWARD_WGSL } from "./kernels/relu_backward.wgsl";
import { SUB_WGSL } from "./kernels/sub.wgsl";
import { NEG_WGSL } from "./kernels/neg.wgsl";
import { DIV_WGSL } from "./kernels/div.wgsl";
import { EXP_WGSL } from "./kernels/exp.wgsl";
import { LOG_WGSL } from "./kernels/log.wgsl";
import { SIGMOID_WGSL } from "./kernels/sigmoid.wgsl";
import { TANH_WGSL } from "./kernels/tanh.wgsl";
import { SIGMOID_BACKWARD_WGSL } from "./kernels/sigmoid_backward.wgsl";
import { TANH_BACKWARD_WGSL } from "./kernels/tanh_backward.wgsl";
import { FILL_WGSL } from "./kernels/fill.wgsl";
import { SUM_WGSL } from "./kernels/sum.wgsl";
import { MAX_WGSL } from "./kernels/max.wgsl";
import { SUM_AXIS_WGSL } from "./kernels/sum_axis.wgsl";
import { MAX_AXIS_WGSL } from "./kernels/max_axis.wgsl";
import { MAX_AXIS_BACKWARD_WGSL } from "./kernels/max_axis_backward.wgsl";
import { AXPY_WGSL } from "./kernels/axpy.wgsl";
import { PAD_WGSL } from "./kernels/pad.wgsl";
import { GATHER_WGSL } from "./kernels/gather.wgsl";
import { SCATTER_WGSL } from "./kernels/scatter.wgsl";
import { CAT_WGSL } from "./kernels/cat.wgsl";
import { WHERE_WGSL } from "./kernels/where.wgsl";
import { DROPOUT_WGSL } from "./kernels/dropout.wgsl";
import { MAXPOOL2D_WGSL } from "./kernels/maxpool2d.wgsl";
import { AVGPOOL2D_WGSL } from "./kernels/avgpool2d.wgsl";
import { IM2COL_WGSL } from "./kernels/im2col.wgsl";
import { COL2IM_WGSL } from "./kernels/col2im.wgsl";
import { PERMUTE_WGSL } from "./kernels/permute.wgsl";
import { BATCHED_MATMUL_WGSL } from "./kernels/batched_matmul.wgsl";
import { MATMUL_BIAS_RELU_WGSL } from "./kernels/matmul_bias_relu.wgsl";

/**
 * WHAT: 모든 WGSL 셰이더 코드를 커널 이름에 매핑하여 저장하는 전역 읽기 전용 레지스트리 맵입니다.
 * WHY: 런타임에 셰이더 코드를 이름으로 조회하고 파이프라인 캐시 초기화 시 한 번에 반영하기 위해 존재합니다.
 * HOW: Map 객체를 생성하여 문자열 키와 WGSL 코드 문자열 값을 쌍으로 저장합니다.
 */
export const KERNEL_REGISTRY: ReadonlyMap<string, string> = new Map([
  ['matmul', MATMUL_WGSL],
  ['matmul_bias_relu', MATMUL_BIAS_RELU_WGSL],
  ['batched_matmul', BATCHED_MATMUL_WGSL],
  ['relu', RELU_WGSL],
  ['add', ADD_WGSL],
  ['mul', MUL_WGSL],
  ['transpose', TRANSPOSE_WGSL],
  ['relu_backward', RELU_BACKWARD_WGSL],
  ['sub', SUB_WGSL],
  ['neg', NEG_WGSL],
  ['div', DIV_WGSL],
  ['exp', EXP_WGSL],
  ['log', LOG_WGSL],
  ['sigmoid', SIGMOID_WGSL],
  ['tanh', TANH_WGSL],
  ['sigmoid_backward', SIGMOID_BACKWARD_WGSL],
  ['tanh_backward', TANH_BACKWARD_WGSL],
  ['fill', FILL_WGSL],
  ['sum', SUM_WGSL],
  ['max', MAX_WGSL],
  ['sum_axis', SUM_AXIS_WGSL],
  ['max_axis', MAX_AXIS_WGSL],
  ['max_axis_backward', MAX_AXIS_BACKWARD_WGSL],
  ['axpy', AXPY_WGSL],
  ['pad', PAD_WGSL],
  ['gather', GATHER_WGSL],
  ['scatter', SCATTER_WGSL],
  ['cat', CAT_WGSL],
  ['where', WHERE_WGSL],
  ['dropout', DROPOUT_WGSL],
  ['maxpool2d', MAXPOOL2D_WGSL],
  ['avgpool2d', AVGPOOL2D_WGSL],
  ['im2col', IM2COL_WGSL],
  ['col2im', COL2IM_WGSL],
  ['permute', PERMUTE_WGSL],
]);

// VUL-001 Fix: Register kernel names automatically to keep whitelist in sync
registerKernelNames(KERNEL_REGISTRY.keys());

/**
 * WHAT: CPU로 읽어오기 위해 대기 중인 GPU 스테이징 버퍼들을 추적하는 전역 맵입니다.
 * WHY: 비동기 맵핑(mapAsync)이 완료된 버퍼를 기록해 두고 나중에 동기적으로 데이터를 읽어올 수 있게 하기 위해 필요합니다.
 * HOW: 텐서 핸들(문자열)을 키로, 매핑된 GPUBuffer와 AllocationToken 객체를 값으로 유지합니다.
 */
const _pendingStagingBuffers = new Map<TensorHandle, { stagingBuffer: GPUBuffer, token: AllocationToken }>();
const _inFlightMapPromises = new Map<TensorHandle, Promise<void>>();

/**
 * WHAT: GPU 코어의 런타임 메모리와 모든 캐시된 리소스를 초기화(해제)하는 함수입니다.
 * WHY: 디바이스 유실(Device Lost) 이벤트가 발생하거나 시스템 강제 리셋 시 남은 자원의 메모리 누수를 방지하기 위해 존재합니다.
 * HOW: 텐서 레지스트리, 쿼터 매니저, 파이프라인 캐시를 지우고, 대기 중인 스테이징 버퍼들도 순회하여 언맵(unmap) 및 파괴(destroy)합니다.
 */
export function resetRuntimeMemory(reason: string = "manual-reset"): void {
  _safeLog(`[RuntimeReset] start: ${reason}`);
  
  // 1. Pending staging buffers & staging pool cleanup
  try {
    for (const [, obj] of _pendingStagingBuffers) {
      try { obj.stagingBuffer.unmap(); } catch { /* already unmapped */ }
      try { obj.stagingBuffer.destroy(); } catch { /* already destroyed */ }
      _globalQuotaManager.releaseToken(obj.token);
    }
    _pendingStagingBuffers.clear();
  } catch (e) {
    _safeLog(`[RuntimeReset] staging buffer cleanup error: ${e}`);
  }

  try {
    clearStagingPool(); // VULN-04: Clear pool buffers & tokens
  } catch (e) {
    _safeLog(`[RuntimeReset] clearStagingPool error: ${e}`);
  }

  // 2. In-flight promises & pipeline cache
  try {
    _inFlightMapPromises.clear();
    _globalPipelineCache.clear();
  } catch (e) {
    _safeLog(`[RuntimeReset] pipeline cache error: ${e}`);
  }

  // 3. Quota & registry reset
  try {
    _globalRegistry.clear();
  } catch (e) {
    _safeLog(`[RuntimeReset] registry clear error: ${e}`);
  }

  try {
    _globalQuotaManager.reset();
  } catch (e) {
    _safeLog(`[RuntimeReset] quota reset error: ${e}`);
  }

  _safeLog(`[RuntimeReset] done: ${reason}`);
}

/**
 * WHAT: 시스템 로거가 존재할 경우 로그 메시지를 남기는 래퍼 함수입니다.
 * WHY: 글로벌 환경(예: Pyodide)에 주입된 로그 함수가 있을 때만 호출하여 콘솔 오염을 막고 안전한 디버깅을 하기 위함입니다.
 * HOW: globalThis에서 log 함수를 찾아 존재하면 호출하고 오류 발생 시 조용히 무시(catch)합니다.
 */
function _safeLog(msg: string) {
  try {
    if (typeof (globalThis as any).log === 'function') {
      (globalThis as any).log(msg, 'system');
    }
  } catch (e) {}
}

/**
 * WHAT: WebGPU 하위 시스템을 초기화하고 메모리 한도 설정 및 셰이더 컴파일을 수행하는 비동기 진입점 함수입니다.
 * WHY: 텐서 연산을 수행하기 전에 GPU 디바이스를 획득하고 하드웨어 제약을 파악하며 파이프라인을 준비하기 위해 필수적입니다.
 * HOW: initWebGPU를 호출하여 디바이스를 얻고, 디바이스 어댑터의 limits를 조회하여 메모리 할당 한도를 설정한 뒤, 모든 커널을 사전 컴파일(warmup)합니다.
 */
export async function init(
  options?: GPURequestAdapterOptions & { vramLimitBytes?: number }
): Promise<void> {
  _safeLog(`[gpuCore.ts] init started`);
  setDeviceLostCallback(() => {
    resetRuntimeMemory();
  });

  try {
    _safeLog(`[gpuCore.ts] calling initWebGPU...`);
    await initWebGPU(options);
    _safeLog(`[gpuCore.ts] initWebGPU finished`);
  } catch (e: any) {
    _safeLog(`[gpuCore.ts] initWebGPU threw error: ${e.message}`);
    throw e;
  }

  // NH-03: 실제 GPU 제한 조회 후 쿼터 조정
  /**
   * WHAT: 초기화된 WebGPU 어댑터 객체입니다.
   * WHY: 현재 시스템 GPU의 하드웨어 한계(limits)와 기능 정보를 파악하여 안전한 메모리 할당량을 계산하기 위해 조회합니다.
   * HOW: getAdapter() 함수를 호출하여 가져옵니다.
   */
  const adapter = getAdapter();
  if (adapter) {
    /**
     * WHAT: 현재 GPU 어댑터가 지원하는 하드웨어 제약사항을 담은 객체입니다.
     * WHY: 버퍼 바인딩 크기나 컴퓨트 워크그룹 크기의 안전 한계선을 알기 위해 참조합니다.
     * HOW: adapter.limits 프로퍼티를 통해 가져옵니다.
     */
    const limits = adapter.limits;

    if (limits.maxComputeWorkgroupSizeX < 64) {
      console.warn(`[AMEVA] Warning: Device maxComputeWorkgroupSizeX (${limits.maxComputeWorkgroupSizeX}) is less than 64. Kernels are optimized for 64.`);
    }

    /**
     * WHAT: 스토리지 버퍼가 단일 바인딩 시 사용할 수 있는 최대 바이트 크기입니다.
     * WHY: 이 값을 기준으로 사용 가능한 전체 VRAM 용량을 간접적으로 추정하기 위해 필요합니다.
     * HOW: limits.maxStorageBufferBindingSize를 사용하며, 정보가 없으면 기본값(256MB)으로 설정합니다.
     */
    const maxBinding = limits.maxStorageBufferBindingSize ?? 256 * 1024 * 1024;

    /**
     * WHAT: 사용자가 직접 명시한 VRAM 사용 상한(바이트)입니다.
     * WHY: 시스템의 기본 휴리스틱을 무시하고 사용자 설정에 따라 자원을 제어할 수 있도록 옵션으로 받습니다.
     * HOW: options 인자에서 vramLimitBytes 프로퍼티를 참조합니다.
     */
    const userLimit = options?.vramLimitBytes;
    
    /**
     * WHAT: 할당할 수 있는 최대 하드 VRAM 한도입니다.
     * WHY: 시스템 메모리 초과를 방지하기 위해 엄격한 상한선을 두기 위해 계산합니다.
     * HOW: 사용자 지정값이 있으면 8GB를 넘지 않는 선에서 채택하고, 없으면 바인딩 크기의 4배와 8GB 중 작은 값을 사용합니다.
     */
    const hardLimit = userLimit
      ? Math.min(userLimit, 8 * 1024 * 1024 * 1024)
      : Math.min(maxBinding * 4, 8 * 1024 * 1024 * 1024); // binding 크기의 4배를 총 VRAM 추정
      
    /**
     * WHAT: 메모리 압박이 시작될 때 경고를 보내거나 GC를 유도하기 위한 소프트 한도입니다.
     * WHY: 하드 한도에 도달하기 전 선제적인 리소스 회수 타이밍을 잡기 위해 존재합니다.
     * HOW: 하드 한도의 75%로 계산합니다.
     */
    const softLimit = Math.floor(hardLimit * 0.75);

    _globalQuotaManager.setLimits(Math.floor(hardLimit), Math.floor(softLimit));
    console.info(
      `[AMEVA] GPU quota set: soft=${(softLimit / 1e9).toFixed(2)}GB, ` +
      `hard=${(hardLimit / 1e9).toFixed(2)}GB ` +
      `(maxStorageBindingSize=${(maxBinding / 1e9).toFixed(2)}GB)`
    );
  }

  // H-NEW-08: 비동기 파이프라인 사전 컴파일
  await warmupKernels();
}

/**
 * WHAT: 등록된 모든 커널 셰이더를 WebGPU 컴퓨트 파이프라인으로 사전 컴파일하는 함수입니다.
 * WHY: 실행 시점에 셰이더 컴파일이 발생하여 프레임 드랍이나 실행 지연이 생기는 것을 방지하기 위함입니다.
 * HOW: KERNEL_REGISTRY 맵을 순회하여 각 셰이더 코드와 이름 배열을 추출하고 _globalPipelineCache.warmup()을 호출합니다.
 */
export async function warmupKernels(): Promise<void> {
  /**
   * WHAT: KERNEL_REGISTRY에서 추출한 커널 이름(key)과 셰이더 소스코드(wgslCode) 객체의 배열입니다.
   * WHY: 파이프라인 캐시의 warmup 메서드에 한꺼번에 전달할 형식을 맞추기 위해 생성합니다.
   * HOW: Array.from()을 사용하여 맵 엔트리를 배열로 변환한 후 map()으로 객체화합니다.
   */
  const entries = Array.from(KERNEL_REGISTRY.entries()).map(
    ([key, wgslCode]) => ({ key, wgslCode })
  );
  await _globalPipelineCache.warmup(entries);
}

/**
 * WHAT: 핸들에 해당하는 텐서의 메타데이터(크기, 타입, 버퍼 크기 등)를 반환하는 함수입니다.
 * WHY: 파이썬 브릿지나 외부에서 현재 텐서의 형태 정보를 조회해야 할 때 사용됩니다.
 * HOW: 전역 레지스트리에서 핸들로 레코드를 조회한 뒤 TensorInfo 객체를 구성하여 반환합니다.
 */
export function getTensorInfo(handle: TensorHandle): TensorInfo {
  /**
   * WHAT: 핸들로 조회된 내부 텐서 레코드 객체입니다.
   * WHY: 저장된 shape, dtype 등의 메타데이터를 추출하기 위해 필요합니다.
   * HOW: _globalRegistry.get(handle)을 호출하여 얻어옵니다.
   */
  const record = _globalRegistry.get(handle);
  return {
    handle: record.handle,
    shape: [...record.shape],
    dtype: record.dtype,
    byteLength: record.byteLength,
    disposed: record.disposed
  };
}

/**
 * WHAT: 주어진 텐서의 데이터를 GPU에서 CPU로 비동기적으로 읽어 Float32Array로 반환하는 함수입니다.
 * WHY: 연산 결과가 포함된 GPU 버퍼의 데이터를 사용자나 프레임워크가 확인할 수 있도록 하기 위해 제공됩니다.
 * HOW: 레지스트리에서 버퍼를 조회하고 readBufferToFloat32Array 헬퍼를 사용해 데이터를 복사 후 반환합니다.
 */
export function read(handle: TensorHandle): Promise<Float32Array> {
  /**
   * WHAT: 핸들로 조회된 텐서 레코드 객체입니다.
   * WHY: 실제 GPUBuffer 참조와 버퍼 길이를 알아내기 위해 필요합니다.
   * HOW: _globalRegistry.get(handle) 호출을 통해 가져옵니다.
   */
  const record = _globalRegistry.get(handle);
  return readBufferToFloat32Array(record.buffer, record.byteLength);
}

/**
 * WHAT: 텐서 버퍼의 데이터를 읽기 위해 GPU 메모리를 매핑(map)하는 비동기 함수입니다.
 * WHY: 즉시 읽기(read)와 달리 맵핑과 데이터 복사를 분리하여 제로 카피(Zero Copy)나 스트리밍 최적화를 지원하기 위해 존재합니다.
 * HOW: 레지스트리에서 버퍼를 조회한 뒤 맵핑을 수행하고 반환된 스테이징 버퍼를 _pendingStagingBuffers에 저장합니다.
 */
export async function mapBufferAsync(handle: TensorHandle): Promise<void> {
  // If already staged and mapped, return immediately
  if (_pendingStagingBuffers.has(handle)) {
    return;
  }
  // If a mapping operation is already in-flight for this handle, coalesce with existing promise
  const inFlight = _inFlightMapPromises.get(handle);
  if (inFlight) {
    return inFlight;
  }

  const record = _globalRegistry.get(handle);
  const promise = (async () => {
    try {
      const { stagingBuffer, token } = await _mapBufferAsync(record.buffer, record.byteLength);
      _pendingStagingBuffers.set(handle, { stagingBuffer, token });
    } finally {
      _inFlightMapPromises.delete(handle);
    }
  })();

  _inFlightMapPromises.set(handle, promise);
  return promise;
}

/**
 * WHAT: 매핑이 완료된 스테이징 버퍼에서 대상 배열로 데이터를 동기 복사하는 함수입니다.
 * WHY: mapBufferAsync 호출 이후 실제 데이터를 사용자의 자바스크립트 버퍼 혹은 Pyodide 메모리로 옮기기 위해 사용됩니다.
 * HOW: _pendingStagingBuffers에서 버퍼를 찾아 실제 대상 배열(outArray)에 복사하고 스테이징 버퍼를 정리합니다.
 */
export function readMappedInto(handle: TensorHandle, outArray: any): void {
  /**
   * WHAT: 이전 mapBufferAsync 호출로 준비된 스테이징 버퍼 관련 정보 객체입니다.
   * WHY: 복사해올 실제 소스 버퍼에 접근하기 위해 맵에서 꺼내어 참조합니다.
   * HOW: _pendingStagingBuffers.get(handle)을 통해 조회합니다.
   */
  const obj = _pendingStagingBuffers.get(handle);
  if (!obj) {
    throw new Error(
      `[AMEVA] No staged buffer for handle "${handle}". Call mapBufferAsync first.`
    );
  }
  _pendingStagingBuffers.delete(handle);

  /**
   * WHAT: Pyodide나 WebAssembly 환경의 메모리 뷰를 감싸는 프록시 객체입니다.
   * WHY: 외부 WASM 메모리를 다룰 때 버퍼 포인터 획득과 해제를 안전하게 처리하기 위해 변수에 저장합니다.
   * HOW: 초기엔 null로 두고 outArray 타입에 따라 getBuffer() 결과가 할당됩니다.
   */
  let bufProxy: any = null;
  try {
    /**
     * WHAT: 데이터 복사가 기록될 최종 대상 Float32Array입니다.
     * WHY: 스테이징 버퍼의 데이터를 CPU가 직접 다룰 수 있는 형식으로 전달받기 위해 필요합니다.
     * HOW: bufProxy.data를 통해 참조를 얻거나 outArray 자체를 Float32Array로 캐스팅합니다.
     */
    let actualData: Float32Array;
    if (outArray && typeof outArray.getBuffer === 'function') {
      bufProxy = outArray.getBuffer("f32");
      actualData = bufProxy.data;
    } else {
      actualData = outArray as Float32Array;
    }
    
    // H-02 Fix: WASM 메모리 바운드 사전 검증
    if (actualData && actualData.buffer) {
      assertWasmRange(actualData.byteOffset, actualData.byteLength, actualData.buffer.byteLength);
    }
    
    // F-009 Fix: 대상 배열 크기와 원본 텐서 크기 검증
    const record = _globalRegistry.get(handle);
    if (actualData.byteLength !== record.byteLength) {
      throw new Error(
        `[AMEVA Forge] readMappedInto size mismatch. Expected ${record.byteLength} bytes, got ${actualData.byteLength} bytes.`
      );
    }

    _readMappedInto(obj.stagingBuffer, obj.token, actualData);
  } finally {
    // _readMappedInto already releases the token!
    // H-NEW-06: bufProxy.release() 실패 시에도 리소스 정리 보장
    if (bufProxy) {
      try { bufProxy.release(); } catch { /* ignore */ }
    }
  }
}

/**
 * WHAT: 사용을 마친 특정 텐서를 해제하는 함수입니다.
 * WHY: 외부 사용자가 더 이상 텐서 메모리를 사용하지 않을 때 메모리를 GPU에서 해제하기 위해 호출됩니다.
 * HOW: _globalRegistry.dispose()를 호출하여 핸들에 연결된 레코드를 삭제하고 버퍼 소멸 스케줄을 잡습니다.
 */
export function dispose(handle: TensorHandle): void {
  _globalRegistry.dispose(handle);
}

// ─────────────────────────────────────────────────────────────────────────────
// L-01 Fix: dispatchKernel 헬퍼 — 모든 op의 반복 코드를 통합
// NH-07 Fix: assertAllowedKernelName() 호출
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WHAT: dispatchKernel 함수에 전달되는 매개변수 객체의 인터페이스입니다.
 * WHY: 개별 오퍼레이션(add, mul 등)이 실행될 때 필요한 셰이더, 입력 버퍼, 차원(dispatch x, y) 등을 통일된 포맷으로 전달받기 위함입니다.
 * HOW: opKey, wgslCode, paramsData, inputBuffers, outBuffer, dispatchX, dispatchY 등의 속성을 가집니다.
 */
interface KernelDispatchOptions {
  opKey: string;
  wgslCode: string;
  paramsData: Uint32Array;
  inputBuffers: GPUBuffer[];
  outBuffer: GPUBuffer;
  dispatchX: number;
  dispatchY?: number;
}

/**
 * WHAT: 단일 WebGPU 컴퓨트 셰이더 커널을 디스패치(실행 요청)하는 공통 헬퍼 함수입니다.
 * WHY: 개별 연산 함수(add, sub 등)에 중복되는 버퍼 바인딩 및 파이프라인 생성 코드를 통합하여 유지보수성을 높이기 위해 존재합니다.
 * HOW: 유니폼 파라미터 버퍼를 생성하고 파이프라인 캐시를 조회한 뒤, 바인드 그룹을 설정하여 컴퓨트 패스를 큐에 제출합니다.
 */
function dispatchKernel(opts: KernelDispatchOptions): void {
  // NH-07 Fix: shaderGuard에서 커널 이름 검증
  assertAllowedKernelName(opts.opKey);

  /**
   * WHAT: WebGPU 작업을 제출할 대상 논리 디바이스입니다.
   * WHY: 커맨드 인코더 생성과 버퍼 조작을 위해 필요합니다.
   * HOW: getDevice() 함수를 호출하여 가져옵니다.
   */
  const device = getDevice();

  /**
   * WHAT: 셰이더로 전달될 스칼라 인자(크기, 차원 등)를 담는 GPU 유니폼 버퍼입니다.
   * WHY: GPU 셰이더 내에서 텐서 크기 등의 동적인 파라미터를 읽을 수 있어야 연산이 가능하기 때문입니다.
   * HOW: 최소 16바이트 정렬 크기를 만족하도록 디바이스에서 UNIFORM 용도로 할당합니다.
   */
  const { buffer: paramsBuffer, token: paramsToken } = allocateBuffer(
    Math.max(16, opts.paramsData.byteLength), // 최소 16바이트 (WebGPU uniform 정렬)
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    'uniform',
    `dispatchKernel_${opts.opKey}`
  );
  device.queue.writeBuffer(paramsBuffer, 0, opts.paramsData.buffer);

  // H-01: 파이프라인 캐시에서 조회 (없으면 컴파일 후 캐시)
  /**
   * WHAT: 컴파일이 완료된 WebGPU 컴퓨트 파이프라인 객체입니다.
   * WHY: 셰이더 코드를 기반으로 GPU가 작업을 어떻게 수행해야 하는지 구조를 알고 있어야 하기 때문입니다.
   * HOW: opKey와 wgslCode를 사용하여 _globalPipelineCache에서 가져옵니다.
   */
  const { pipeline } = _globalPipelineCache.getPipeline(opts.opKey, opts.wgslCode);

  /**
   * WHAT: 파이프라인에 바인딩될 리소스들의 배열(유니폼 버퍼, 입력 버퍼들, 출력 버퍼)입니다.
   * WHY: 셰이더의 각 바인딩 슬롯(binding 0, 1, 2...)에 정확한 버퍼를 매핑하기 위해 리스트로 준비합니다.
   * HOW: paramsBuffer를 binding 0에, 입력 버퍼들을 그 다음 순서에, 출력 버퍼를 마지막에 배치하여 구성합니다.
   */
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: paramsBuffer } },
    ...opts.inputBuffers.map((buf, i) => ({
      binding: i + 1,
      resource: { buffer: buf }
    })),
    { binding: opts.inputBuffers.length + 1, resource: { buffer: opts.outBuffer } }
  ];

  /**
   * WHAT: 준비된 entries를 기반으로 셰이더와 런타임 버퍼를 연결해주는 바인드 그룹 객체입니다.
   * WHY: 디바이스 커맨드 패스에 리소스 그룹을 설정하기 위해 필수적입니다.
   * HOW: device.createBindGroup을 통해 파이프라인의 레이아웃과 entries를 결합하여 생성합니다.
   */
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries
  });

  /**
   * WHAT: GPU 명령들을 기록하기 위한 커맨드 인코더입니다.
   * WHY: 복사, 컴퓨트 패스 등 여러 GPU 조작을 묶어서 큐에 제출하기 위해 사용됩니다.
   * HOW: device.createCommandEncoder()로 생성합니다.
   */
  const commandEncoder = device.createCommandEncoder();
  
  /**
   * WHAT: 컴퓨트 연산을 기록하는 패스 인코더입니다.
   * WHY: 파이프라인, 바인드 그룹, 디스패치 워크그룹 수 등을 설정하기 위해 필요합니다.
   * HOW: commandEncoder.beginComputePass()를 호출하여 가져옵니다.
   */
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(opts.dispatchX, opts.dispatchY ?? 1);
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);

  // params 버퍼는 GPU 제출 완료 후 중앙 allocator를 통해 해제
  void device.queue.onSubmittedWorkDone().then(() => {
    try { freeBuffer(paramsBuffer, paramsToken); } catch (e) { _safeLog(`[gpuCore] Failed to free params buffer: ${e}`); }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 개별 op 함수들 (내부 사용, pyodideBridge에서는 executeGraph를 통해서만 접근)
// NH-01 Note: 이 함수들은 JS 테스트와 직접 호출에서만 사용
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WHAT: 무작위 값(0~1)으로 채워진 지정된 형태(shape)의 텐서를 생성하는 함수입니다.
 * WHY: 신경망 가중치 초기화나 테스트 코드에서 임의의 데이터가 필요할 때 사용됩니다.
 * HOW: CPU(자바스크립트) 상에서 Float32Array 배열에 난수를 채우고 allocateBuffer로 얻은 GPU버퍼로 복사하여 레지스트리에 등록합니다.
 */
export function random(shape: number[], dtype: DType = "float32"): TensorHandle {
  validateDType(dtype);
  /**
   * WHAT: 텐서의 모든 차원을 곱해 산출된 총 원소의 개수입니다.
   * WHY: 1차원 Float32Array를 얼마나 크게 할당하고 루프를 돌릴지 결정하기 위해 계산됩니다.
   * HOW: validateShape 헬퍼를 통해 모양 검증과 동시에 산출됩니다.
   */
  const elements = validateShape(shape, dtype);
  
  /**
   * WHAT: CPU 메모리 상에 존재하는 실수 데이터 배열입니다.
   * WHY: GPU로 데이터를 전송하기 전 난수값을 임시로 기록하기 위해 할당합니다.
   * HOW: 원소 수(elements)만큼의 크기로 Float32Array를 생성합니다.
   */
  const data = new Float32Array(elements);
  
  /**
   * WHAT: 배열의 각 위치를 순회하며 난수를 채우는 반복문입니다.
   * WHY: 텐서 전체를 임의의 값으로 초기화하기 위해 실행됩니다.
   * HOW: i를 0부터 elements 전까지 증가시키며 Math.random() 값을 배열에 대입합니다.
   */
  for (let i = 0; i < elements; i++) data[i] = Math.random();
  
  /**
   * WHAT: 텐서 전체 데이터가 차지할 실제 바이트 크기입니다.
   * WHY: GPU 버퍼를 할당할 때 정확한 메모리 공간 크기가 필요하므로 계산합니다.
   * HOW: Float32 원소 개수에 4(바이트)를 곱합니다.
   */
  const byteLength = elements * 4;
  
  /**
   * WHAT: GPU 메모리 내에 새로 할당된 버퍼와 추적 토큰입니다.
   * WHY: 텐서 데이터를 영속적으로 저장하고 나중에 사용할 수 있도록 하기 위함입니다.
   * HOW: allocateBuffer 헬퍼를 사용하여 STORAGE, COPY_SRC, COPY_DST 용도로 버퍼를 생성합니다.
   */
  const { buffer, token } = allocateBuffer(
    byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  writeFloat32Array(buffer, data);
  return _globalRegistry.register({ buffer, token, shape, dtype, byteLength });
}

/**
 * WHAT: 기존의 Float32Array 데이터를 GPU 텐서로 업로드(복사)하여 핸들을 반환하는 함수입니다.
 * WHY: 외부 이미지 데이터나 입력 특징(feature) 배열을 GPU 메모리로 올려 연산을 수행할 수 있게 만들기 위해 존재합니다.
 * HOW: Pyodide 버퍼 프록시 혹은 일반 배열 데이터를 기반으로 GPU 버퍼를 할당하고 값을 복사한 후 레지스트리에 등록합니다.
 */
export function uploadFloat32Array(data: any, shape: number[]): TensorHandle {
  /**
   * WHAT: 업로드할 원본 데이터가 복사된 또는 참조된 Float32Array입니다.
   * WHY: WebGPU 버퍼에 쓰기 명령을 수행하려면 반드시 이 형태의 타입화된 배열이어야 하기 때문입니다.
   * HOW: 조건에 따라 bufProxy.data 또는 data 자체를 캐스팅하여 할당합니다.
   */
  let actualData: Float32Array;
  
  /**
   * WHAT: 외부 WASM 환경(Pyodide 등)에서 제공하는 버퍼 메모리 프록시 객체입니다.
   * WHY: 외부에 노출된 메모리 포인터 접근 후 자원 누수를 막기 위해 명시적인 해제(release)가 필요하기 때문에 변수에 잡아둡니다.
   * HOW: data 객체가 getBuffer 함수를 가지고 있으면 이를 호출하여 초기화하고 아니면 null을 유지합니다.
   */
  let bufProxy: any = null;
  if (data && typeof data.getBuffer === 'function') {
    bufProxy = data.getBuffer("f32");
    actualData = bufProxy.data;
  } else {
    actualData = data as Float32Array;
  }
  
  // H-02 Fix: WASM 메모리 바운드 사전 검증
  if (actualData && actualData.buffer) {
    assertWasmRange(actualData.byteOffset, actualData.byteLength, actualData.buffer.byteLength);
  }
  
  /**
   * WHAT: 입력된 형태(shape)가 지녀야 할 원소 총 개수입니다.
   * WHY: 형태 배열과 실제 전달된 배열의 바이트 길이가 일치하는지 검증하기 위해 필요합니다.
   * HOW: validateShape를 호출하며 actualData의 바이트 크기를 넘겨 정합성을 검사합니다.
   */
  const elements = validateShape(shape, "float32", actualData.byteLength);
  
  /**
   * WHAT: GPU에 할당될 메모리 총 바이트 수입니다.
   * WHY: allocateBuffer 헬퍼에 필요한 바이트 단위를 맞추기 위해 사용됩니다.
   * HOW: 산출된 원소 개수에 4를 곱합니다.
   */
  const byteLength = elements * 4;
  const { buffer, token } = allocateBuffer(
    byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  writeFloat32Array(buffer, actualData);
  if (bufProxy) bufProxy.release();
  return _globalRegistry.register({ buffer, token, shape, dtype: "float32", byteLength });
}

/**
 * WHAT: 두 개의 2차원 텐서에 대해 행렬 곱셈(Matmul)을 수행하는 함수입니다.
 * WHY: 신경망의 완전 연결층(Dense Layer)이나 어텐션 매커니즘 등 주요 선형 대수 연산을 지원하기 위해 존재합니다.
 * HOW: 두 텐서의 차원을 검증하고, 결과용 버퍼를 새로 생성한 뒤 matmul 셰이더를 dispatchKernel로 호출합니다.
 */
export function matmul(handleA: TensorHandle, handleB: TensorHandle): TensorHandle {
  /**
   * WHAT: 첫 번째 입력 행렬(A)의 레코드입니다.
   * WHY: A 행렬의 shape와 GPU 버퍼 포인터를 알아내기 위해 필요합니다.
   * HOW: 전역 레지스트리에서 handleA를 키로 조회합니다.
   */
  const a = _globalRegistry.get(handleA);
  
  /**
   * WHAT: 두 번째 입력 행렬(B)의 레코드입니다.
   * WHY: B 행렬의 shape와 메모리 버퍼를 확보하여 연산 인자로 쓰기 위해 필요합니다.
   * HOW: 전역 레지스트리에서 handleB로 조회합니다.
   */
  const b = _globalRegistry.get(handleB);

  if (a.shape.length !== 2 || b.shape.length !== 2)
    throw new AMEVAForgeShapeError("Matmul requires 2D tensors");
  if (a.dtype !== "float32" || b.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Matmul requires float32 tensors");

  /**
   * WHAT: A 행렬의 행, A의 열(B의 행), B의 행, B의 열을 나타내는 차원 변수들입니다.
   * WHY: 행렬 곱이 성립하기 위한 내부 차원(K) 일치 여부를 검사하고 워크그룹 수를 계산하기 위함입니다.
   * HOW: 각 텐서의 shape 배열에서 인덱스로 값을 구조 분해하여 할당합니다.
   */
  const M = a.shape[0], K = a.shape[1], K2 = b.shape[0], N = b.shape[1];
  if (K !== K2) throw new AMEVAForgeShapeError(`Inner dim mismatch: ${K} != ${K2}`);

  /**
   * WHAT: 결과 행렬(C)이 차지할 총 바이트 크기입니다.
   * WHY: 행렬 곱의 결과 텐서를 저장할 적절한 크기의 GPU 버퍼를 할당하기 위해 계산합니다.
   * HOW: 행 크기(M)와 열 크기(N)를 곱한 값에 float32 크기인 4를 곱합니다.
   */
  const byteLength = M * N * 4;
  const { buffer: cBuffer, token } = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  dispatchKernel({
    opKey: 'matmul',
    wgslCode: MATMUL_WGSL,
    paramsData: new Uint32Array([M, N, K, 0]),
    inputBuffers: [a.buffer, b.buffer],
    outBuffer: cBuffer,
    // M-05: X=col방향=N, Y=row방향=M
    dispatchX: Math.ceil(N / 8),
    dispatchY: Math.ceil(M / 8),
  });

  return _globalRegistry.register({ buffer: cBuffer, token, shape: [M, N], dtype: "float32", byteLength });
}

/**
 * WHAT: 주어진 텐서의 모든 원소에 대해 ReLU(Rectified Linear Unit) 활성화 함수를 적용하는 함수입니다.
 * WHY: 신경망에서 음수 값을 제거하여 비선형성을 부여하기 위해 핵심적인 오퍼레이션입니다.
 * HOW: 단일 텐서 버퍼를 읽고, 동일 크기의 출력 버퍼를 만든 후 relu 커널을 디스패치합니다.
 */
export function relu(handle: TensorHandle): TensorHandle {
  /**
   * WHAT: 입력 텐서 레코드입니다.
   * WHY: 연산 대상 데이터가 들어있는 GPU 버퍼와 크기를 가져오기 위함입니다.
   * HOW: 레지스트리에서 핸들로 조회합니다.
   */
  const x = _globalRegistry.get(handle);
  if (x.dtype !== "float32") throw new AMEVAForgeDTypeError("ReLU requires float32");
  
  /**
   * WHAT: 입력 텐서 내에 존재하는 실수 요소의 총 개수입니다.
   * WHY: 워크그룹 수를 계산하여 디스패치 크기를 결정하고 셰이더 내에서 배열 경계 검사를 수행하기 위해 필요합니다.
   * HOW: 총 바이트 길이를 4로 나누어 구합니다.
   */
  const numElements = x.byteLength / 4;
  const { buffer: outBuffer, token } = allocateBuffer(x.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const dispatch = computeDispatch2D(numElements, 64);

  dispatchKernel({
    opKey: 'relu',
    wgslCode: RELU_WGSL,
    paramsData: new Uint32Array([numElements, dispatch.workgroupsX, 0, 0]),
    inputBuffers: [x.buffer],
    outBuffer,
    dispatchX: dispatch.dispatchX,
    dispatchY: dispatch.dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [...x.shape], dtype: "float32", byteLength: x.byteLength });
}

/**
 * WHAT: 두 텐서 간의 요소별 덧셈(Element-wise Addition)을 수행하는 함수입니다.
 * WHY: 편향(bias) 더하기, 잔차 연결(residual connection) 등 신경망 연산에서 두 특징 맵을 합칠 때 사용됩니다.
 * HOW: 형태가 같은 두 텐서 버퍼를 넘겨받아 add 셰이더를 실행시키고 새로운 텐서를 생성해 반환합니다.
 */
export function add(handleA: TensorHandle, handleB: TensorHandle): TensorHandle {
  const a = _globalRegistry.get(handleA);
  const b = _globalRegistry.get(handleB);
  if (a.shape.length !== b.shape.length || !a.shape.every((v, i) => v === b.shape[i]))
    throw new AMEVAForgeShapeError("Add requires tensors of the exact same shape");
  if (a.dtype !== "float32" || b.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Add requires float32");

  const numElements = a.byteLength / 4;
  const { buffer: outBuffer, token } = allocateBuffer(a.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  const { dOut, effSA, effSB } = computeBroadcastParams(a.shape, a.shape, b.shape);
  const dispatch = computeDispatch2D(numElements, 64);
  const paramsData = new Uint32Array(28);
  paramsData[0] = numElements;
  paramsData[1] = dispatch.workgroupsX;
  paramsData[2] = a.shape.length;
  paramsData[3] = 0;
  for (let k = 0; k < 8; k++) paramsData[4 + k] = dOut[k];
  for (let k = 0; k < 8; k++) paramsData[12 + k] = effSA[k];
  for (let k = 0; k < 8; k++) paramsData[20 + k] = effSB[k];

  dispatchKernel({
    opKey: 'add',
    wgslCode: ADD_WGSL,
    paramsData,
    inputBuffers: [a.buffer, b.buffer],
    outBuffer,
    dispatchX: dispatch.dispatchX,
    dispatchY: dispatch.dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [...a.shape], dtype: "float32", byteLength: a.byteLength });
}

/**
 * WHAT: 두 텐서 간의 요소별 곱셈(Element-wise Multiplication)을 수행하는 함수입니다.
 * WHY: 어텐션 스코어 마스킹이나 활성화된 게이트 통과 등 데이터를 요소별로 가중치와 곱할 때 필요합니다.
 * HOW: 형태가 같은 두 텐서를 기반으로 mul 커널을 디스패치합니다.
 */
export function mul(handleA: TensorHandle, handleB: TensorHandle): TensorHandle {
  const a = _globalRegistry.get(handleA);
  const b = _globalRegistry.get(handleB);
  if (a.shape.length !== b.shape.length || !a.shape.every((v, i) => v === b.shape[i]))
    throw new AMEVAForgeShapeError("Mul requires tensors of the exact same shape");
  if (a.dtype !== "float32" || b.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Mul requires float32");

  const numElements = a.byteLength / 4;
  const { buffer: outBuffer, token } = allocateBuffer(a.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  const { dOut, effSA, effSB } = computeBroadcastParams(a.shape, a.shape, b.shape);
  const dispatch = computeDispatch2D(numElements, 64);
  const paramsData = new Uint32Array(28);
  paramsData[0] = numElements;
  paramsData[1] = dispatch.workgroupsX;
  paramsData[2] = a.shape.length;
  paramsData[3] = 0;
  for (let k = 0; k < 8; k++) paramsData[4 + k] = dOut[k];
  for (let k = 0; k < 8; k++) paramsData[12 + k] = effSA[k];
  for (let k = 0; k < 8; k++) paramsData[20 + k] = effSB[k];

  dispatchKernel({
    opKey: 'mul',
    wgslCode: MUL_WGSL,
    paramsData,
    inputBuffers: [a.buffer, b.buffer],
    outBuffer,
    dispatchX: dispatch.dispatchX,
    dispatchY: dispatch.dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [...a.shape], dtype: "float32", byteLength: a.byteLength });
}

/**
 * WHAT: 2차원 텐서(행렬)의 행과 열을 뒤집는 전치(Transpose) 연산을 수행하는 함수입니다.
 * WHY: 행렬 곱셈을 수행하기 전에 데이터의 축을 맞추거나 그래디언트 역전파를 위해 텐서를 변형할 때 사용됩니다.
 * HOW: 입력 형태(shape)의 [M, N]을 [N, M]으로 뒤집은 결과를 반환할 출력 버퍼에 기록하도록 transpose 셰이더를 실행합니다.
 */
export function transpose(handle: TensorHandle): TensorHandle {
  const x = _globalRegistry.get(handle);
  if (x.shape.length !== 2)
    throw new AMEVAForgeShapeError("Transpose requires 2D tensors");
  if (x.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Transpose requires float32");

  const M = x.shape[0], N = x.shape[1];
  const { buffer: outBuffer, token } = allocateBuffer(x.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  dispatchKernel({
    opKey: 'transpose',
    wgslCode: TRANSPOSE_WGSL,
    paramsData: new Uint32Array([M, N, 1, 0]),
    inputBuffers: [x.buffer],
    outBuffer,
    dispatchX: Math.ceil(M / 8),
    dispatchY: Math.ceil(N / 8),
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [N, M], dtype: "float32", byteLength: x.byteLength });
}

/**
 * WHAT: ReLU 활성화 함수의 도함수(그래디언트)를 계산하여 역전파(Backward)를 수행하는 함수입니다.
 * WHY: 오차 역전파 과정에서 순전파 시 입력값이 0 이상이었던 위치에만 상위 그래디언트를 흘려보내기 위해 필요합니다.
 * HOW: 원본 입력 텐서(x)와 위층에서 전달된 그래디언트 텐서(grad)를 받아, x가 0보다 큰 곳은 grad를, 아니면 0을 출력 버퍼에 씁니다.
 */
export function relu_backward(handleX: TensorHandle, handleGrad: TensorHandle): TensorHandle {
  const x = _globalRegistry.get(handleX);
  const grad = _globalRegistry.get(handleGrad);
  if (x.shape.length !== grad.shape.length || !x.shape.every((v, i) => v === grad.shape[i]))
    throw new AMEVAForgeShapeError("ReLU backward: shape mismatch");

  const numElements = x.byteLength / 4;
  const { buffer: outBuffer, token } = allocateBuffer(x.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const dispatch = computeDispatch2D(numElements, 64);

  dispatchKernel({
    opKey: 'relu_backward',
    wgslCode: RELU_BACKWARD_WGSL,
    paramsData: new Uint32Array([numElements, dispatch.workgroupsX, 0, 0]),
    inputBuffers: [x.buffer, grad.buffer],
    outBuffer,
    dispatchX: dispatch.dispatchX,
    dispatchY: dispatch.dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [...x.shape], dtype: "float32", byteLength: x.byteLength });
}

export const gpuCore = {
  add,
  mul,
  matmul,
  relu,
  relu_backward,
  transpose,
};


```

---

## `packages/forge/src/tensor/graphExecutor.ts`

```typescript
/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:59:35+09:00: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 *   - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * graphExecutor.ts — JSON 그래프 파서 & GPU 스케줄러
 *
 * C-04 Fix: JSON 입력에 대한 강력한 검증 추가
 * M-05 Fix: matmul dispatch X/Y swap 수정
 * H-01 Fix: _globalPipelineCache를 모든 op에 적용
 * NC-06 Fix: inst.in null-guard 추가 (! 비null 단언 제거)
 * NH-07 Fix: shaderGuard.assertAllowedKernelName() 실제 호출
 * NM-05 Fix: device.pushErrorScope()로 op별 에러 감지
 */

import { getDevice } from "../webgpu/device";
import { _globalRegistry, TensorRegistry } from "./tensorRegistry";
import { TensorHandle, DType } from "../types";
import { allocateBuffer, writeFloat32Array, freeBuffer } from "../webgpu/buffers";
import { _globalQuotaManager, AllocationToken } from "../webgpu/quota";
import { AMEVAForgeShapeError, AMEVAForgeSecurityError, AMEVAForgeUnsupportedOpError, AMEVAForgeValidationError, AMEVAForgeOutOfMemoryError, AMEVAForgeInternalGPUError } from "../errors";
import { assertAllowedKernelName } from "../webgpu/shaderGuard";
import { assertWasmRange } from "../webgpu/validateWasmRange";
import { _globalPipelineCache } from "../webgpu/pipelineCache";
import { computeBroadcastParams } from "./broadcastParams";
import { computeDispatch2D } from "./dispatchShape";
import { _globalUniformPool, UniformEntry } from "../webgpu/uniformPool";


// kernels
import { MATMUL_WGSL } from "./kernels/matmul.wgsl";
import { MATMUL_BIAS_RELU_WGSL } from "./kernels/matmul_bias_relu.wgsl";
import { BATCHED_MATMUL_WGSL } from "./kernels/batched_matmul.wgsl";
import { RELU_WGSL } from "./kernels/relu.wgsl";
import { ADD_WGSL } from "./kernels/add.wgsl";
import { MUL_WGSL } from "./kernels/mul.wgsl";
import { TRANSPOSE_WGSL } from "./kernels/transpose.wgsl";
import { RELU_BACKWARD_WGSL } from "./kernels/relu_backward.wgsl";
import { SUB_WGSL } from "./kernels/sub.wgsl";
import { NEG_WGSL } from "./kernels/neg.wgsl";
import { DIV_WGSL } from "./kernels/div.wgsl";
import { EXP_WGSL } from "./kernels/exp.wgsl";
import { LOG_WGSL } from "./kernels/log.wgsl";
import { SIGMOID_WGSL } from "./kernels/sigmoid.wgsl";
import { TANH_WGSL } from "./kernels/tanh.wgsl";
import { SIGMOID_BACKWARD_WGSL } from "./kernels/sigmoid_backward.wgsl";
import { TANH_BACKWARD_WGSL } from "./kernels/tanh_backward.wgsl";
import { FILL_WGSL } from "./kernels/fill.wgsl";
import { SUM_WGSL } from "./kernels/sum.wgsl";
import { MAX_WGSL } from "./kernels/max.wgsl";
import { SUM_AXIS_WGSL } from "./kernels/sum_axis.wgsl";
import { MAX_AXIS_WGSL } from "./kernels/max_axis.wgsl";
import { MAX_AXIS_BACKWARD_WGSL } from "./kernels/max_axis_backward.wgsl";
import { AXPY_WGSL } from "./kernels/axpy.wgsl";
import { MAXPOOL2D_WGSL } from "./kernels/maxpool2d.wgsl";
import { AVGPOOL2D_WGSL } from "./kernels/avgpool2d.wgsl";
import { IM2COL_WGSL } from "./kernels/im2col.wgsl";
import { COL2IM_WGSL } from "./kernels/col2im.wgsl";
import { PAD_WGSL } from "./kernels/pad.wgsl";
import { GATHER_WGSL } from "./kernels/gather.wgsl";
import { SCATTER_WGSL } from "./kernels/scatter.wgsl";
import { CAT_WGSL } from "./kernels/cat.wgsl";
import { WHERE_WGSL } from "./kernels/where.wgsl";
import { DROPOUT_WGSL } from "./kernels/dropout.wgsl";
import { PERMUTE_WGSL } from "./kernels/permute.wgsl";

/** 
 * WHAT: 그래프 실행기가 처리할 수 있는 모든 허용된 오퍼레이션(op)의 집합입니다.
 * WHY: 악의적인 JSON 그래프가 알 수 없거나 금지된 셰이더를 실행하여 GPU를 공격하는 것을 방지하기 위한 화이트리스트입니다.
 * HOW: Set 자료구조에 허용되는 오퍼레이션 문자열을 초기화하여 빠른 조회(O(1))를 제공합니다.
 */
const ALLOWED_OPS = new Set([
  'upload', 'load', 'matmul', 'batched_matmul', 'relu', 'add', 'mul', 'transpose', 'relu_backward',
  'sub', 'neg', 'div', 'exp', 'log', 'sigmoid', 'tanh', 'sigmoid_backward', 'tanh_backward',
  'fill', 'sum', 'max', 'sum_axis', 'max_axis', 'max_axis_backward', 'axpy', 'cat', 'where', 'pad', 'gather', 'scatter', 'maxpool2d', 'avgpool2d',
  'im2col', 'col2im', 'dropout', 'permute', 'matmul_bias_relu', 'reshape'
]);

export type ForgeRuntimeConfig = {
  workloadBudgetElements?: number;
  maxOpsPerSubmit?: number;
  maxShapeDim?: number;
  maxElements?: number;
  maxInstructions?: number;
  allowNonFinite?: boolean;
};

const DEFAULT_RUNTIME_CONFIG: Required<ForgeRuntimeConfig> = {
  workloadBudgetElements: 100_000_000,
  maxOpsPerSubmit: 256,
  maxShapeDim: 8,
  maxElements: 256 * 1024 * 1024,
  maxInstructions: 10_000,
  allowNonFinite: false,
};

let _runtimeConfig = { ...DEFAULT_RUNTIME_CONFIG };

export function configureRuntime(config: ForgeRuntimeConfig): void {
  _runtimeConfig = {
    ..._runtimeConfig,
    ...config,
  };
}

export function getRuntimeConfig(): Required<ForgeRuntimeConfig> {
  return { ..._runtimeConfig };
}

const BUFFER_USAGE_STORAGE_COPY = typeof GPUBufferUsage !== 'undefined'
  ? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST)
  : (0x0080 | 0x0004 | 0x0008);

const BUFFER_USAGE_UNIFORM_COPY = typeof GPUBufferUsage !== 'undefined'
  ? (GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
  : (0x0040 | 0x0008);

const BUFFER_USAGE_STORAGE_SRC = typeof GPUBufferUsage !== 'undefined'
  ? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
  : (0x0080 | 0x0004);

/**
 * WHAT: 단일 텐서 연산을 지시하는 그래프 명령어의 데이터 타입 인터페이스입니다.
 * WHY: JSON 형태의 무타입 입력 데이터를 검증하고, 이후의 컴파일 과정에서 정적 타입 체크를 하기 위해 존재합니다.
 * HOW: 연산 종류(op), 식별자(id), 차원(shape), 입력 배열(in), 파라미터(params) 등의 속성을 정의합니다.
 */
interface GraphInstruction {
  op: string;
  id: number;
  shape: number[];
  in?: number[];
  handle?: string;
  params?: number[];
}

export interface PendingTensorRecord {
  handle: TensorHandle;
  buffer: GPUBuffer;
  token: AllocationToken;
  shape: number[];
  dtype: DType;
  byteLength: number;
}

function _safeLog(msg: string): void {
  try {
    if (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'development') {
      console.warn(msg);
    }
  } catch { /* intentionally empty: _safeLog is the outermost logging fallback, catching here prevents infinite recursion */ }
}

interface DeferredBufferRecord {
  buffer: GPUBuffer;
  token: AllocationToken;
  retries: number;
}

const _deferredGCQueue: DeferredBufferRecord[] = [];

/**
 * WHAT: 롤백 과정에서 즉시 destroy에 실패한 GPU 버퍼들의 지연 해제를 재시도합니다.
 * WHY: 일시적 GPU busy 상태 등으로 파괴 실패 시 유령 VRAM 누수를 방지합니다.
 */
export function processDeferredGC(): void {
  for (let i = _deferredGCQueue.length - 1; i >= 0; i--) {
    const item = _deferredGCQueue[i];
    try {
      item.buffer.destroy();
      _globalQuotaManager.releaseToken(item.token);
      _deferredGCQueue.splice(i, 1);
    } catch (e) {
      item.retries++;
      if (item.retries >= 3) {
        try {
          _globalQuotaManager.markOrphaned(item.token, String(e));
        } catch (err) { _safeLog(`[DeferredGC] markOrphaned failed: ${err}`); }
        _deferredGCQueue.splice(i, 1);
        _safeLog(`[DeferredGC] Failed to destroy buffer after 3 attempts, token marked orphaned: ${item.token.id}`);
      }
    }
  }
  
  if (_deferredGCQueue.length > 100) {
    _safeLog(`[DeferredGC] WARNING: ${_deferredGCQueue.length} items still pending after flush`);
  }
}

export class GraphTransaction {
  private readonly pending = new Map<TensorHandle, PendingTensorRecord>();

  add(record: PendingTensorRecord): void {
    if (this.pending.has(record.handle)) {
      throw new AMEVAForgeValidationError(`Duplicate pending handle: ${record.handle}`);
    }
    this.pending.set(record.handle, record);
  }

  get(handle: TensorHandle): PendingTensorRecord | undefined {
    return this.pending.get(handle);
  }

  get handles(): TensorHandle[] {
    return Array.from(this.pending.keys());
  }

  commit(registry: TensorRegistry): void {
    for (const record of this.pending.values()) {
      registry.registerRecord(record);
    }
    this.pending.clear();
  }

  rollback(): void {
    for (const record of this.pending.values()) {
      try {
        record.buffer.destroy();
        _globalQuotaManager.releaseToken(record.token);
      } catch (e) {
        _safeLog(`[GraphTransaction.rollback] Buffer destroy failed, queued for deferred GC: ${e}`);
        _deferredGCQueue.push({
          buffer: record.buffer,
          token: record.token,
          retries: 0
        });
      }
    }
    this.pending.clear();
    processDeferredGC();
  }
}


/**
 * WHAT: JSON에서 파싱된 단일 명령어 객체의 무결성을 엄격하게 검증하는 함수입니다.
 * WHY: 타입 오류나 범위 초과 등을 가진 악성 데이터가 하위 WebGPU 계층으로 흘러가 충돌을 일으키지 않도록 방어하기 위함입니다.
 * HOW: 속성의 존재 유무와 타입, 배열 길이, 연산의 결과 오버플로우 등을 꼼꼼하게 검사합니다.
 */
function validateInstruction(inst: unknown, idx: number): GraphInstruction {
  if (typeof inst !== 'object' || inst === null) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: must be an object`);
  }

  /**
   * WHAT: 타입 캐스팅을 위해 임시로 생성된 레코드 변수입니다.
   * WHY: unknown 타입을 Record<string, unknown>으로 변환하여 속성에 동적으로 접근하기 위해 필요합니다.
   * HOW: inst를 타입 단언(as)으로 캐스팅합니다.
   */
  const i = inst as Record<string, unknown>;

  if (typeof i.op !== 'string') {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: op must be a string`);
  }
  if (!ALLOWED_OPS.has(i.op)) {
    throw new AMEVAForgeUnsupportedOpError(`Instruction[${idx}]: unknown op "${i.op}"`);
  }

  if (!Number.isSafeInteger(i.id) || (i.id as number) < 1) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: id must be a positive safe integer`);
  }

  if (!Array.isArray(i.shape)) {
    throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape must be an array`);
  }
  // NM-06: rank 0 허용 (스칼라)
  if (i.shape.length > _runtimeConfig.maxShapeDim) {
    throw new AMEVAForgeShapeError(
      `Instruction[${idx}]: shape rank must be 0–${_runtimeConfig.maxShapeDim}, got ${i.shape.length}`
    );
  }

  /**
   * WHAT: 해당 명령어 텐서의 누적 원소 수를 계산하는 변수입니다.
   * WHY: 차원의 곱이 안전한 정수 범위를 넘거나 최대 한계(_runtimeConfig.maxElements)를 초과하는지 확인하기 위해 계산합니다.
   * HOW: 루프를 통해 차원(dim)을 곱하여 누적합니다. 초기값은 스칼라 연산을 위해 1로 시작합니다.
   */
  let elements = 1;
  
  /**
   * WHAT: shape 배열의 각 차원에 대해 안전성을 검사하는 루프입니다.
   * WHY: 음수 차원, 부동소수점 차원, 정수 오버플로우로 인한 악의적 크기 공격을 차단하기 위해 순회합니다.
   * HOW: for...of 구문으로 각 차원(dim)을 검사하고 elements 변수에 곱합니다.
   */
  for (const dim of i.shape) {
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new AMEVAForgeShapeError(
        `Instruction[${idx}]: shape dim must be a positive safe integer, got ${dim}`
      );
    }
    if (dim > Number.MAX_SAFE_INTEGER / elements) {
      throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape product integer overflow`);
    }
    elements *= dim;
  }
  if (elements > _runtimeConfig.maxElements) {
    throw new AMEVAForgeShapeError(
      `Instruction[${idx}]: tensor too large (${elements} elements > ${_runtimeConfig.maxElements})`
    );
  }

  // NC-06: in 필드가 있으면 배열인지 확인
  if (i.in !== undefined && !Array.isArray(i.in)) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: 'in' field must be an array`);
  }
  if (i.params !== undefined && !Array.isArray(i.params)) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: 'params' field must be an array`);
  }
  
  // F-017 Fix: 각 커널별 엄격한 스키마 검증 (in 개수 및 params 길이 강제)
  const OP_SCHEMA: Record<string, { minIn: number, exactIn?: boolean, minParams: number, exactParams?: boolean }> = {
    'upload': { minIn: 0, exactIn: true, minParams: 0, exactParams: true },
    'load': { minIn: 0, exactIn: true, minParams: 0, exactParams: true },
    'fill': { minIn: 0, exactIn: true, minParams: 2, exactParams: true },
    'sum': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'max': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'relu': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'exp': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'log': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'sigmoid': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'tanh': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'neg': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'relu_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: false },
    'sigmoid_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'tanh_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'pad': { minIn: 1, exactIn: true, minParams: 9, exactParams: false }, // pad는 최대 8차원 144바이트 = 36 uint32s.
    'sum_axis': { minIn: 1, exactIn: true, minParams: 2, exactParams: false },
    'max_axis': { minIn: 1, exactIn: true, minParams: 2, exactParams: false },
    'max_axis_backward': { minIn: 2, exactIn: true, minParams: 2, exactParams: false },
    'dropout': { minIn: 1, exactIn: true, minParams: 2, exactParams: true },
    'maxpool2d': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'avgpool2d': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'im2col': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'col2im': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'transpose': { minIn: 1, exactIn: true, minParams: 2, exactParams: false },
    'permute': { minIn: 1, exactIn: true, minParams: 1, exactParams: false }, // rank 길이 가변
    'reshape': { minIn: 1, exactIn: true, minParams: 0, exactParams: false },
    'add': { minIn: 2, exactIn: true, minParams: 0, exactParams: false },
    'sub': { minIn: 2, exactIn: true, minParams: 0, exactParams: false },
    'mul': { minIn: 2, exactIn: true, minParams: 0, exactParams: false },
    'div': { minIn: 2, exactIn: true, minParams: 0, exactParams: false },
    'axpy': { minIn: 2, exactIn: true, minParams: 2, exactParams: false },
    'gather': { minIn: 2, exactIn: true, minParams: 7, exactParams: false },
    'scatter': { minIn: 2, exactIn: false, minParams: 7, exactParams: false },
    'matmul': { minIn: 2, exactIn: true, minParams: 3, exactParams: true },
    'matmul_bias_relu': { minIn: 3, exactIn: true, minParams: 3, exactParams: true },
    'batched_matmul': { minIn: 2, exactIn: true, minParams: 4, exactParams: false },
    'where': { minIn: 3, exactIn: true, minParams: 0, exactParams: false },
    'cat': { minIn: 2, exactIn: false, minParams: 1, exactParams: false } // 가변 개수 입력, params는 axis 등
  };

  const opStr = i.op as string;
  const schema = OP_SCHEMA[opStr];
  if (schema) {
    const inLen = i.in ? (i.in as unknown[]).length : 0;
    const pLen = i.params ? (i.params as unknown[]).length : 0;
    
    if (schema.exactIn && inLen !== schema.minIn) {
      throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected exact ${schema.minIn} inputs, got ${inLen}`);
    } else if (inLen < schema.minIn) {
      throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected min ${schema.minIn} inputs, got ${inLen}`);
    }
    
    if (schema.exactParams && pLen !== schema.minParams) {
      throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected exact ${schema.minParams} params, got ${pLen}`);
    } else if (pLen < schema.minParams) {
      throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected min ${schema.minParams} params, got ${pLen}`);
    }
  }

  // params 타입 검증 (전부 안전한 number 이어야 함)
  if (i.params) {
    for (const p of i.params as unknown[]) {
      if (typeof p !== 'number' || !Number.isFinite(p)) {
        throw new AMEVAForgeSecurityError(`Instruction[${idx}]: param must be a finite number`);
      }
    }
  }

  return i as unknown as GraphInstruction;
}

/**
 * executeGraph — Python 레이지 그래프를 단일 FFI 호출로 GPU에 실행한다.
 * WHAT: Python 등 외부 환경에서 직렬화된 연산 그래프(JSON)를 받아 일괄적으로 GPU에서 실행하는 함수입니다.
 * WHY: 매 연산마다 JS와 WebAssembly/GPU 사이를 왕복(context switch)하면 극심한 오버헤드가 발생하므로, 한 번의 호출로 많은 명령을 처리(Transaction)하기 위해 설계되었습니다.
 * HOW: JSON을 파싱하고, 명령을 검증하며, 적절한 청크로 분할하여 WebGPU 커맨드 버퍼에 기록하고 제출(submit)합니다. 실패 시 트랜잭션을 롤백합니다.
 */
let _executionQueueChain: Promise<any> = Promise.resolve();

export async function executeGraph(
  instructionsJson: string,
  inputs: (Float32Array | any)[],
  outputIds?: number[]
): Promise<Record<string, TensorHandle>> {
  const previous = _executionQueueChain;
  let releaseLock: () => void;
  _executionQueueChain = new Promise<void>((resolve) => {
    releaseLock = resolve;
  });

  try {
    await previous;
  } catch {
    // Suppress previous transaction error so queue continues processing
  }

  try {
    return await _executeGraphCore(instructionsJson, inputs);
  } finally {
    releaseLock!();
  }
}

async function _executeGraphCore(
  instructionsJson: string,
  jsInputs: unknown
): Promise<Record<number, TensorHandle>> {
  // Flush any pending deferred GC items before new execution
  processDeferredGC();

  // ── 1. Parse ──
  let rawInstructions: unknown[];
  try {
    rawInstructions = JSON.parse(instructionsJson, (key, value) => {
      // M-01 Fix: JSON Prototype Pollution 방어
      if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
        throw new AMEVAForgeSecurityError(`Forbidden property name in JSON: ${key}`);
      }
      return value;
    });
  } catch (e) {
    if (e instanceof AMEVAForgeSecurityError) throw e;
    throw new AMEVAForgeSecurityError("executeGraph: invalid JSON in instructionsJson");
  }

  if (!Array.isArray(rawInstructions)) {
    throw new AMEVAForgeSecurityError("executeGraph: instructionsJson must be a JSON array");
  }
  if (rawInstructions.length > _runtimeConfig.maxInstructions) {
    throw new AMEVAForgeSecurityError(
      `executeGraph: too many instructions (${rawInstructions.length} > ${_runtimeConfig.maxInstructions})`
    );
  }

  // ── 2. Validate ──
  const instructions: GraphInstruction[] = rawInstructions.map(validateInstruction);

  // VULN-06: Ensure AXPY is only executed in the optimizer commit phase and not followed by downstream ops
  let seenAxpy = false;
  for (const inst of instructions) {
    if (inst.op === 'axpy') {
      seenAxpy = true;
    } else if (seenAxpy) {
      throw new AMEVAForgeSecurityError(
        `Invalid graph execution: In-place 'axpy' is an optimizer commit phase operation and cannot be followed by downstream op '${inst.op}' in the same transaction.`
      );
    }
  }

  let inputs: unknown[];
  if (jsInputs && typeof (jsInputs as any).toJs === 'function') {
    inputs = (jsInputs as any).toJs();
  } else if (Array.isArray(jsInputs)) {
    inputs = jsInputs;
  } else {
    inputs = [];
  }

  // ── 3. Plan ──
  // (In the future: calculate peak memory, check dependency DAG)
  
  // ── 4. Execute ──
  const device = getDevice();
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('internal');

  let commandEncoder = device.createCommandEncoder();
  let opsInCurrentBatch = 0;
  let encoderHasCommands = false;
  let workloadElements = 0;
  
  const idToHandle: Record<number, TensorHandle> = {};
  const idToBuffer: Record<number, GPUBuffer> = {};
  const idToByteLength: Record<number, number> = {};
  const idToShape: Record<number, number[]> = {};
  const transaction = new GraphTransaction();
  let inputIdx = 0;
  
  const paramsAllocations: Array<{ buffer: GPUBuffer, token: AllocationToken, isUniformPool?: boolean, uniformEntry?: UniformEntry }> = [];

  try {
    /**
     * WHAT: 검증된 각 그래프 명령어를 순차적으로 순회하며 GPU 작업으로 변환하는 메인 루프입니다.
     * WHY: 계획된 그래프 연산들을 실제 WebGPU 파이프라인 디스패치로 번역하기 위해 반드시 실행해야 합니다.
     * HOW: for...of 구문을 사용하여 instructions 배열의 각 객체(inst)를 처리합니다.
     */
    for (const inst of instructions) {
      /**
       * WHAT: 현재 명령어가 결과로 생성할 텐서의 바이트 크기입니다.
       * WHY: 결과를 담을 출력 버퍼(OutBuffer)의 크기를 GPU에 요청할 때 필요합니다.
       * HOW: 배열 차원(shape)을 모두 곱한 뒤, float32 크기(4)를 곱하여 계산합니다.
       */
      const byteLength = inst.shape.reduce((a, b) => a * b, 1) * 4;

      if (inst.op === 'load') {
        /**
         * WHAT: load 명령에 전달된 기존 텐서의 핸들 문자열입니다.
         * WHY: 이미 VRAM에 존재하는 텐서를 그래프의 내부 ID에 매핑하여 입력으로 사용하기 위해 필요합니다.
         * HOW: inst.handle 속성을 읽어오고 유효성을 검증합니다.
         */
        const handle = inst.handle;
        if (typeof handle !== 'string') {
          throw new AMEVAForgeSecurityError(`load instruction missing handle`);
        }
        
        if (!_globalRegistry.has(handle)) {
          console.error(`[GraphExecutor DIAGNOSTIC] load op failed for handle="${handle}". Registered handles count=${_globalRegistry.snapshotHandles().length}, list=${JSON.stringify(_globalRegistry.snapshotHandles())}`);
        }
        const record = _globalRegistry.get(handle);
        // F-018 Fix: JSON 형상과 레지스트리 실제 형상 일치 여부 검사
        if (inst.shape.length !== record.shape.length || !inst.shape.every((v, i) => v === record.shape[i])) {
          throw new AMEVAForgeShapeError(`load instruction shape mismatch for handle ${handle}. Expected [${record.shape}], got [${inst.shape}]`);
        }
        
        idToHandle[inst.id] = handle;
        idToBuffer[inst.id] = record.buffer;
        idToByteLength[inst.id] = record.byteLength;
        idToShape[inst.id] = record.shape;
        continue;
      }

      if (inst.op === 'upload') {
        const rawData = inputs[inputIdx++];
        let actualData: Float32Array;
        let bufProxy: any = null;

        if (rawData && typeof (rawData as any).getBuffer === 'function') {
          bufProxy = (rawData as any).getBuffer("f32");
          actualData = bufProxy.data;
        } else if (rawData instanceof Float32Array) {
          actualData = rawData;
        } else if (rawData && typeof (rawData as any).toJs === 'function') {
          const converted = (rawData as any).toJs();
          actualData = converted instanceof Float32Array ? converted : new Float32Array(converted);
        } else {
          throw new AMEVAForgeSecurityError(`upload input[${inputIdx - 1}] is not a Float32Array`);
        }

        // H-02 Fix: WASM 메모리 바운드 및 Detached 버퍼 사전 검증
        if (actualData && actualData.buffer) {
          const buf = actualData.buffer as any;
          if (buf.detached === true || actualData.byteLength === 0) {
            if (bufProxy) bufProxy.release();
            throw new AMEVAForgeSecurityError(`upload input[${inputIdx - 1}] buffer is detached (WASM heap growth)`);
          }
          assertWasmRange(actualData.byteOffset, actualData.byteLength, actualData.buffer.byteLength);
        }

        // VULN-10: NaN / Inf fail-fast check (Strictly governed by trusted ForgeRuntimeConfig)
        const allowNonFinite = _runtimeConfig.allowNonFinite === true;
        for (let i = 0; i < actualData.length; i++) {
          if (!Number.isFinite(actualData[i])) {
            if (allowNonFinite) {
              _safeLog(`[GraphExecutor] Non-finite value in upload input[${inputIdx - 1}] allowed by runtime config`);
            } else {
              if (bufProxy) bufProxy.release();
              throw new AMEVAForgeValidationError(
                `Invalid tensor data: upload input[${inputIdx - 1}] contains NaN or Infinity at index ${i}. ` +
                `Configure runtime allowNonFinite=true to bypass if intended.`
              );
            }
          }
        }

        const { buffer, token } = allocateBuffer(
          byteLength,
          BUFFER_USAGE_STORAGE_COPY,
          'tensor',
          `Graph_${instructions[0]?.id}`
        );
        try {
          writeFloat32Array(buffer, actualData);
        } finally {
          if (bufProxy) bufProxy.release();
        }

        const uuid = typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).substring(2, 15);
        const handle = `tensor_${uuid}`;
        transaction.add({
          handle,
          buffer,
          token,
          shape: inst.shape,
          dtype: "float32",
          byteLength
        });
        idToHandle[inst.id] = handle;
        idToBuffer[inst.id] = buffer;
        idToByteLength[inst.id] = byteLength;
        idToShape[inst.id] = inst.shape;
        continue;
      }

      if (inst.op === 'reshape') {
        if (!inst.in || inst.in.length < 1) {
          throw new AMEVAForgeSecurityError(`reshape instruction missing 'in' tensor`);
        }
        const inBuf = idToBuffer[inst.in[0]];
        const inByteLength = idToByteLength[inst.in[0]];
        if (!inBuf) {
          throw new AMEVAForgeSecurityError(`reshape input tensor not found for id ${inst.in[0]}`);
        }
        if (inByteLength !== byteLength) {
          throw new AMEVAForgeShapeError(
            `reshape size mismatch: input has ${inByteLength / 4} elements, output has ${byteLength / 4} elements`
          );
        }

        const { buffer: outBuffer, token } = allocateBuffer(
          byteLength,
          BUFFER_USAGE_STORAGE_COPY,
          'tensor',
          `Graph_${instructions[0]?.id}`
        );
        commandEncoder.copyBufferToBuffer(inBuf, 0, outBuffer, 0, byteLength);
        encoderHasCommands = true;

        const uuid = typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).substring(2, 15);
        const handle = `tensor_${uuid}`;

        transaction.add({
          handle,
          buffer: outBuffer,
          token,
          shape: inst.shape,
          dtype: "float32",
          byteLength
        });

        idToHandle[inst.id] = handle;
        idToBuffer[inst.id] = outBuffer;
        idToByteLength[inst.id] = byteLength;
        idToShape[inst.id] = inst.shape;
        continue;
      }

      assertAllowedKernelName(inst.op);

      let outBuffer: GPUBuffer;
      if (inst.op === 'axpy') {
        if (!inst.in || inst.in.length < 2) {
          throw new AMEVAForgeSecurityError(`Instruction axpy is missing 'in' fields.`);
        }
        outBuffer = idToBuffer[inst.in[1]];
        idToHandle[inst.id] = idToHandle[inst.in[1]];
        idToBuffer[inst.id] = outBuffer;
        idToByteLength[inst.id] = byteLength;
        idToShape[inst.id] = inst.shape;
      } else {
        const { buffer, token } = allocateBuffer(
          byteLength,
          BUFFER_USAGE_STORAGE_COPY,
          'tensor',
          `Graph_${instructions[0]?.id}`
        );
        outBuffer = buffer;
        const uuid = typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).substring(2, 15);
        const handle = `tensor_${uuid}`;
        transaction.add({
          handle,
          buffer: outBuffer,
          token,
          shape: inst.shape,
          dtype: "float32",
          byteLength
        });
        idToHandle[inst.id] = handle;
        idToBuffer[inst.id] = outBuffer;
        idToByteLength[inst.id] = byteLength;
        idToShape[inst.id] = inst.shape;
      }

      /**
       * WHAT: 현재 오퍼레이션의 유니폼 파라미터를 담기 위해 필요한 바이트 크기입니다.
       * WHY: 오퍼레이션(패딩, 풀링 등)마다 셰이더가 요구하는 인자의 종류와 개수가 다르므로 가변적인 버퍼 크기를 잡기 위해 결정합니다.
       * HOW: inst.op 문자열을 판별하여 필요한 바이트 수(최소 32바이트)를 할당합니다.
       */
      let paramsSize = 32;
      if (inst.op === 'pad') paramsSize = 144;
      else if (inst.op === 'gather' || inst.op === 'scatter') paramsSize = 112;
      else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d') paramsSize = 64;
      else if (inst.op === 'im2col' || inst.op === 'col2im') paramsSize = 48;
      else if (inst.op === 'permute') paramsSize = 112;
      else if (['add', 'sub', 'mul', 'div'].includes(inst.op)) paramsSize = 112;

      const { buffer: paramsBuffer, token: paramsToken } = allocateBuffer(
        paramsSize,
        BUFFER_USAGE_UNIFORM_COPY,
        'uniform',
        `Graph_${instructions[0]?.id}_params`
      );
      paramsAllocations.push({ buffer: paramsBuffer, token: paramsToken });

      let wgslCode = "";
      let dispatchX = 1, dispatchY = 1, dispatchZ = 1;
      let isMatmul = false;
      let B = 1, M = 1, N = 1, K = 1;

      if (inst.op === 'matmul' || inst.op === 'matmul_bias_relu') {
        if (!inst.params || inst.params.length < 3) {
          throw new AMEVAForgeSecurityError(`${inst.op} instruction missing params`);
        }
        [M, N, K] = inst.params;
        wgslCode = inst.op === 'matmul_bias_relu' ? MATMUL_BIAS_RELU_WGSL : MATMUL_WGSL;
        isMatmul = true;
        // TS-H01 Fix: matmul X축도 65535 클램핑 — 초과분은 Z 차원으로 분산
        const rawDispatchX = Math.ceil(N / 8);
        if (rawDispatchX <= 65535) {
          dispatchX = rawDispatchX;
        } else {
          dispatchX = 65535;
          dispatchZ = Math.ceil(rawDispatchX / 65535);
        }
        const maxWorkgroupsM = Math.ceil(M / 8);
        if (maxWorkgroupsM > 65535) {
          throw new AMEVAForgeSecurityError(
            `Matmul M dimension (${M}) exceeds single-pass dispatch limit (524,280 rows). Partition tensor or reduce batch size.`
          );
        }
        dispatchY = maxWorkgroupsM;
      } else if (inst.op === 'batched_matmul') {
        if (!inst.params || inst.params.length < 4) {
          throw new AMEVAForgeSecurityError(`batched_matmul instruction missing params`);
        }
        const [B_param, N_param, P_param, M_param] = inst.params;
        B = B_param;
        wgslCode = BATCHED_MATMUL_WGSL;
        
        const rawDispatchX = Math.ceil(P_param / 8);
        if (rawDispatchX <= 65535) {
          dispatchX = rawDispatchX;
        } else {
          throw new AMEVAForgeSecurityError(`batched_matmul dispatchX exceeded limit: ${rawDispatchX}`);
        }
        
        const rawDispatchY = Math.ceil(N_param / 8);
        if (rawDispatchY <= 65535) {
          dispatchY = rawDispatchY;
        } else {
          throw new AMEVAForgeSecurityError(`batched_matmul dispatchY exceeded limit: ${rawDispatchY}`);
        }

        if (B <= 65535) {
          dispatchZ = B;
        } else {
          throw new AMEVAForgeSecurityError(`batched_matmul dispatchZ (Batch) exceeded limit: ${B}`);
        }
        
        const strideA = inst.params.length >= 7 ? inst.params[4] : N_param * M_param;
        const strideB = inst.params.length >= 7 ? inst.params[5] : M_param * P_param;
        const strideC = inst.params.length >= 7 ? inst.params[6] : N_param * P_param;
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([B_param, N_param, P_param, M_param, strideA, strideB, strideC, 0]));
      } else if (inst.op === 'transpose') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`transpose instruction missing params`);
        }
        const rM = inst.params[0];
        const rN = inst.params[1];
        const rB = inst.params.length >= 3 ? inst.params[2] : 1;
        wgslCode = TRANSPOSE_WGSL;
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([rM, rN, rB, 0]));
        dispatchX = Math.ceil(rM / 8);
        dispatchY = Math.ceil(rN / 8);
        dispatchZ = rB;
      } else if (inst.op === 'sum_axis' || inst.op === 'max_axis') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`${inst.op} instruction missing params`);
        }
        let outer_size = 1;
        let reduction_size = 1;
        let inner_stride = 1;
        if (inst.params.length >= 3) {
          [outer_size, reduction_size, inner_stride] = inst.params;
        } else {
          [reduction_size, inner_stride] = inst.params;
          outer_size = 1;
        }
        const output_numel = outer_size * inner_stride;
        wgslCode = inst.op === 'sum_axis' ? SUM_AXIS_WGSL : MAX_AXIS_WGSL;
        const totalWGs = Math.ceil(output_numel / 64);
        if (totalWGs <= 65535) {
          dispatchX = totalWGs;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWGs)));
          dispatchY = Math.min(65535, Math.ceil(totalWGs / dispatchX));
        }
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([outer_size, reduction_size, inner_stride, output_numel, dispatchX, 0, 0, 0]));
      } else if (inst.op === 'max_axis_backward') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`max_axis_backward instruction missing params`);
        }
        let outer_size = 1;
        let reduction_size = 1;
        let inner_stride = 1;
        if (inst.params.length >= 3) {
          [outer_size, reduction_size, inner_stride] = inst.params;
        } else {
          [reduction_size, inner_stride] = inst.params;
          outer_size = 1;
        }
        const input_numel = outer_size * reduction_size * inner_stride;
        wgslCode = MAX_AXIS_BACKWARD_WGSL;
        const totalWGs = Math.ceil(input_numel / 64);
        if (totalWGs <= 65535) {
          dispatchX = totalWGs;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWGs)));
          dispatchY = Math.min(65535, Math.ceil(totalWGs / dispatchX));
        }
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([outer_size, reduction_size, inner_stride, input_numel, dispatchX, 0, 0, 0]));
      } else if (inst.op === 'fill') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`fill instruction missing params`);
        }
        const numElements = inst.params[0];
        const fillValue = inst.params[1];
        wgslCode = FILL_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const f32arr = new Float32Array([0, fillValue, 0, 0]);
        const u32arr = new Uint32Array(f32arr.buffer);
        u32arr[0] = numElements;
        u32arr[2] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, u32arr);
      } else if (inst.op === 'axpy') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`axpy instruction missing params`);
        }
        const numElements = inst.params[0];
        const lr = inst.params[1];
        wgslCode = AXPY_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const f32arr = new Float32Array([0, lr, 0, 0]);
        const u32arr = new Uint32Array(f32arr.buffer);
        u32arr[0] = numElements;
        u32arr[2] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, u32arr);
      } else if (inst.op === 'pad') {
        const numElements = byteLength / 4;
        wgslCode = PAD_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const p = new Uint32Array(36);
        /**
         * WHAT: 패딩 옵션들을 유니폼 버퍼 배열에 복사하는 루프입니다.
         * WHY: 셰이더에서 사용될 스칼라 인자(정수 및 실수)를 메모리에 연속적으로 배치하기 위해 사용됩니다.
         * HOW: for 루프를 통해 inst.params 배열의 인자들을 p 배열로 옮기며, 실수형인 패딩 값은 Float32Array 뷰를 통해 씁니다.
         */
        for (let i = 0; i < inst.params!.length; i++) {
          if (i === 2) new Float32Array(p.buffer)[2] = inst.params![2];
          else p[i] = inst.params![i];
        }
        p[3] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'gather') {
        const numElements = byteLength / 4;
        wgslCode = GATHER_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const p = new Uint32Array(28);
        /**
         * WHAT: 파라미터를 복사하는 짧은 루프입니다.
         * WHY: gather 커널에 필요한 형태와 인덱싱 오프셋 정보들을 전송하기 위해 복사합니다.
         * HOW: 파라미터를 하나씩 Uint32Array에 대입합니다.
         */
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        p[3] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'scatter') {
        const numElements = inst.params![0];
        wgslCode = SCATTER_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const p = new Uint32Array(28);
        /**
         * WHAT: scatter 셰이더 인자를 복사하는 루프입니다.
         * WHY: 분산 배치할 인덱스 스텝 정보를 넘기기 위함입니다.
         * HOW: 파라미터를 하나씩 Uint32Array에 대입합니다.
         */
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        p[3] = dispatchX; // workgroups_x
        if (inst.params!.length < 28) {
          const shapeX = (inst.in && inst.in.length >= 3 && idToShape[inst.in[2]]) ? idToShape[inst.in[2]] : inst.shape;
          for (let i = 0; i < shapeX.length; i++) {
            p[20 + i] = shapeX[i];
          }
        }
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'dropout') {
        const numElements = byteLength / 4;
        const rawSeed = Number(inst.params![0]);
        const seed_u32 = (Number.isFinite(rawSeed) && rawSeed !== 0)
          ? (rawSeed >>> 0)
          : ((typeof crypto !== 'undefined' && crypto.getRandomValues)
              ? crypto.getRandomValues(new Uint32Array(1))[0]
              : (Math.floor(Math.random() * 0xFFFFFFFF) >>> 0));
        const p_val = inst.params![1];
        wgslCode = DROPOUT_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const buf = new ArrayBuffer(16);
        const u32view = new Uint32Array(buf);
        const f32view = new Float32Array(buf);
        u32view[0] = numElements;
        u32view[1] = seed_u32;
        f32view[2] = p_val;
        u32view[3] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, buf);
      } else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d') {
        const numElements = byteLength / 4;
        wgslCode = inst.op === 'maxpool2d' ? MAXPOOL2D_WGSL : AVGPOOL2D_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const p = new Uint32Array(16);
        /**
         * WHAT: 풀링 파라미터를 복사하는 루프입니다.
         * WHY: 윈도우 크기, 스트라이드, 패딩 등 컨볼루션 구조를 셰이더에 넘기기 위함입니다.
         * HOW: 요소별로 배열에 대입합니다.
         */
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        p[12] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'im2col' || inst.op === 'col2im') {
        const numElements = byteLength / 4;
        wgslCode = inst.op === 'im2col' ? IM2COL_WGSL : COL2IM_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const p = new Uint32Array(12);
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        p[10] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'permute') {
        const numElements = byteLength / 4;
        wgslCode = PERMUTE_WGSL;
        const dims = inst.params!;
        const rank = dims.length;
        
        const inHandle = idToHandle[inst.in![0]];
        const inShape = idToShape[inst.in![0]] ?? (_globalRegistry.has(inHandle) ? _globalRegistry.get(inHandle)!.shape : inst.shape);
        
        const inStrides = new Array(rank).fill(0);
        let s = 1;
        /**
         * WHAT: 입력 텐서의 각 차원별 메모리 보폭(stride)을 계산하는 역순 루프입니다.
         * WHY: 다차원 인덱스를 1차원 플랫 메모리 오프셋으로 변환할 때 곱해줄 가중치를 구하기 위해 필요합니다.
         * HOW: 가장 마지막 차원(우측)부터 시작하여 1부터 차례로 곱해나가며 배열을 채웁니다.
         */
        for (let i = rank - 1; i >= 0; i--) {
            inStrides[i] = s;
            s *= inShape[i];
        }
        
        const outStrides = new Array(rank).fill(0);
        let s2 = 1;
        /**
         * WHAT: 출력 텐서의 각 차원별 스트라이드를 계산하는 역순 루프입니다.
         * WHY: 출력을 기록할 1차원 주소를 생성할 때 사용될 가중치를 미리 연산해두기 위함입니다.
         * HOW: 마찬가지로 맨 우측 차원부터 누적하여 곱합니다.
         */
        for (let i = rank - 1; i >= 0; i--) {
            outStrides[i] = s2;
            s2 *= inst.shape[i];
        }
        
        const dispatch = computeDispatch2D(numElements, 64);
        dispatchX = dispatch.dispatchX;
        dispatchY = dispatch.dispatchY;

        const p = new Uint32Array(28);
        p[0] = rank;
        p[1] = numElements;
        p[2] = dispatch.workgroupsX;
        p[3] = 0;
        
        /**
         * WHAT: 계산된 각 차원들의 스트라이드와 형태 정보를 WebGPU vec4 정렬 규칙에 맞게 유니폼 버퍼 패딩 구조에 삽입하는 루프입니다.
         * WHY: GPU 셰이더 내에서 배열이나 벡터 형태로 데이터를 오차 없이 접근하기 위해 메모리 오프셋을 맞추어 기록합니다.
         * HOW: i를 0부터 rank 전까지 증가시키며 4개 단위 벡터 위치를 계산하여 씁니다.
         */
        for (let i = 0; i < rank; i++) {
           const vecOffset = i < 4 ? 4 + i : 8 + (i - 4);
           p[vecOffset] = inStrides[dims[i]];
           
           const outShapeOffset = i < 4 ? 12 + i : 16 + (i - 4);
           p[outShapeOffset] = inst.shape[i];
           
           const outStrideOffset = i < 4 ? 20 + i : 24 + (i - 4);
           p[outStrideOffset] = outStrides[i];
        }
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'sum' || inst.op === 'max') {
        // Handled entirely dynamically below, but we need to bypass normal flow
        wgslCode = inst.op === 'sum' ? SUM_WGSL : MAX_WGSL;
      } else {
        const numElements = byteLength / 4;
        wgslCode = inst.op === 'relu'          ? RELU_WGSL :
                   inst.op === 'add'           ? ADD_WGSL :
                   inst.op === 'mul'           ? MUL_WGSL :
                   inst.op === 'sub'           ? SUB_WGSL :
                   inst.op === 'neg'           ? NEG_WGSL :
                   inst.op === 'div'           ? DIV_WGSL :
                   inst.op === 'relu_backward' ? RELU_BACKWARD_WGSL :
                   inst.op === 'exp'           ? EXP_WGSL :
                   inst.op === 'log'           ? LOG_WGSL :
                   inst.op === 'sigmoid'       ? SIGMOID_WGSL :
                   inst.op === 'tanh'          ? TANH_WGSL :
                   inst.op === 'sigmoid_backward' ? SIGMOID_BACKWARD_WGSL :
                   inst.op === 'tanh_backward' ? TANH_BACKWARD_WGSL : 
                   inst.op === 'cat'           ? CAT_WGSL :
                   inst.op === 'where'         ? WHERE_WGSL : 
                   inst.op === 'dropout'       ? DROPOUT_WGSL : '';

        if (!wgslCode) {
          throw new AMEVAForgeSecurityError(`Unknown op "${inst.op}"`);
        }
        
        const totalWorkgroups = Math.ceil(numElements / 64);
        // TS-C01 Fix: 65535 초과 시 2D 그리드로 분산
        if (totalWorkgroups <= 65535) {
            dispatchX = totalWorkgroups;
            dispatchY = 1;
        } else {
            // 2D 분산: sqrt로 균등 분할
            dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
            dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        if (['add', 'sub', 'mul', 'div'].includes(inst.op)) {
          let shapeA = [1];
          let shapeB = [1];
          if (inst.in && inst.in.length >= 2) {
            const in0Handle = idToHandle[inst.in[0]];
            const in1Handle = idToHandle[inst.in[1]];
            shapeA = idToShape[inst.in[0]] ?? (_globalRegistry.has(in0Handle) ? _globalRegistry.get(in0Handle).shape : [1]);
            shapeB = idToShape[inst.in[1]] ?? (_globalRegistry.has(in1Handle) ? _globalRegistry.get(in1Handle).shape : [1]);
          }
          const { dOut, effSA, effSB } = computeBroadcastParams(inst.shape, shapeA, shapeB);
          const p = new Uint32Array(28);
          p[0] = numElements;
          p[1] = dispatchX;
          p[2] = inst.shape.length;
          p[3] = 0;
          for (let k = 0; k < 8; k++) p[4 + k] = dOut[k];
          for (let k = 0; k < 8; k++) p[12 + k] = effSA[k];
          for (let k = 0; k < 8; k++) p[20 + k] = effSB[k];
          device.queue.writeBuffer(paramsBuffer, 0, p);
        } else {
          let numA = 0;
          let numB = 0;
          if (inst.in && inst.in.length >= 2) {
            numA = (idToByteLength[inst.in[0]] ?? byteLength) / 4;
            numB = (idToByteLength[inst.in[1]] ?? byteLength) / 4;
          }
          device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, dispatchX, numA, numB, 0, 0, 0, 0]));
        }

        if (inst.op === 'cat') {
          if (!inst.params || inst.params.length < 3) {
            throw new AMEVAForgeSecurityError(`cat instruction missing params`);
          }
          const [a_dim, b_dim, stride] = inst.params;
          // Overwrite the params for cat
          device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, dispatchX, a_dim, b_dim, stride, 0, 0, 0]));
        }
      }

      const { pipeline } = _globalPipelineCache.getPipeline(inst.op, wgslCode);

      if (inst.op === 'sum' || inst.op === 'max') {
          if (!inst.in || inst.in.length === 0) {
              throw new AMEVAForgeSecurityError(`Instruction op="${inst.op}" is missing 'in' field.`);
          }
          const REDUCTION_WG_SIZE = 256;
          const reductionInputHandle = idToHandle[inst.in[0]];
          if (!reductionInputHandle) throw new AMEVAForgeSecurityError(`Unresolved reduction input id ${inst.in[0]}`);
          
          let currentByteLength = idToByteLength[inst.in[0]];
          if (currentByteLength === undefined) {
            const rec = _globalRegistry.has(reductionInputHandle)
              ? _globalRegistry.get(reductionInputHandle)
              : transaction.get(reductionInputHandle);
            currentByteLength = rec ? rec.byteLength : 4;
          }
          let currentSize = currentByteLength / 4;
          let currentInputBuf = idToBuffer[inst.in[0]];
          const intermediateAllocations: Array<{ buffer: GPUBuffer, token: AllocationToken }> = [];
          
          while (currentSize > 1) {
              const numWGs = Math.ceil(currentSize / REDUCTION_WG_SIZE);
              let rDispatchX = 1;
              let rDispatchY = 1;
              if (numWGs <= 65535) {
                rDispatchX = numWGs;
                rDispatchY = 1;
              } else {
                rDispatchX = Math.min(65535, Math.ceil(Math.sqrt(numWGs)));
                rDispatchY = Math.min(65535, Math.ceil(numWGs / rDispatchX));
              }

              const { buffer: passBuf, token: passBufToken } = allocateBuffer(
                  Math.max(4, numWGs * 4),
                  BUFFER_USAGE_STORAGE_SRC,
                  'temporary',
                  `Graph_${instructions[0]?.id}_reduction`
              );
              intermediateAllocations.push({ buffer: passBuf, token: passBufToken });
              
              const { buffer: passParamsBuf, token: passParamsToken } = allocateBuffer(
                  16,
                  BUFFER_USAGE_UNIFORM_COPY,
                  'uniform',
                  `Graph_${instructions[0]?.id}_reduction_params`
              );
              intermediateAllocations.push({ buffer: passParamsBuf, token: passParamsToken });
              device.queue.writeBuffer(passParamsBuf, 0, new Uint32Array([currentSize, rDispatchX, 0, 0]));
              
              const wgsl = inst.op === 'sum' ? SUM_WGSL : MAX_WGSL;
              const { pipeline: reducePipeline } = _globalPipelineCache.getPipeline(inst.op + '_pass', wgsl);
              
              const passEncoder = commandEncoder.beginComputePass();
              passEncoder.setPipeline(reducePipeline);
              passEncoder.setBindGroup(0, device.createBindGroup({
                  layout: reducePipeline.getBindGroupLayout(0),
                  entries: [
                      { binding: 0, resource: { buffer: passParamsBuf } },
                      { binding: 1, resource: { buffer: currentInputBuf } },
                      { binding: 2, resource: { buffer: passBuf } },
                  ],
              }));
              passEncoder.dispatchWorkgroups(rDispatchX, rDispatchY, 1);
              passEncoder.end();
              encoderHasCommands = true;
              currentInputBuf = passBuf;
              currentSize = numWGs;
          }
          
          commandEncoder.copyBufferToBuffer(currentInputBuf, 0, outBuffer, 0, 4);
          encoderHasCommands = true;
          
          for (const alloc of intermediateAllocations) {
              paramsAllocations.push(alloc);
          }
          continue;
      }

      if (inst.op !== 'fill' && (!inst.in || inst.in.length === 0)) {
        throw new AMEVAForgeSecurityError(`Instruction op="${inst.op}" is missing 'in' field.`);
      }

      let bindGroupEntries: GPUBindGroupEntry[] = [];
      if (inst.op === 'fill') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'axpy') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'gather' || inst.op === 'scatter') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } },
          { binding: 3, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'matmul_bias_relu') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } },
          { binding: 3, resource: { buffer: idToBuffer[inst.in![2]] } },
          { binding: 4, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'where') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } },
          { binding: 3, resource: { buffer: idToBuffer[inst.in![2]] } },
          { binding: 4, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'dropout') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: outBuffer } },
        ];
      } else {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
        ];

        if (inst.in!.length > 1) {
          bindGroupEntries.push({ binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } });
          bindGroupEntries.push({ binding: 3, resource: { buffer: outBuffer } });
        } else {
          bindGroupEntries.push({ binding: 2, resource: { buffer: outBuffer } });
        }
      }

      if (isMatmul) {
        const MACS_PER_CHUNK = 2_000_000_000;
        const macsPerRow = N * K;
        let chunkY = Math.max(1, Math.floor(MACS_PER_CHUNK / macsPerRow));
        chunkY = Math.min(chunkY, 65535 * 8);
        chunkY = Math.min(M, chunkY);

        const has_bias = inst.op === 'matmul_bias_relu' ? (inst.params?.[3] ?? 1) : 0;
        const has_relu = inst.op === 'matmul_bias_relu' ? (inst.params?.[4] ?? 1) : 0;

        for (let offsetY = 0; offsetY < M; offsetY += chunkY) {
          const currentChunkY = Math.min(chunkY, M - offsetY);
          
          const chunkParamEntry = _globalUniformPool.acquire(32);
          const chunkParamsBuffer = chunkParamEntry.buffer;
          paramsAllocations.push({ buffer: chunkParamsBuffer, token: chunkParamEntry.token, isUniformPool: true, uniformEntry: chunkParamEntry });
          device.queue.writeBuffer(chunkParamsBuffer, 0, new Uint32Array([M, N, K, offsetY, has_bias, has_relu, 0, 0]));
          
          const chunkBindGroupEntries = bindGroupEntries.map(e => {
            if (e.binding === 0) return { binding: 0, resource: { buffer: chunkParamsBuffer } };
            return e;
          });
          const chunkBindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: chunkBindGroupEntries
          });

          const passEncoder = commandEncoder.beginComputePass();
          passEncoder.setPipeline(pipeline);
          passEncoder.setBindGroup(0, chunkBindGroup);
          passEncoder.dispatchWorkgroups(dispatchX, Math.ceil(currentChunkY / 8), dispatchZ);
          passEncoder.end();

          opsInCurrentBatch++;
          workloadElements += (dispatchX * currentChunkY * 8 * 8); 
          
          if (offsetY + currentChunkY < M || workloadElements >= _runtimeConfig.workloadBudgetElements || opsInCurrentBatch >= _runtimeConfig.maxOpsPerSubmit) {
            device.queue.submit([commandEncoder.finish()]);
            commandEncoder = device.createCommandEncoder();
            opsInCurrentBatch = 0;
            workloadElements = 0;
          }
        }
      } else {
        if (inst.op === 'scatter') {
          // If in[2] exists (base tensor x), copy x to outBuffer so unscattered elements retain x values
          if (inst.in && inst.in.length >= 3 && idToBuffer[inst.in[2]]) {
            commandEncoder.copyBufferToBuffer(idToBuffer[inst.in[2]], 0, outBuffer, 0, byteLength);
            encoderHasCommands = true;
          }
        }

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: bindGroupEntries
        });

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        passEncoder.end();

        opsInCurrentBatch++;
        workloadElements += byteLength / 4;
        if (workloadElements >= _runtimeConfig.workloadBudgetElements || opsInCurrentBatch >= _runtimeConfig.maxOpsPerSubmit) {
          device.queue.submit([commandEncoder.finish()]);
          commandEncoder = device.createCommandEncoder();
          opsInCurrentBatch = 0;
          workloadElements = 0;
        }
      }
    }

  } catch (err: any) {
    // ── 5. Rollback on Sync Error ──
    _safeLog(`[AMEVA Forge] Transaction Sync Failed. Rolling back... ${err}`);
    
    transaction.rollback();
    
    for (const alloc of paramsAllocations) {
      if (alloc.isUniformPool && alloc.uniformEntry) {
        _globalUniformPool.releaseSync(alloc.uniformEntry);
      } else {
        try { freeBuffer(alloc.buffer, alloc.token); } catch (e) {}
      }
    }
    try {
      await device.popErrorScope();
      await device.popErrorScope();
      await device.popErrorScope();
    } catch {}
    throw err;
  }

  if (encoderHasCommands || opsInCurrentBatch > 0) {
    device.queue.submit([commandEncoder.finish()]);
    encoderHasCommands = false;
  }

  // ── 5. Commit / Rollback (Async) — await error scopes before returning ──
  const internalError = await device.popErrorScope();
  const oomError = await device.popErrorScope();
  const validationError = await device.popErrorScope();

  // Check for GPU errors BEFORE returning handles
  const gpuError = internalError || oomError || validationError;
  if (gpuError) {
    _safeLog(`[AMEVA Forge] GPU error detected. Rolling back transaction... ${gpuError}`);
    transaction.rollback();
    for (const alloc of paramsAllocations) {
      if (alloc.isUniformPool && alloc.uniformEntry) {
        _globalUniformPool.releaseAfterSubmit(alloc.uniformEntry);
      } else {
        try { freeBuffer(alloc.buffer, alloc.token); } catch (e) {}
      }
    }
    void _globalUniformPool.retireSubmitted(device);
    // Determine error type
    if (internalError) {
      throw new AMEVAForgeInternalGPUError(`Internal GPU Error: ${internalError.message}`);
    } else if (oomError) {
      throw new AMEVAForgeOutOfMemoryError(`GPU Out of Memory: ${oomError.message}`);
    } else {
      throw new AMEVAForgeValidationError(`GPU Validation Error: ${validationError!.message}`);
    }
  }

  // ── 6. Commit transaction to global registry only on verified success ──
  transaction.commit(_globalRegistry);

  // ── 7. Cleanup temporary/uniform allocations after GPU completion ──
  if (paramsAllocations.length > 0) {
    const nonPoolAllocs: Array<{ buffer: GPUBuffer, token: AllocationToken }> = [];
    for (const alloc of paramsAllocations) {
      if (alloc.isUniformPool && alloc.uniformEntry) {
        _globalUniformPool.releaseAfterSubmit(alloc.uniformEntry);
      } else {
        nonPoolAllocs.push(alloc);
      }
    }
    if (_globalUniformPool.inFlightBytes() > 512 * 1024) {
      await _globalUniformPool.retireSubmitted(device);
    } else {
      void _globalUniformPool.retireSubmitted(device);
    }
    if (nonPoolAllocs.length > 0) {
      device.queue.onSubmittedWorkDone().then(() => {
        for (const alloc of nonPoolAllocs) {
          try { freeBuffer(alloc.buffer, alloc.token); } catch (e) {}
        }
      }).catch(() => {
        for (const alloc of nonPoolAllocs) {
          try { freeBuffer(alloc.buffer, alloc.token); } catch (e) {}
        }
      });
    }
  }

  return idToHandle;
}

```

---

## `packages/forge/src/tensor/kernels/add.wgsl.ts`

```typescript
/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *   - 2026-08-18 14:10:00 +0900: Feat: Full 8D Multi-Dimensional Stride Broadcasting Decoder
 */
export const ADD_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  rank: u32,
  pad0: u32,
  dim0: u32, dim1: u32, dim2: u32, dim3: u32,
  dim4: u32, dim5: u32, dim6: u32, dim7: u32,
  stride_a0: u32, stride_a1: u32, stride_a2: u32, stride_a3: u32,
  stride_a4: u32, stride_a5: u32, stride_a6: u32, stride_a7: u32,
  stride_b0: u32, stride_b1: u32, stride_b2: u32, stride_b3: u32,
  stride_b4: u32, stride_b5: u32, stride_b6: u32, stride_b7: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    var temp = idx;
    let c7 = temp % params.dim7; temp = temp / params.dim7;
    let c6 = temp % params.dim6; temp = temp / params.dim6;
    let c5 = temp % params.dim5; temp = temp / params.dim5;
    let c4 = temp % params.dim4; temp = temp / params.dim4;
    let c3 = temp % params.dim3; temp = temp / params.dim3;
    let c2 = temp % params.dim2; temp = temp / params.dim2;
    let c1 = temp % params.dim1; temp = temp / params.dim1;
    let c0 = temp;

    let idx_a = c0 * params.stride_a0 + c1 * params.stride_a1 + c2 * params.stride_a2 + c3 * params.stride_a3 +
                c4 * params.stride_a4 + c5 * params.stride_a5 + c6 * params.stride_a6 + c7 * params.stride_a7;
    let idx_b = c0 * params.stride_b0 + c1 * params.stride_b1 + c2 * params.stride_b2 + c3 * params.stride_b3 +
                c4 * params.stride_b4 + c5 * params.stride_b5 + c6 * params.stride_b6 + c7 * params.stride_b7;

    out[idx] = a[idx_a] + b[idx_b];
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/avgpool2d.wgsl.ts`

```typescript
/**
 * 생성일 (Created): 2026-08-12 12:59:35 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const AVGPOOL2D_WGSL = `
/**
 * @struct Params
 * @brief 2D 평균 풀링(Average Pooling 2D) 연산에 필요한 하이퍼파라미터 및 텐서 차원 정보를 저장합니다. (What)
 * 셰이더 내에서 입력 텐서의 특정 영역을 순회하고 평균을 계산하기 위한 기준 값들로 사용됩니다. (Why)
 */
struct Params {
    // 배치(batch) 크기입니다. 여러 이미지를 동시에 처리하기 위한 차원입니다.
    batch: u32,
    // 채널(channel) 수입니다. 예를 들어 RGB 이미지의 경우 3이 될 수 있습니다.
    channels: u32,
    // 입력 이미지의 높이(height) 차원 크기입니다.
    in_h: u32,
    // 입력 이미지의 너비(width) 차원 크기입니다.
    in_w: u32,
    // 출력 이미지의 높이 차원 크기입니다. 연산 후의 공간적 크기를 나타냅니다.
    out_h: u32,
    // 출력 이미지의 너비 차원 크기입니다.
    out_w: u32,
    // 풀링 커널(kernel)의 높이 크기입니다.
    kH: u32,
    // 풀링 커널(kernel)의 너비 크기입니다.
    kW: u32,
    // 높이 방향의 스트라이드(stride, 이동 보폭)입니다.
    sH: u32,
    // 너비 방향의 스트라이드(stride, 이동 보폭)입니다.
    sW: u32,
    // 높이 방향의 패딩(padding) 크기입니다.
    pH: u32,
    // 너비 방향의 패딩(padding) 크기입니다.
    pW: u32,
    // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수입니다.
    workgroups_x: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

// params: 연산 정보를 제공하는 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// input: 풀링 연산을 수행할 원본 입력 데이터 배열(읽기 전용)입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;
// output: 풀링 연산 결과가 기록될 출력 데이터 배열입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

/**
 * @function main
 * @brief 컴퓨트 셰이더의 진입점으로, 각 스레드가 하나의 출력 픽셀에 대한 2D 평균 풀링 연산을 수행합니다. (What)
 * GPU의 수많은 스레드를 활용하여 이미지 전체 영역 및 배치 데이터를 병렬로 압축 처리하기 위해 존재합니다. (Why)
 * @param global_id 워크그룹 및 스레드의 전역적인 3차원 위치(인덱스)입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 2D 디스패치 그리드로부터 복원한 현재 스레드의 선형 인덱스를 가져옵니다.
    let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
    
    // 계산해야 할 전체 출력 요소의 총합을 구합니다 (배치 * 채널 * 출력높이 * 출력너비). (How)
    let total = params.batch * params.channels * params.out_h * params.out_w;
    
    // 스레드 인덱스가 유효 범위를 벗어나면 즉시 함수를 종료(return)하여 잘못된 메모리 접근을 막습니다. (Why)
    if (idx >= total) {
        return;
    }
    
    // 1차원 인덱스 idx를 4차원 좌표 (b, c, oh, ow)로 변환하는 과정입니다. (How)
    // 현재 픽셀이 속한 출력 이미지의 너비 위치(ow)를 구합니다.
    let ow = idx % params.out_w;
    // 현재 픽셀이 속한 출력 이미지의 높이 위치(oh)를 구합니다.
    let oh = (idx / params.out_w) % params.out_h;
    // 현재 픽셀이 속한 채널 위치(c)를 구합니다.
    let c = (idx / (params.out_w * params.out_h)) % params.channels;
    // 현재 픽셀이 속한 배치 위치(b)를 구합니다.
    let b = idx / (params.out_w * params.out_h * params.channels);
    
    // 입력 이미지에서 현재 커널이 적용될 시작 Y좌표(높이)를 계산합니다. 패딩을 고려하여 음수가 될 수도 있습니다. (What)
    let h_start = i32(oh * params.sH) - i32(params.pH);
    // 입력 이미지에서 현재 커널이 적용될 시작 X좌표(너비)를 계산합니다. 패딩을 고려합니다.
    let w_start = i32(ow * params.sW) - i32(params.pW);
    
    // 풀링 영역 내의 픽셀 값들을 누적하기 위한 합계 변수입니다. (What)
    var sum = 0.0;
    // 풀링 영역 내에서 실제로 유효한 픽셀의 개수를 셉니다. (경계 밖은 제외하기 위함) (Why)
    var count = 0.0;
    
    // 커널의 높이만큼 반복하여 수직 방향 픽셀들을 순회합니다. (How)
    for (var kh = 0u; kh < params.kH; kh++) {
        // 커널의 너비만큼 반복하여 수평 방향 픽셀들을 순회합니다. (How)
        for (var kw = 0u; kw < params.kW; kw++) {
            // 현재 순회 중인 픽셀의 실제 입력 텐서상 Y좌표입니다.
            let h = h_start + i32(kh);
            // 현재 순회 중인 픽셀의 실제 입력 텐서상 X좌표입니다.
            let w = w_start + i32(kw);
            
            // 유효성 검사: 계산된 (h, w)가 이미지 경계를 벗어나지 않는지(0 이상, 입력 크기 미만) 확인합니다. (What)
            // 패딩 영역이나 이미지 범위를 넘어간 곳의 값은 무시하여 올바른 평균을 구하기 위함입니다. (Why)
            if (h >= 0 && h < i32(params.in_h) && w >= 0 && w < i32(params.in_w)) {
                // 4차원 좌표 (b, c, h, w)를 다시 1차원 인덱스(in_idx)로 변환합니다. (How)
                let in_idx = ((b * params.channels + c) * params.in_h + u32(h)) * params.in_w + u32(w);
                
                // 해당 입력 픽셀의 값을 합산 변수에 누적시킵니다.
                sum += input[in_idx];
                // 유효한 픽셀을 한 개 처리했으므로 카운트를 증가시킵니다.
                count += 1.0;
            }
        }
    }
    
    // 유효한 픽셀 카운트가 1개 이상일 경우 정상적으로 평균을 계산합니다. (What)
    // 0으로 나누기(Division by zero) 오류를 방지하기 위함입니다. (Why)
    if (count > 0.0) {
        // 총 누적 합(sum)을 유효 픽셀 개수(count)로 나누어 평균을 구한 후, 출력 배열의 1차원 인덱스에 저장합니다. (How)
        output[idx] = sum / count;
    } else {
        // 유효한 픽셀이 전혀 없었다면(예: 모두 패딩 영역인 경우) 결과값을 0으로 처리합니다. (How)
        output[idx] = 0.0;
    }
}
`;

```

---

## `packages/forge/src/tensor/kernels/axpy.wgsl.ts`

```typescript
/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *   - 2026-08-18 14:10:00 +0900: Pure Standard IEEE-754 SGD Update without silent NaN/Inf zeroing
 */
export const AXPY_WGSL = `
/**
 * @struct Params
 * @brief AXPY (param = param - lr * grad) 연산 파라미터 구조체
 */
struct Params {
  numElements: u32,
  lr: f32,
  workgroups_x: u32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> param: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  if (idx >= params.numElements) {
    return;
  }
  
  let g = grad[idx];
  // Standard SGD in-place update (IEEE 754 float32)
  param[idx] = param[idx] - params.lr * g;
}
`;

```

---

## `packages/forge/src/tensor/kernels/batched_matmul.wgsl.ts`

```typescript
/**
 * 생성일 (Created): 2026-08-12 12:59:35 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const BATCHED_MATMUL_WGSL = `
/**
 * @struct Params
 * @brief 배치 행렬 곱셈(Batched Matrix Multiplication)을 제어하기 위한 행렬의 차원 크기와 스트라이드(stride) 정보를 저장합니다. (What)
 * 입력 행렬 텐서 A와 B의 형태(M, N, K)와 연속적인 배치 접근을 위한 메모리 오프셋을 계산할 때 사용하기 위해 정의되었습니다. (Why)
 */
struct Params {
  // 배치(Batch)의 개수입니다. 한 번에 여러 쌍의 행렬 곱셈을 병렬 처리하기 위한 차원입니다.
  B: u32,
  // 결과 행렬(C)과 왼쪽 행렬(A)의 행(Row) 개수입니다.
  M: u32,
  // 결과 행렬(C)과 오른쪽 행렬(B)의 열(Column) 개수입니다.
  N: u32,
  // 왼쪽 행렬(A)의 열 개수이자 오른쪽 행렬(B)의 행 개수로, 내적(Dot product)이 이루어지는 공통 차원의 길이입니다.
  K: u32,
  // 왼쪽 행렬(A)에서 다음 배치로 넘어가기 위해 필요한 원소의 개수(보폭)입니다.
  strideA: u32,
  // 오른쪽 행렬(B)에서 다음 배치로 넘어가기 위해 필요한 원소의 개수(보폭)입니다.
  strideB: u32,
  // 결과 행렬(C)에서 다음 배치로 넘어가기 위해 필요한 원소의 개수(보폭)입니다.
  strideC: u32,
};

// params: 배치 크기 및 행렬 차원 정보를 GPU 스레드들에게 제공하는 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// a: 첫 번째(왼쪽) 입력 행렬 데이터들을 담고 있는 1차원 배열(읽기 전용)입니다.
@group(0) @binding(1) var<storage, read> a: array<f32>;
// b: 두 번째(오른쪽) 입력 행렬 데이터들을 담고 있는 1차원 배열(읽기 전용)입니다.
@group(0) @binding(2) var<storage, read> b: array<f32>;
// c: 행렬 곱셈의 결과가 저장될 출력 배열(읽기/쓰기 가능)입니다.
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

/**
 * @function main
 * @brief 주어진 배치(Batch)에 대해 행렬 A와 B의 내적을 수행하여 행렬 C의 각 요소를 계산합니다. (What)
 * 어텐션 메커니즘 등 신경망 구조에서 다중 배치의 텐서를 한 번에 곱하기 위해 (Why) 3차원 그리드로 병렬 실행됩니다.
 * 
 * @param global_id 워크그룹과 스레드의 3차원 인덱스입니다. (x: 열(Column), y: 행(Row), z: 배치(Batch)를 나타냅니다.) (How)
 */
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 스레드의 x 인덱스로, 연산할 결과 행렬 C의 열(Column) 위치를 할당합니다.
  let col = global_id.x;
  // 스레드의 y 인덱스로, 연산할 결과 행렬 C의 행(Row) 위치를 할당합니다.
  let row = global_id.y;
  // 스레드의 z 인덱스로, 현재 처리할 배치(Batch) 번호를 할당합니다.
  let batch = global_id.z;

  // 할당된 인덱스들이 지정된 행렬 크기나 배치 수를 초과하는지 검사합니다. (What)
  // 워크그룹 크기(8x8)로 인해 남는 스레드가 유효하지 않은 메모리에 접근하는 것을 방지하기 위함입니다. (Why)
  if (row >= params.M || col >= params.N || batch >= params.B) {
    return;
  }

  // 1차원 배열 A에서 현재 배치의 현재 행이 시작되는 오프셋을 계산합니다. (How)
  let a_offset = batch * params.strideA + row * params.K;
  // 1차원 배열 B에서 현재 배치의 현재 열이 시작되는 오프셋을 계산합니다.
  let b_offset = batch * params.strideB + col;
  // 1차원 결과 배열 C에서 현재 배치의 위치(row, col)에 해당하는 저장 인덱스를 계산합니다.
  let c_offset = batch * params.strideC + row * params.N + col;

  // 내적(Dot product)을 누적하기 위한 실수형 변수를 선언하고 0으로 초기화합니다. (What)
  var sum: f32 = 0.0;
  
  // 공통 차원인 K번만큼 반복하여 행렬 A의 특정 행과 행렬 B의 특정 열의 요소들을 곱하고 더합니다. (How)
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    // 행렬 A에서는 열(k) 방향으로 이동하고, 행렬 B에서는 행(k) 방향으로 이동(B의 행 길이인 N만큼 점프)하면서 값을 곱하여 sum에 누적시킵니다. (How)
    sum = sum + a[a_offset + k] * b[b_offset + k * params.N];
  }

  // 계산된 내적 최종 결과(sum)를 출력 배열 C의 오프셋 위치에 저장합니다. (What)
  c[c_offset] = sum;
}
`;

```

---

## `packages/forge/src/tensor/kernels/cat.wgsl.ts`

```typescript
/**
 * 생성일 (Created): 2026-08-12 12:23:09 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
 */
export const CAT_WGSL = `
/**
 * @struct Params
 * @brief 두 텐서를 특정 차원(dimension)을 기준으로 결합(concatenate)할 때 사용하는 파라미터 구조체입니다. (What)
 * 텐서의 형태와 크기를 기반으로 각 텐서에서 어떤 위치의 값을 가져올지 인덱스를 계산하기 위해 존재합니다. (Why)
 */
struct Params {
  // 결합이 완료된 결과 텐서의 전체 요소(element) 개수입니다.
  size: u32,
  // X축 워크그룹 수. 2차원 그리드 인덱싱을 1차원 인덱스로 풀기 위한 변수입니다.
  workgroups_x: u32,
  // 결합하려는 축(axis)에서 첫 번째 텐서(A)가 차지하는 차원의 크기입니다.
  a_dim: u32,
  // 결합하려는 축(axis)에서 두 번째 텐서(B)가 차지하는 차원의 크기입니다.
  b_dim: u32,
  // 결합 축(axis)보다 하위에 있는 차원들의 요소 개수 곱입니다(Stride). 
  // 상위 차원이나 배치(batch)를 뛰어넘기 위한 보폭 역할을 합니다. (How)
  stride: u32,
  // 메모리 정렬(16바이트)을 위한 패딩 변수 1입니다.
  pad1: u32,
  // 메모리 정렬을 위한 패딩 변수 2입니다.
  pad2: u32,
  // 메모리 정렬을 위한 패딩 변수 3입니다.
  pad3: u32,
};

// params: 결합 연산에 필요한 차원 및 크기 정보를 제공하는 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// a: 결합될 첫 번째 입력 텐서 데이터 배열입니다 (읽기 전용).
@group(0) @binding(1) var<storage, read> a: array<f32>;
// b: 결합될 두 번째 입력 텐서 데이터 배열입니다 (읽기 전용).
@group(0) @binding(2) var<storage, read> b: array<f32>;
// out: A와 B가 이어진(Concatenated) 결과가 저장되는 배열입니다.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

/**
 * @function main
 * @brief 결과 텐서의 각 요소가 입력 텐서 A 혹은 B 중 어디서 와야 하는지를 계산하고 복사합니다. (What)
 * 병렬 인덱싱을 통하여 다차원 텐서의 결합 연산을 빠르게 수행하기 위해 만들어졌습니다. (Why)
 * @param global_id 워크그룹 및 스레드의 3차원 전역 인덱스 변수입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 파라미터에서 전체 데이터 개수를 가져옵니다.
  let num_elements = params.size;
  // 파라미터에서 X 방향 워크그룹 개수를 가져옵니다.
  let workgroups_x = params.workgroups_x;
  // 3차원 워크그룹 및 스레드 ID를 1차원 선형 인덱스로 변환합니다. (How)
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 계산된 인덱스가 전체 요소 개수를 넘어갈 경우 안전하게 함수를 종료합니다. (What)
  // 배열 범위를 벗어난 메모리에 대한 불법적인 쓰기를 방지하기 위해서입니다. (Why)
  if (idx >= num_elements) {
    return;
  }
  
  // 파라미터 구조체에서 내부 차원의 크기(stride)를 로드합니다.
  let stride = params.stride;
  // 파라미터 구조체에서 A 텐서의 결합 축 크기를 로드합니다.
  let a_dim = params.a_dim;
  // 파라미터 구조체에서 B 텐서의 결합 축 크기를 로드합니다.
  let b_dim = params.b_dim;
  
  // 결합된 이후 결과 텐서의 해당 축 길이를 계산합니다. (What)
  let out_dim_size = a_dim + b_dim;
  // 한 블록(결합 축 1개 단위 + 하위 차원 전체)이 차지하는 총 요소 개수(청크 크기)를 계산합니다. (How)
  let chunk_size = out_dim_size * stride;
  
  // 현재 1차원 인덱스가 어떤 배치(상위 차원들)에 속하는지 계산합니다. (How)
  let batch_idx = idx / chunk_size;
  // 현재 청크(chunk) 내에서 몇 번째 인덱스인지를 구합니다. (나머지 연산)
  let rem = idx % chunk_size;
  // 현재 청크 내에서 결합 축을 기준으로 몇 번째 위치에 있는지를 구합니다. (How)
  let dim_idx = rem / stride;
  // 결합 축보다 하위에 있는 차원에서 몇 번째 위치(stride_idx)인지를 구합니다.
  let stride_idx = rem % stride;
  
  // 현재 계산된 결합 축 상의 위치(dim_idx)가 텐서 A의 크기보다 작은지 검사합니다. (What)
  // 이 조건이 참이면 현재 요소는 텐서 A에서 가져와야 함을 의미합니다. (Why)
  if (dim_idx < a_dim) {
    // 텐서 A 배열 내부에서의 정확한 1차원 원본 인덱스를 복원 계산합니다. (How)
    // 배치 크기 * (A 차원 크기 * 스트라이드) + (A 안에서의 축 위치 * 스트라이드) + 하위 차원 오프셋
    let a_index = batch_idx * (a_dim * stride) + dim_idx * stride + stride_idx;
    // 계산된 인덱스를 사용해 텐서 A의 값을 결과 텐서에 복사합니다.
    out[idx] = a[a_index];
  } else {
    // dim_idx가 a_dim 이상이면 현재 요소는 텐서 B에서 가져와야 합니다.
    // 텐서 B의 내부 차원 인덱스로 변환하기 위해 A가 차지했던 크기를 뺍니다. (How)
    let b_dim_idx = dim_idx - a_dim;
    // 텐서 B 배열 내부에서의 원본 위치를 계산합니다. (How)
    let b_index = batch_idx * (b_dim * stride) + b_dim_idx * stride + stride_idx;
    // 계산된 인덱스를 사용해 텐서 B의 값을 결과 텐서에 복사합니다.
    out[idx] = b[b_index];
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/col2im.wgsl.ts`

```typescript
/**
 * 생성일 (Created): 2026-08-12 12:59:35 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const COL2IM_WGSL = `
/**
 * @struct Params
 * @brief 합성곱(Convolution) 연산의 역전파 과정에서 필요한 col2im (Column to Image) 연산용 파라미터 구조체입니다. (What)
 * im2col을 통해 펼쳐진 행렬 형태의 그레이디언트를 다시 원래 텐서(이미지) 형태로 복원하기 위한 정보를 담고 있습니다. (Why)
 */
struct Params {
  // 배치 크기 (Batch size)
  N: u32,
  // 채널의 개수 (Channels)
  C: u32,
  // 원본 입력 텐서의 높이 (Height)
  H: u32,
  // 원본 입력 텐서의 너비 (Width)
  W: u32,
  // 합성곱 커널의 높이 크기
  K_h: u32,
  // 합성곱 커널의 너비 크기
  K_w: u32,
  // 필터 이동 보폭 (Stride)
  stride: u32,
  // 텐서 테두리에 덧붙인 패딩 크기
  padding: u32,
  // 합성곱 연산 결과 출력 텐서의 높이
  H_out: u32,
  // 합성곱 연산 결과 출력 텐서의 너비
  W_out: u32,
  // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수입니다.
  workgroups_x: u32,
  pad1: u32,
};

// params: col2im 역산 및 복원 계산을 위한 각종 텐서 차원들을 포함한 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// grad_x_col: im2col 형태로 전개되어 있던 그레이디언트 1차원 배열입니다 (읽기 전용).
@group(0) @binding(1) var<storage, read> grad_x_col: array<f32>;
// grad_x: 다시 원래 이미지 크기(N, C, H, W)로 합산 복원될 입력 텐서에 대한 그레이디언트 배열입니다.
@group(0) @binding(2) var<storage, read_write> grad_x: array<f32>;

/**
 * @function main
 * @brief 컴퓨트 셰이더의 메인 함수로, 원본 이미지의 픽셀 인덱스별로 연관되었던 모든 커널 윈도우들의 기울기(gradient)를 합산(accumulate)합니다. (What)
 * CNN 합성곱 층에서 입력값에 대한 역전파(Backpropagation)를 수행하여 가중치 갱신에 필요한 값을 도출하기 위해 (Why) 작성되었습니다.
 * 
 * @param global_id 워크그룹 내 스레드의 3차원 전역 인덱스입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 2D 디스패치 그리드로부터 복원한 현재 스레드가 처리할 원본 텐서 상의 1차원 인덱스입니다. (How)
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  // 전체 요소 개수 = 배치 * 채널 * 높이 * 너비 를 계산합니다.
  let num_elements = params.N * params.C * params.H * params.W;
  
  // 인덱스가 전체 크기를 벗어나면 즉시 함수를 빠져나가(return) 오류를 막습니다. (Why)
  if (idx >= num_elements) { return; }

  // 1차원 인덱스 idx를 4차원 좌표인 (n, c, h, w)로 복원하기 위한 임시 변수입니다. (How)
  var temp = idx;
  // 너비 차원 (Width) 복원
  let w = temp % params.W;
  temp = temp / params.W;
  // 높이 차원 (Height) 복원
  let h = temp % params.H;
  temp = temp / params.H;
  // 채널 차원 (Channel) 복원
  let c = temp % params.C;
  // 배치 차원 (Batch) 복원
  let n = temp / params.C;

  // 원본 텐서의 특정 픽셀 (n, c, h, w)에 모여들 그레이디언트 값을 누적하기 위한 실수형 변수입니다. (What)
  var val = 0.0;
  
  // 커널의 높이(K_h)만큼 반복하며 이 픽셀에 영향을 주었던 합성곱 윈도우들을 역추적합니다. (How)
  for (var k_h = 0u; k_h < params.K_h; k_h = k_h + 1u) {
    // 패딩이 적용된 높이 좌표를 계산합니다. (What)
    let h_plus_pad = h + params.padding;
    
    // 현재 커널 인덱스 k_h보다 크거나 같은지 검사하여 필터 범위를 벗어나지 않았는지 판단합니다. (Why)
    if (h_plus_pad >= k_h) {
      // 커널 내부에서의 오프셋을 제거하여 원본 인덱스를 역계산합니다. (How)
      let h_rem = h_plus_pad - k_h;
      // 스트라이드(stride) 조건에 맞게 정확하게 나누어 떨어지는 윈도우 위치인지 검사합니다. (What)
      if (h_rem % params.stride == 0u) {
        // 출력 텐서 상의 y좌표(h_out)를 복원 계산합니다.
        let h_out = h_rem / params.stride;
        // 계산된 출력 좌표가 실제 출력 텐서의 높이 범위 내에 있는지 검사합니다.
        if (h_out < params.H_out) {
          
          // 커널의 너비(K_w)만큼 반복하며 수평 방향 윈도우들을 탐색합니다. (How)
          for (var k_w = 0u; k_w < params.K_w; k_w = k_w + 1u) {
            // 패딩이 적용된 너비 좌표를 계산합니다.
            let w_plus_pad = w + params.padding;
            
            // 현재 커널 인덱스 k_w보다 크거나 같은지 확인하여 유효 범위인지 검사합니다.
            if (w_plus_pad >= k_w) {
              // 커널 너비 내의 오프셋을 제거합니다.
              let w_rem = w_plus_pad - k_w;
              // 수평 스트라이드 조건에 정확히 부합하는지 확인합니다. (What)
              if (w_rem % params.stride == 0u) {
                // 출력 텐서 상의 x좌표(w_out)를 계산합니다. (How)
                let w_out = w_rem / params.stride;
                // 계산된 출력 좌표가 실제 출력 텐서 너비 범위에 들어오는지 검증합니다.
                if (w_out < params.W_out) {
                  // 배치 번호는 원본과 동일하게 가져옵니다.
                  let n_out = n;
                  // 출력 평면 2D 상의 1차원 선형 인덱스(hw_out)를 계산합니다. (How)
                  let hw_out = h_out * params.W_out + w_out;
                  // 커널 안에서의 채널 및 2D 윈도우 인덱스(c_kw_kh)를 1차원으로 계산합니다.
                  let c_kw_kh = (c * params.K_h + k_h) * params.K_w + k_w;
                  
                  // 위에서 계산한 값들을 바탕으로, 2차원으로 전개되었던 grad_x_col 배열의 실제 접근 인덱스를 합성합니다. (What)
                  let col_idx = (n_out * (params.H_out * params.W_out) + hw_out) * (params.C * params.K_h * params.K_w) + c_kw_kh;
                  // 전개된 배열에서 가져온 기울기(gradient) 값을 현재 픽셀의 누적기(val)에 더합니다. (How)
                  val = val + grad_x_col[col_idx];
                }
              }
            }
          }
          
        }
      }
    }
  }

  // 역추적된 윈도우들로부터 누적 계산이 모두 끝난 총 그레이디언트 값을 출력 배열에 저장합니다. (What)
  grad_x[idx] = val;
}
`;

```

---

## `packages/forge/src/tensor/kernels/div.wgsl.ts`

```typescript
/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *   - 2026-08-18 14:10:00 +0900: Feat: Full 8D Multi-Dimensional Stride Broadcasting Decoder
 */
export const DIV_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  rank: u32,
  pad0: u32,
  dim0: u32, dim1: u32, dim2: u32, dim3: u32,
  dim4: u32, dim5: u32, dim6: u32, dim7: u32,
  stride_a0: u32, stride_a1: u32, stride_a2: u32, stride_a3: u32,
  stride_a4: u32, stride_a5: u32, stride_a6: u32, stride_a7: u32,
  stride_b0: u32, stride_b1: u32, stride_b2: u32, stride_b3: u32,
  stride_b4: u32, stride_b5: u32, stride_b6: u32, stride_b7: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    var temp = idx;
    let c7 = temp % params.dim7; temp = temp / params.dim7;
    let c6 = temp % params.dim6; temp = temp / params.dim6;
    let c5 = temp % params.dim5; temp = temp / params.dim5;
    let c4 = temp % params.dim4; temp = temp / params.dim4;
    let c3 = temp % params.dim3; temp = temp / params.dim3;
    let c2 = temp % params.dim2; temp = temp / params.dim2;
    let c1 = temp % params.dim1; temp = temp / params.dim1;
    let c0 = temp;

    let idx_a = c0 * params.stride_a0 + c1 * params.stride_a1 + c2 * params.stride_a2 + c3 * params.stride_a3 +
                c4 * params.stride_a4 + c5 * params.stride_a5 + c6 * params.stride_a6 + c7 * params.stride_a7;
    let idx_b = c0 * params.stride_b0 + c1 * params.stride_b1 + c2 * params.stride_b2 + c3 * params.stride_b3 +
                c4 * params.stride_b4 + c5 * params.stride_b5 + c6 * params.stride_b6 + c7 * params.stride_b7;

    out[idx] = a[idx_a] / b[idx_b];
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/dropout.wgsl.ts`

```typescript
/**
 * 생성일 (Created): 2026-08-12 12:59:35 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const DROPOUT_WGSL = `
/**
 * @struct Params
 * @brief 드롭아웃(Dropout) 연산을 수행하기 위해 필요한 메타데이터를 저장하는 구조체입니다. (What)
 * 과적합(Overfitting) 방지를 위해 무작위로 뉴런(값)을 0으로 끄는 확률(p)과 난수 시드(seed) 정보를 GPU에 전달하기 위해 사용됩니다. (Why)
 */
struct Params {
  // 드롭아웃을 적용할 전체 데이터 원소의 개수입니다.
  num_elements: u32,
  // 난수 생성의 기반이 되는 32비트 정수 시드(seed) 값입니다.
  seed: u32,
  // 드롭아웃 확률(p)입니다. (0.0 ~ 1.0) 이 확률보다 낮게 난수가 나오면 해당 값을 0으로 끕니다.
  p: f32,
  // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수
  workgroups_x: u32,
}

// params: 드롭아웃 파라미터를 담고 있는 Uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// x: 입력 데이터를 보관하고 있는 텐서(1차원 배열)입니다. (읽기 전용)
@group(0) @binding(1) var<storage, read> x: array<f32>;
// out: 드롭아웃 적용 이후 결과가 저장될 출력 데이터 텐서입니다.
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

/**
 * @function pcg_hash
 * @brief PCG (Permuted Congruential Generator) 기반의 해시 함수를 통해 32비트 정수형 난수를 생성합니다. (What)
 * 셰이더 내부에는 내장된 난수 생성기가 없으므로, 인덱스와 시드를 바탕으로 빠르고 균일하게 의사 난수(Pseudo Random)를 만들기 위해 고안되었습니다. (Why)
 * @param input 난수의 입력이 되는 시드 역할의 부호 없는 정수입니다. (How)
 * @return 해시 변환된 새로운 32비트 난수(u32)를 반환합니다.
 */
fn pcg_hash(input: u32) -> u32 {
    // 입력된 정수에 큰 소수를 곱하고 상수를 더해 초기 상태(state)를 섞습니다. (How)
    var state = input * 747796405u + 2891336453u;
    // 비트 시프트 연산과 XOR을 통해 비트 패턴을 비선형적으로 한 번 더 혼합합니다. (How)
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    // 최종적으로 비트 이동 후 XOR하여 고품질의 난수를 반환합니다.
    return (word >> 22u) ^ word;
}

/**
 * @function rand_f32
 * @brief 32비트 정수 형태의 난수를 0.0 이상 1.0 미만의 부동소수점(float) 형태로 정규화합니다. (What)
 * 드롭아웃 확률(p)과 직접 크기를 비교하기 위해 0~1 사이의 값이 필요하기 때문입니다. (Why)
 * @param hash pcg_hash로부터 전달받은 32비트 무작위 정수입니다.
 * @return 0.0과 1.0 사이로 매핑된 실수 난수입니다. (How)
 */
fn rand_f32(hash: u32) -> f32 {
    // 32비트 정수의 최대값(4294967295)으로 나누어 0~1 범위의 실수로 변환합니다. (How)
    return f32(hash) / 4294967295.0;
}

/**
 * @function main
 * @brief 스레드별로 난수를 발생시켜 지정된 확률 p 미만이면 0을, 그 이상이면 스케일링된 원본 값을 출력 텐서에 기록합니다. (What)
 * 신경망 학습 시 특정 노드에 과도하게 의존하는 현상을 막기 위해(Why) 병렬 스레드를 이용하여 고속으로 무작위 노드 비활성화를 수행합니다.
 * @param global_id 워크그룹 및 스레드의 3차원 전역 인덱스입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 2차원(Y) 워크그룹 구조를 1차원으로 풀어 현재 스레드의 전역 선형 인덱스를 계산합니다. (How)
    let index = global_id.x + global_id.y * params.workgroups_x * 64u;
    
    // 계산된 인덱스가 전체 텐서의 원소 수보다 크거나 같으면 실행을 즉시 중단합니다. (What)
    // 올바르지 않은 메모리 범위를 건드리지 않도록 차단하는 역할입니다. (Why)
    if (index >= params.num_elements) {
        return;
    }
    
    // 현재 인덱스와 외부에서 입력받은 32비트 시드를 조합하여 난수를 생성합니다. (How)
    let hash = pcg_hash(index + params.seed);
    // 정수 형태의 해시를 0.0 ~ 1.0 사이의 실수 난수로 변환합니다. (How)
    let rand = rand_f32(hash);
    
    // 생성된 난수가 설정된 드롭아웃 확률 p보다 작은지 검사합니다. (What)
    if (rand < params.p) {
        // 확률 분포에 걸렸을 경우(노드 비활성화), 해당 인덱스의 출력값을 0.0으로 만듭니다. (How)
        out[index] = 0.0;
    } else {
        // 확률 분포에 걸리지 않은 경우, 원본 데이터를 그대로 유지하되 기대값을 보존하기 위해 1/(1-p) 만큼 스케일링(Scaling)하여 저장합니다. (How)
        // 이는 Inverted Dropout 기법으로, 테스트 단계에서 별도의 스케일링 작업 없이 바로 모델을 쓸 수 있게 만들기 위함입니다. (Why)
        out[index] = x[index] * (1.0 / (1.0 - params.p));
    }
}
`;

```

---

## `packages/forge/src/tensor/kernels/exp.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const EXP_WGSL = `
/**
 * 이 구조체(Params)는 워크그룹과 데이터의 크기를 설정하기 위해 존재합니다.
 * 패딩 변수들은 WebGPU 버퍼의 16바이트 정렬 규칙을 준수하기 위해 사용됩니다.
 */
struct Params {
  size: u32, // 처리해야 할 전체 요소의 개수입니다.
  workgroups_x: u32, // X축 방향으로 스패닝된 워크그룹의 총 개수입니다.
  pad2: u32, // 메모리 정렬을 위해 존재하는 사용되지 않는 패딩 변수입니다.
  pad3: u32, // 메모리 정렬을 위해 존재하는 사용되지 않는 패딩 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 전달되는 상수 파라미터입니다.
@group(0) @binding(1) var<storage, read> x: array<f32>; // 읽기 전용으로 설정된 입력 텐서 데이터입니다.
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // 연산 결과가 쓰여질 출력 텐서 데이터입니다.

/**
 * main 함수는 각 텐서 요소에 대해 자연 상수 e를 밑으로 하는 지수 함수(exp) 연산을 수행합니다.
 * 이 함수가 존재하는 이유는 텐서의 모든 원소에 대해 병렬적으로 지수 연산을 처리하기 위함입니다.
 * GPU의 각 스레드는 고유한 global_id를 받아 배열 내 자신의 작업 위치를 계산하고 결과를 출력 버퍼에 저장합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size; // 전체 계산해야 하는 원소의 개수를 가져옵니다.
  let workgroups_x = params.workgroups_x; // 3D 그리드 기반의 1D 인덱스 계산을 위해 x축 워크그룹 수를 가져옵니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u; // 2D 형태로 스패닝된 글로벌 ID를 1D 인덱스로 변환하여 현재 스레드가 처리할 데이터의 위치를 구합니다.
  
  // 현재 스레드의 인덱스가 전체 배열 크기를 초과하면, 더 이상 처리하지 않고 함수를 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 계산된 인덱스의 입력값 x[idx]에 대해 지수 함수를 적용한 뒤, 그 결과를 출력 배열 y의 동일한 위치에 저장합니다.
  y[idx] = exp(x[idx]);
}
`;

```

---

## `packages/forge/src/tensor/kernels/fill.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const FILL_WGSL = `
/**
 * 이 구조체(Params)는 텐서를 특정 값으로 채우기 위한 설정 정보를 담고 있습니다.
 * GPU에 유니폼 버퍼를 통해 전달되며, 16바이트 정렬을 맞추기 위해 패딩을 포함합니다.
 */
struct Params {
  numElements: u32, // 값을 채울 배열의 전체 요소 개수입니다.
  value: f32, // 배열을 채울 특정 단일 부동 소수점 값입니다.
  workgroups_x: u32, // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 개수입니다.
  pad2: u32, // 메모리 정렬을 위해 추가된 두 번째 패딩용 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에서 읽어들일 유니폼 데이터입니다.
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // 채워진 값이 쓰여질 출력 버퍼입니다.

/**
 * main 함수는 출력 배열의 모든 요소에 지정된 값을 병렬로 기록합니다.
 * 텐서를 특정 상수값으로 초기화하는 fill 연산을 GPU에서 고속으로 수행하기 위해 존재합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.numElements; // 전체 요소 개수를 유니폼 변수에서 가져옵니다.
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u; // 2D 디스패치 선형 인덱스 복원
  
  // 계산된 인덱스가 전체 배열 크기보다 크거나 같다면 작업을 수행하지 않고 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 지정된 인덱스 위치에 설정된 상수 값(params.value)을 저장합니다.
  output[idx] = params.value;
}
`;

```

---

## `packages/forge/src/tensor/kernels/gather.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-12 12:23:09 +0900 (commit fc28607f9d46845175a9bdaf0e9e8c44bace5ecb)
 * 수정 이력:
 * - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
 */
export const GATHER_WGSL = `
/**
 * 이 구조체(Params)는 gather 연산에 필요한 형태(shape), 차원(stride), 대상 차원(dim) 정보를 담고 있습니다.
 * 다차원 텐서 인덱싱을 1차원 메모리에서 올바르게 계산하기 위한 정보를 제공하기 위해 존재합니다.
 */
struct Params {
  num_elements: u32, // 출력 텐서의 총 원소 개수입니다.
  dim: u32, // 요소를 수집할(gather) 대상 차원(axis)의 인덱스입니다.
  rank: u32, // 텐서의 차원 수 (랭크)입니다.
  workgroups_x: u32, // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수
  x_strides: array<u32, 8>, // 원본 입력 텐서의 각 차원별 스트라이드(보폭)입니다.
  out_strides: array<u32, 8>, // 출력 텐서의 각 차원별 스트라이드(보폭)입니다.
  x_shape: array<u32, 8>, // 원본 입력 텐서의 모양(각 차원의 크기)입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // 메타데이터 및 형태 정보가 담긴 유니폼 데이터입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // 수집 대상이 되는 원본 데이터 배열입니다.
@group(0) @binding(2) var<storage, read> index: array<f32>; // 수집할 인덱스를 지정하는 배열입니다.
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // 수집된 데이터가 쓰여질 결과 배열입니다.

/**
 * main 함수는 다차원 텐서에서 지정된 축(dim)을 기준으로 index 배열에 명시된 위치의 값들을 가져와 출력 텐서를 생성합니다.
 * PyTorch/NumPy의 gather 연산을 GPU에서 병렬 처리하기 위해 존재하며, 각 스레드는 출력 배열의 한 요소에 대응합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u; // 2D 디스패치 선형 인덱스 복원
  
  // 현재 처리할 인덱스가 전체 요소 수를 초과하면 실행을 중단합니다.
  if (idx >= params.num_elements) { return; }

  var temp = idx; // 다차원 좌표를 계산하기 위해 인덱스를 임시 변수에 복사합니다.
  var in_idx = 0u; // 입력 텐서에서 실제 참조해야 할 1D 메모리 인덱스를 누적할 변수입니다.

  // 출력 텐서의 각 차원(0부터 rank-1까지)에 대해 루프를 돕니다.
  // 이 루프는 출력 텐서의 1D 인덱스(idx)를 다차원 좌표로 변환하고, 이를 다시 입력 텐서의 1D 인덱스(in_idx)로 매핑합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    let coord = temp / params.out_strides[i]; // 현재 차원 i에서의 다차원 좌표 값입니다.
    temp = temp % params.out_strides[i]; // 다음 하위 차원 좌표 계산을 위해 나머지를 구합니다.
    
    // 현재 차원이 수집 대상 차원(dim)인 경우, 계산된 좌표 대신 index 배열에서 값을 읽어옵니다.
    if (i == params.dim) {
      let raw_idx = index[idx];
      let dim_size = i32(params.x_shape[i]);
      var signed_idx = i32(raw_idx);
      if (signed_idx < 0) {
        signed_idx = signed_idx + dim_size;
      }
      let clamped_idx = u32(clamp(signed_idx, 0, max(0, dim_size - 1)));
      in_idx = in_idx + clamped_idx * params.x_strides[i];
    } else {
      // 수집 대상 차원이 아닌 경우, 출력 텐서와 동일한 좌표를 유지합니다.
      in_idx = in_idx + coord * params.x_strides[i]; // 동일한 좌표에 원본 텐서의 해당 차원 스트라이드를 곱해 누적합니다.
    }
  }

  // 최종적으로 계산된 입력 텐서 인덱스(in_idx)의 값을 읽어 출력 텐서의 현재 인덱스(idx)에 저장합니다.
  output[idx] = input[in_idx];
}
`;

```

---

## `packages/forge/src/tensor/kernels/im2col.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-12 12:59:35 +0900 (commit 67c4ce9901dbb7caf2710e9ad03514f48956cfa6)
 * 수정 이력:
 * - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const IM2COL_WGSL = `
/**
 * 이 구조체(Params)는 이미지 데이터(공간적 텐서)를 열(Column) 기반 행렬로 변환하기 위한 컨볼루션 인자들을 담고 있습니다.
 * 입력 이미지의 크기, 커널(필터)의 크기, 스트라이드, 패딩 등 im2col 연산에 필수적인 하이퍼파라미터를 제공하기 위해 존재합니다.
 */
struct Params {
  N: u32, // 배치 크기 (Batch size)입니다.
  C: u32, // 입력 채널 수 (Channels)입니다.
  H: u32, // 입력 이미지의 원본 높이 (Height)입니다.
  W: u32, // 입력 이미지의 원본 너비 (Width)입니다.
  K_h: u32, // 커널(필터)의 높이입니다.
  K_w: u32, // 커널(필터)의 너비입니다.
  stride: u32, // 합성곱 연산 시 필터가 이동하는 보폭(스트라이드)입니다.
  padding: u32, // 입력 이미지 가장자리에 추가할 제로 패딩의 크기입니다.
  H_out: u32, // 연산 후 생성될 출력 특성 맵의 높이입니다.
  W_out: u32, // 연산 후 생성될 출력 특성 맵의 너비입니다.
  workgroups_x: u32, // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수입니다.
  pad1: u32, // 16바이트 메모리 정렬을 위한 패딩입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 컨볼루션 설정값을 전달하는 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // NCHW 형태로 펼쳐진 원본 이미지 입력 배열입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 변환된 행렬 형태의 데이터가 기록될 출력 배열입니다.

/**
 * main 함수는 합성곱(Convolution) 연산을 행렬 곱(MatMul)으로 효율적으로 수행하기 위해
 * 이미지 데이터의 국소적 패치(Local patch)를 추출하여 2D 행렬 형태로 재배치(im2col)합니다.
 * 이를 통해 GPU 상에서 고속의 GEMM(General Matrix Multiply) 라이브러리 및 최적화를 활용할 수 있습니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u; // 2D 디스패치 선형 인덱스 복원
  // 변환될 출력 배열의 총 요소 개수를 계산합니다 (N * H_out * W_out * C * K_h * K_w).
  let num_elements = params.N * params.H_out * params.W_out * params.C * params.K_h * params.K_w;
  
  // 계산할 인덱스가 배열 크기를 넘어가면 실행을 종료합니다.
  if (idx >= num_elements) { return; }

  var temp = idx; // 1차원 인덱스를 다차원 인덱스로 역계산하기 위해 임시 변수에 저장합니다.
  
  // 출력 버퍼의 인덱스에서 채널 및 커널 위치에 해당하는 차원 값을 추출합니다.
  let c_kw_kh = temp % (params.C * params.K_h * params.K_w);
  temp = temp / (params.C * params.K_h * params.K_w); // 다음 차원 추출을 위해 값을 나눕니다.
  
  // 출력 특성 맵의 공간적 위치(높이, 너비) 차원 값을 추출합니다.
  let h_out_w_out = temp % (params.H_out * params.W_out);
  temp = temp / (params.H_out * params.W_out); // 다음 차원 추출을 위해 값을 나눕니다.
  
  // 최종적으로 배치(Batch) 인덱스를 추출합니다.
  let n = temp % params.N;

  // 커널 내에서의 로컬 x, y 좌표 및 채널 인덱스를 계산합니다.
  let k_w = c_kw_kh % params.K_w; // 커널 내에서의 너비 인덱스
  let k_h = (c_kw_kh / params.K_w) % params.K_h; // 커널 내에서의 높이 인덱스
  let c = c_kw_kh / (params.K_w * params.K_h); // 입력 채널 인덱스

  // 출력 특성 맵 내에서의 x, y 좌표를 계산합니다.
  let w_out = h_out_w_out % params.W_out; // 출력 맵에서의 너비 위치
  let h_out = h_out_w_out / params.W_out; // 출력 맵에서의 높이 위치

  // 커널 위치와 스트라이드, 패딩을 고려하여 원본 입력 이미지 상의 실제 y, x 좌표를 역산합니다.
  let h_in = i32(h_out * params.stride) - i32(params.padding) + i32(k_h);
  let w_in = i32(w_out * params.stride) - i32(params.padding) + i32(k_w);

  // 계산된 원본 위치가 이미지 경계 내부인지 검사합니다.
  if (h_in >= 0 && h_in < i32(params.H) && w_in >= 0 && w_in < i32(params.W)) {
    // 경계 내부라면 NCHW 포맷에 따라 입력 배열의 1D 인덱스를 계산하고 값을 가져와 저장합니다.
    let in_idx = ((n * params.C + c) * params.H + u32(h_in)) * params.W + u32(w_in);
    output[idx] = input[in_idx];
  } else {
    // 경계 밖이라면 패딩 영역이므로 0.0을 채워 넣습니다.
    output[idx] = 0.0;
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/log.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const LOG_WGSL = `
/**
 * 이 구조체(Params)는 워크그룹과 데이터의 크기를 설정하기 위해 존재합니다.
 * 패딩 변수들은 WebGPU 버퍼의 16바이트 정렬 규칙을 준수하기 위해 사용됩니다.
 */
struct Params {
  size: u32, // 처리해야 할 텐서의 전체 원소 개수입니다.
  workgroups_x: u32, // X축을 따라 생성된 워크그룹의 총 개수입니다.
  pad2: u32, // 16바이트 메모리 정렬을 위해 남겨둔 미사용 변수입니다.
  pad3: u32, // 16바이트 메모리 정렬을 위해 남겨둔 미사용 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // 메타데이터 및 설정값이 담긴 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> x: array<f32>; // 자연로그 연산을 수행할 대상이 되는 입력 텐서입니다.
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // 자연로그 연산 결과가 저장될 출력 텐서입니다.

/**
 * main 함수는 입력 텐서의 각 요소에 대하여 자연로그(log) 연산을 수행합니다.
 * 요소별(element-wise) 자연로그 연산을 GPU의 병렬 처리 능력을 통해 가속화하기 위해 존재합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size; // 텐서의 전체 원소 개수를 유니폼 변수로부터 가져옵니다.
  let workgroups_x = params.workgroups_x; // 1차원 인덱스로 변환하기 위해 X축 워크그룹 크기를 가져옵니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u; // 2D 형태의 global_id를 1차원 평면 인덱스로 펼쳐서 현재 스레드의 작업 위치를 결정합니다.
  
  // 계산된 현재 스레드의 인덱스가 전체 텐서 크기를 벗어나면 작업을 수행하지 않고 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 현재 인덱스에 해당하는 입력 텐서 값에 대해 내장 함수 log()를 호출하고, 그 결과를 출력 텐서의 동일 위치에 저장합니다.
  y[idx] = log(x[idx]);
}
`;

```

---

## `packages/forge/src/tensor/kernels/matmul.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const MATMUL_WGSL = `
/**
 * 이 구조체(Params)는 행렬 곱셈 연산(A x B = C)에 필요한 차원 정보와 오프셋을 제공하기 위해 존재합니다.
 * A행렬은 (M x K), B행렬은 (K x N), C행렬은 (M x N) 차원을 가집니다.
 */
struct Params {
  M: u32, // 행렬 A와 C의 행(Row) 개수입니다.
  N: u32, // 행렬 B와 C의 열(Column) 개수입니다.
  K: u32, // 행렬 A의 열(Column) 개수이자 행렬 B의 행(Row) 개수입니다 (내적을 수행할 길이).
  offsetY: u32, // 워크그룹 파견 한계(dispatch limit)를 우회하기 위해 y축 시작 오프셋을 지정합니다.
};

@group(0) @binding(0) var<uniform> params: Params; // 행렬의 형태 및 오프셋 정보를 담은 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> a: array<f32>; // (M x K) 크기의 첫 번째 입력 행렬 데이터입니다.
@group(0) @binding(2) var<storage, read> b: array<f32>; // (K x N) 크기의 두 번째 입력 행렬 데이터입니다.
@group(0) @binding(3) var<storage, read_write> c: array<f32>; // 결과값(M x N)이 기록될 출력 행렬 데이터입니다.

/**
 * main 함수는 두 행렬 A와 B를 곱하여 결과 행렬 C를 계산합니다.
 * 딥러닝에서 가장 핵심적인 연산인 GEMM(General Matrix Multiply)을 GPU로 분산하여 병렬 처리하기 위해 존재합니다.
 */
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // global_id.z 를 X축 타일 오프셋으로 사용
  // dispatcher가 z = ceil(N / (65535*8))만큼 dispatch
  
  // 현재 스레드가 계산을 담당할 출력 행렬 C의 열(Column) 인덱스를 계산합니다.
  // z축 워크그룹 인덱스를 사용하여 1D 한계를 넘는 큰 행렬에 대한 스팬(span)을 지원합니다.
  let col = global_id.x + global_id.z * 65535u * 8u;
  // 현재 스레드가 계산을 담당할 출력 행렬 C의 행(Row) 인덱스를 계산합니다 (오프셋 포함).
  let row = global_id.y + params.offsetY;

  // 계산된 인덱스가 행렬 C의 범위를 초과하는 스레드는 작업을 수행하지 않고 바로 종료합니다.
  if (row >= params.M || col >= params.N) {
    return;
  }

  // A행렬의 row번째 행과 B행렬의 col번째 열 사이의 내적(Dot product)을 누적할 변수입니다.
  var sum: f32 = 0.0;
  
  // 내적을 수행하기 위해 공통 차원인 K번 만큼 루프를 돕니다.
  // A의 원소와 B의 원소를 순차적으로 곱하여 합산합니다.
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + a[row * params.K + k] * b[k * params.N + col];
  }

  // 계산된 최종 내적 값을 결과 행렬 C의 해당하는 1D 인덱스 위치에 저장합니다.
  c[row * params.N + col] = sum;
}
`;

```

---

## `packages/forge/src/tensor/kernels/matmul_bias_relu.wgsl.ts`

```typescript
/**
 * AMEVA-Forge Fused Linear Kernel: MatMul + BiasAdd + ReLU
 * Computes C = ReLU(A @ B + Bias) in a single GPU compute pass.
 */
export const MATMUL_BIAS_RELU_WGSL = `
struct Params {
  M: u32,
  N: u32,
  K: u32,
  offsetY: u32,
  has_bias: u32,
  has_relu: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let col = global_id.x + global_id.z * 65535u * 8u;
  let row = global_id.y + params.offsetY;

  if (row >= params.M || col >= params.N) {
    return;
  }

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + a[row * params.K + k] * b[k * params.N + col];
  }

  if (params.has_bias == 1u) {
    sum = sum + bias[col];
  }

  if (params.has_relu == 1u) {
    sum = max(sum, 0.0);
  }

  c[row * params.N + col] = sum;
}
`;

```

---

## `packages/forge/src/tensor/kernels/max.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * - 2026-08-18 00:30:00 +0900: Fix(SCRUM-157/VULN-05): 2D workgroup linear index reconstruction for >65535 reductions
 */
export const MAX_WGSL = `
/**
 * 이 구조체(Params)는 텐서의 최댓값을 구하는 Reduction(리덕션) 연산에 필요한 정보를 담고 있습니다.
 * 요소의 전체 개수와 2D 분할 그리드 정보를 전달하여 버퍼 경계를 넘는 접근을 방지하고 선형 인덱스를 복원합니다.
 */
struct Params {
  numElements: u32, // 최댓값을 찾을 전체 배열 원소의 개수입니다.
  workgroups_x: u32, // 65,535 초과 시 분할된 2D 그리드의 X축 워크그룹 수입니다.
  pad1: u32, // 16바이트 정렬을 위한 첫 번째 패딩 변수입니다.
  pad2: u32, // 16바이트 정렬을 위한 두 번째 패딩 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 전달되는 메타데이터 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // 최댓값을 탐색할 원본 입력 텐서입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 워크그룹별 부분 최댓값이 저장될 출력 텐서입니다.

// 하나의 워크그룹(256개의 스레드) 내에서 데이터를 공유하고 리덕션을 수행하기 위해 존재하는 공유 메모리 공간입니다.
var<workgroup> s_data: array<f32, 256>;

/**
 * main 함수는 트리 기반의 리덕션(Tree-based Reduction) 알고리즘을 사용하여 배열 내 원소들의 최댓값을 계산합니다.
 * 방대한 데이터를 병렬로 빠르게 비교압축하기 위해 공유 메모리(s_data)와 배리어(barrier) 동기화를 사용합니다.
 */
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  let wg_linear = workgroup_id.y * params.workgroups_x + workgroup_id.x; // 2D 워크그룹 좌표에서 선형 인덱스를 복원합니다.
  let gid = wg_linear * 256u + local_id.x; // 글로벌 단위에서 현재 스레드의 1차원 인덱스입니다.
  let lid = local_id.x; // 워크그룹 내부에서 현재 스레드의 1차원 인덱스(0~255)입니다.
  let wid = wg_linear; // 현재 스레드가 속한 워크그룹의 선형 ID(인덱스)입니다.
  
  // 글로벌 인덱스가 데이터 크기 이내라면 입력 데이터를, 벗어난다면 부동소수점의 최소값(-FLT_MAX)을 공유 메모리에 로드합니다.
  if (gid < params.numElements) {
    s_data[lid] = input[gid];
  } else {
    s_data[lid] = -3.402823e+38; // 쓰레기값을 방지하기 위한 최소값 초기화입니다.
  }
  
  // 공유 메모리 로드가 완전히 끝날 때까지 워크그룹 내의 모든 스레드를 대기시킵니다.
  workgroupBarrier();
  
  // 트리 기반 병렬 리덕션 루프입니다.
  // 활성화된 스레드 수를 절반씩 줄여가면서(128 -> 64 -> ... -> 1) 두 요소씩 비교해 최댓값을 찾습니다.
  for (var s = 128u; s > 0u; s >>= 1u) {
    // 현재 단계에서 값을 비교하고 갱신할 권한이 있는 스레드만 실행합니다.
    if (lid < s) {
      s_data[lid] = max(s_data[lid], s_data[lid + s]); // 자신의 값과 s만큼 떨어진 옆의 값을 비교해 큰 값을 저장합니다.
    }
    // 데이터 경합(Data Race)을 막고 다음 단계를 안전하게 수행하기 위해 스레드 동기화를 수행합니다.
    workgroupBarrier();
  }
  
  // 리덕션이 완료되면 공유 메모리의 0번 인덱스에 현재 워크그룹의 전체 최댓값이 남게 됩니다.
  // 0번 스레드가 이를 대표로 전역 출력 버퍼에 기록합니다.
  if (lid == 0u) {
    output[wid] = s_data[0];
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/max_axis.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-18T12:05:00+09:00
 * 역할: 축 방향 최댓값 리덕션 (Max Reduction Along Axis) WGSL 커널
 * 목적: Softmax의 수치적 안정성(x - max(x)) 및 축별 Max 연산을 GPU에서 고속 병렬 처리하기 위함
 */
export const MAX_AXIS_WGSL = `
struct Params {
  outer_size: u32,     // 축소 축 이전의 외부 배치/차원들의 곱
  reduction_size: u32, // 축소할 대상 축의 원소 개수 (Reduction Dimension Size)
  inner_stride: u32,   // 축소 축 이후의 내부 차원들의 스트라이드 곱
  output_numel: u32,   // 결과 텐서의 총 원소 개수 (outer_size * inner_stride)
  workgroups_x: u32,   // 2D 디스패치 분할을 위한 X축 워크그룹 수
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let workgroups_x = params.workgroups_x;
  let out_idx = global_id.x + global_id.y * workgroups_x * 64u;

  if (out_idx >= params.output_numel) {
    return;
  }

  let inner_stride = params.inner_stride;
  let reduction_size = params.reduction_size;
  let outer_idx = out_idx / inner_stride;
  let inner_idx = out_idx % inner_stride;
  let slice_stride = reduction_size * inner_stride;
  let base_offset = outer_idx * slice_stride + inner_idx;

  var max_val = -3.402823e+38;
  for (var r = 0u; r < reduction_size; r = r + 1u) {
    let val = input[base_offset + r * inner_stride];
    if (val > max_val) {
      max_val = val;
    }
  }
  output[out_idx] = max_val;
}
`;

```

---

## `packages/forge/src/tensor/kernels/max_axis_backward.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-18T13:20:00+09:00
 * 역할: 축 방향 최댓값 역전파 (Max Reduction Backward Along Axis) WGSL 커널
 * 목적: GPU 상에서 x.max(axis).backward() 호출 시 기울기 전파 및 중복 최댓값 분산 처리
 */
export const MAX_AXIS_BACKWARD_WGSL = `
struct Params {
  outer_size: u32,
  reduction_size: u32,
  inner_stride: u32,
  input_numel: u32,
  workgroups_x: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> grad_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x + gid.y * params.workgroups_x * 64u;
  if (linear >= params.input_numel) {
    return;
  }

  let inner = linear % params.inner_stride;
  let tmp = linear / params.inner_stride;
  let r = tmp % params.reduction_size;
  let outer = tmp / params.reduction_size;

  let reduced_idx = outer * params.inner_stride + inner;

  var max_val = -3.402823e+38;
  for (var j: u32 = 0u; j < params.reduction_size; j = j + 1u) {
    let idx = outer * params.reduction_size * params.inner_stride + j * params.inner_stride + inner;
    max_val = max(max_val, x[idx]);
  }

  var count: f32 = 0.0;
  for (var j: u32 = 0u; j < params.reduction_size; j = j + 1u) {
    let idx = outer * params.reduction_size * params.inner_stride + j * params.inner_stride + inner;
    if (x[idx] == max_val) {
      count = count + 1.0;
    }
  }

  if (x[linear] == max_val && count > 0.0) {
    grad_x[linear] = grad_out[reduced_idx] / count;
  } else {
    grad_x[linear] = 0.0;
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/maxpool2d.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-12 12:59:35 +0900 (commit 67c4ce9901dbb7caf2710e9ad03514f48956cfa6)
 * 수정 이력:
 * - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const MAXPOOL2D_WGSL = `
/**
 * 이 구조체(Params)는 2D 맥스 풀링(Max Pooling 2D) 연산에 필요한 하이퍼파라미터 및 차원 정보를 담고 있습니다.
 * 입력 이미지의 배치, 채널, 크기 정보와 커널 크기, 스트라이드, 패딩 값을 전달하기 위해 존재합니다.
 */
struct Params {
    batch: u32, // 배치 크기입니다.
    channels: u32, // 입력 텐서의 채널 수입니다.
    in_h: u32, // 원본 입력 이미지의 높이입니다.
    in_w: u32, // 원본 입력 이미지의 너비입니다.
    out_h: u32, // 계산되어 출력될 이미지의 높이입니다.
    out_w: u32, // 계산되어 출력될 이미지의 너비입니다.
    kH: u32, // 풀링 커널(필터)의 높이입니다.
    kW: u32, // 풀링 커널(필터)의 너비입니다.
    sH: u32, // 높이 방향의 스트라이드(보폭)입니다.
    sW: u32, // 너비 방향의 스트라이드(보폭)입니다.
    pH: u32, // 높이 방향에 추가된 제로 패딩 크기입니다.
    pW: u32, // 너비 방향에 추가된 제로 패딩 크기입니다.
    workgroups_x: u32, // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수입니다.
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params; // GPU에 메타데이터를 공급하는 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // NCHW 형태의 입력 데이터 텐서입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 풀링 결과가 저장될 NCHW 형태의 출력 텐서입니다.

/**
 * main 함수는 합성곱 신경망(CNN)의 핵심 구성 요소인 2D 맥스 풀링 연산을 수행합니다.
 * 이미지의 국소 영역(커널 크기)에서 최댓값만을 추출하여 공간적 차원(Spatial Dimension)을 축소하기 위해 존재합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x + global_id.y * params.workgroups_x * 64u; // 2D 디스패치 선형 인덱스 복원
    // 연산이 필요한 전체 출력 데이터의 개수를 계산합니다.
    let total = params.batch * params.channels * params.out_h * params.out_w;
    
    // 할당된 스레드 인덱스가 유효 범위를 벗어나면 연산을 중단합니다.
    if (idx >= total) {
        return;
    }
    
    // 1D 인덱스에서 NCHW 포맷에 따라 출력 좌표 (ow, oh, c, b)를 역산합니다.
    let ow = idx % params.out_w; // 출력 맵의 너비(x) 좌표입니다.
    let oh = (idx / params.out_w) % params.out_h; // 출력 맵의 높이(y) 좌표입니다.
    let c = (idx / (params.out_w * params.out_h)) % params.channels; // 채널 인덱스입니다.
    let b = idx / (params.out_w * params.out_h * params.channels); // 배치 인덱스입니다.
    
    // 스트라이드와 패딩을 적용하여 입력 이미지 기준 시작 좌표를 계산합니다.
    let h_start = i32(oh * params.sH) - i32(params.pH);
    let w_start = i32(ow * params.sW) - i32(params.pW);
    
    // 최댓값 비교를 위한 초기값을 부동소수점 표현 가능한 가장 작은 값으로 설정합니다.
    var max_val = -3.402823466e+38; // -FLT_MAX
    
    // 커널의 높이(kH)와 너비(kW) 영역을 순회하며 최댓값을 찾기 위한 이중 루프입니다.
    for (var kh = 0u; kh < params.kH; kh++) {
        for (var kw = 0u; kw < params.kW; kw++) {
            // 커널 내 오프셋을 더하여 실제 입력 데이터 상의 좌표를 구합니다.
            let h = h_start + i32(kh);
            let w = w_start + i32(kw);
            
            // 계산된 좌표가 이미지 경계 안쪽에 있는지(유효한 데이터인지) 검사합니다.
            if (h >= 0 && h < i32(params.in_h) && w >= 0 && w < i32(params.in_w)) {
                // NCHW 포맷에 따른 입력 텐서의 1D 메모리 인덱스를 계산합니다.
                let in_idx = ((b * params.channels + c) * params.in_h + u32(h)) * params.in_w + u32(w);
                let val = input[in_idx]; // 입력값을 읽어옵니다.
                
                // 기존의 max_val과 비교하여 더 큰 값이면 갱신합니다.
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    // 커널 영역 전체에서 발견한 최댓값을 출력 텐서의 현재 인덱스에 저장합니다.
    output[idx] = max_val;
}
`;

```

---

## `packages/forge/src/tensor/kernels/mul.wgsl.ts`

```typescript
/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *   - 2026-08-18 14:10:00 +0900: Feat: Full 8D Multi-Dimensional Stride Broadcasting Decoder
 */
export const MUL_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  rank: u32,
  pad0: u32,
  dim0: u32, dim1: u32, dim2: u32, dim3: u32,
  dim4: u32, dim5: u32, dim6: u32, dim7: u32,
  stride_a0: u32, stride_a1: u32, stride_a2: u32, stride_a3: u32,
  stride_a4: u32, stride_a5: u32, stride_a6: u32, stride_a7: u32,
  stride_b0: u32, stride_b1: u32, stride_b2: u32, stride_b3: u32,
  stride_b4: u32, stride_b5: u32, stride_b6: u32, stride_b7: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    var temp = idx;
    let c7 = temp % params.dim7; temp = temp / params.dim7;
    let c6 = temp % params.dim6; temp = temp / params.dim6;
    let c5 = temp % params.dim5; temp = temp / params.dim5;
    let c4 = temp % params.dim4; temp = temp / params.dim4;
    let c3 = temp % params.dim3; temp = temp / params.dim3;
    let c2 = temp % params.dim2; temp = temp / params.dim2;
    let c1 = temp % params.dim1; temp = temp / params.dim1;
    let c0 = temp;

    let idx_a = c0 * params.stride_a0 + c1 * params.stride_a1 + c2 * params.stride_a2 + c3 * params.stride_a3 +
                c4 * params.stride_a4 + c5 * params.stride_a5 + c6 * params.stride_a6 + c7 * params.stride_a7;
    let idx_b = c0 * params.stride_b0 + c1 * params.stride_b1 + c2 * params.stride_b2 + c3 * params.stride_b3 +
                c4 * params.stride_b4 + c5 * params.stride_b5 + c6 * params.stride_b6 + c7 * params.stride_b7;

    out[idx] = a[idx_a] * b[idx_b];
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/neg.wgsl.ts`

```typescript
/**
 * 생성일: 2026-08-12T12:14:52+09:00
 * 수정 이력:
 * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const NEG_WGSL = `
// 구조체: Params
// 역할 (WHAT): 음수화(Negation) 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼 데이터를 전달하고 16바이트 정렬 규칙을 준수하기 위해 정의되었습니다.
// 동작 방식 (HOW): 셰이더가 실행될 때 전체 데이터 크기(size)와 2D 기반 분할 시 사용되는 x축 워크그룹 크기를 참조합니다.
struct Params {
  // 변수: size
  // 역할: 처리할 전체 데이터 배열의 요소 개수
  size: u32,
  // 변수: workgroups_x
  // 역할: X축 방향의 작업 그룹 수 (2D 그리드 인덱싱에 사용)
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 구조체의 메모리 정렬을 위한 여분(padding) 공간
  pad2: u32,
  pad3: u32,
};

// 변수: params
// 역할: Params 구조체를 저장하는 유니폼 버퍼 (인덱스 바인딩 0)
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 역할: 입력 데이터를 담는 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 역할: 계산 결과(음수화된 값)가 저장될 읽기/쓰기 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 역할 (WHAT): 입력 텐서의 모든 요소에 대해 부호를 반전시키는 메인 컴퓨트 함수입니다.
// 목적 (WHY): GPU의 병렬 아키텍처를 활용하여 빠르고 동시적인 부호 반전 연산을 수행하기 위함입니다.
// 동작 방식 (HOW): 64 워크그룹 사이즈 내에서 각 스레드가 전역 ID를 사용해 1D 인덱스를 계산하고, 유효한 범위 내에서 - 연산을 적용합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 역할: 유니폼 인자를 통해 전달된 전체 배열 크기를 저장합니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 2D 그리드 맵핑을 풀기 위한 가로(X축) 워크그룹의 개수를 저장합니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: idx
  // 역할: x 및 y 방향 워크그룹 ID와 로컬 ID를 결합하여 처리할 1D 데이터 인덱스를 계산합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: out-of-bounds 방지
  // 역할: 계산된 인덱스가 실제 데이터의 요소 개수를 초과하면 처리를 조기 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 변수 y 갱신
  // 역할: x 배열의 해당 인덱스 값을 읽어와 음수 기호를 붙인 후 y 배열에 씁니다.
  y[idx] = -x[idx];
}
`;

```

---

## `packages/forge/src/tensor/kernels/pad.wgsl.ts`

```typescript
/**
 * 생성일: 2026-08-12T12:23:09+09:00
 * 수정 이력:
 * - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 */
export const PAD_WGSL = `
// 구조체: Params
// 역할 (WHAT): 텐서 패딩 연산에 필요한 모든 형태(Shape), 보폭(Stride), 설정 변수들을 담고 있는 구조체입니다.
// 목적 (WHY): 패딩 된 새로운 텐서를 생성하기 위해 원본 텐서의 좌표와 출력 텐서의 좌표를 매핑하는 데 필요한 정보를 제공하기 위함입니다.
// 동작 방식 (HOW): 각 차원에 대한 크기, 원본/출력 메모리 보폭 정보, 추가할 패딩 값 등을 참조하여 변환된 인덱스를 계산합니다.
struct Params {
  // 변수: num_elements
  // 역할: 패딩이 완료된 최종 출력 텐서의 전체 요소 개수
  num_elements: u32,
  // 변수: rank
  // 역할: 텐서의 차원(Rank) 수
  rank: u32,
  // 변수: pad_val
  // 역할: 빈 공간에 채워 넣을 상수 값(패딩 값)
  pad_val: f32,
  // 변수: workgroups_x
  // 역할: 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수
  workgroups_x: u32,
  // 변수: in_strides
  // 역할: 최대 8차원까지 지원하는 원본 입력 텐서의 차원별 메모리 보폭(Stride) 배열
  in_strides: array<u32, 8>,
  // 변수: out_strides
  // 역할: 패딩 적용 후 출력 텐서의 차원별 메모리 보폭 배열
  out_strides: array<u32, 8>,
  // 변수: pad_before
  // 역할: 각 차원의 앞부분(before)에 추가되는 패딩의 크기를 저장하는 배열
  pad_before: array<u32, 8>,
  // 변수: in_shape
  // 역할: 입력 텐서의 원래 차원별 크기(Shape)를 저장하는 배열
  in_shape: array<u32, 8>,
};

// 변수: params
// 역할: 패딩 연산의 메타데이터를 저장하는 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 역할: 패딩 되기 전의 원본 데이터가 저장되어 있는 스토리지 버퍼
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 역할: 패딩 된 결과 데이터가 기록될 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 함수: main
// 역할 (WHAT): 출력 텐서의 각 인덱스를 기준으로 원본 인덱스를 역추적하여 값을 복사하거나 패딩 값을 채웁니다.
// 목적 (WHY): 입력 배열 주변에 원하는 크기와 값으로 여백(패딩)을 추가하여 크기가 확장된 텐서를 반환하기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 출력 1D 인덱스(idx)를 받아 다차원 좌표(coord)로 변환한 후, 이 좌표가 원본 텐서 영역 내인지 확인하여 원본 값을 쓰거나 pad_val을 채웁니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: idx
  // 역할: 2D 디스패치 그리드로부터 복원한 현재 스레드의 1차원 전역 인덱스
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  
  // 조건문: 인덱스 범위 확인
  // 역할: idx가 결과 텐서의 전체 크기를 넘어서면 실행을 중지하여 유효하지 않은 메모리 접근을 방지합니다.
  if (idx >= params.num_elements) { return; }

  // 변수: temp
  // 역할: 1차원 인덱스를 다차원 좌표로 분해할 때 남은 인덱스 값을 저장 및 갱신하기 위한 임시 변수
  var temp = idx;
  // 변수: in_idx
  // 역할: 역계산된 원본 텐서의 1차원 인덱스를 누적할 변수
  var in_idx = 0u;
  // 변수: in_bounds
  // 역할: 현재 계산하는 출력 좌표가 원본 텐서의 범위 안에 속해 있는지를 나타내는 불리언 플래그
  var in_bounds = true;

  // 반복문: for 루프 (차원 탐색)
  // 역할 (WHAT): 최고 차원부터 최하 차원까지 각 차원의 좌표를 구하고, 이를 이용해 원본 입력 텐서의 플랫(flat) 인덱스를 누적 연산합니다.
  // 목적 (WHY): N차원(최대 8차원) 데이터를 1차원 배열로 평탄화(Flatten)한 메모리 구조에서 정확한 매핑을 계산하기 위함입니다.
  // 동작 방식 (HOW): 나누기와 나머지 연산을 사용해 현재 차원의 좌표(coord)를 구한 뒤, 원본 텐서 구간(pad_before ~ pad_before + in_shape)에 속하는지 검사합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    // 변수: coord
    // 역할: 출력 텐서의 i번째 차원에 대한 구체적 좌표(인덱스)
    let coord = temp / params.out_strides[i];
    
    // 변수: temp 갱신
    // 역할: 다음 하위 차원 계산을 위해 남은 나머지 값을 임시 변수에 대입합니다.
    temp = temp % params.out_strides[i];
    
    // 조건문: 원본 영역 이탈 확인
    // 역할: 계산된 해당 차원의 좌표가 패딩 영역(원본 데이터가 없는 곳)인지 판단합니다.
    if (coord < params.pad_before[i] || coord >= params.pad_before[i] + params.in_shape[i]) {
      // 변수: in_bounds 갱신
      // 역할: 영역 바깥이므로 in_bounds를 false로 변경하고 루프를 탈출합니다.
      in_bounds = false;
      break;
    }
    
    // 변수: in_coord
    // 역할: 출력 텐서 좌표에서 앞부분 패딩(pad_before)을 빼서 원본 텐서 기준의 순수 좌표를 구합니다.
    let in_coord = coord - params.pad_before[i];
    
    // 변수: in_idx 누적
    // 역할: 구한 원본 좌표에 해당 차원의 보폭(in_strides)을 곱하여 1D 원본 인덱스를 점진적으로 계산합니다.
    in_idx = in_idx + in_coord * params.in_strides[i];
  }

  // 조건문: 값 삽입 결정
  // 역할: 구해진 플래그(in_bounds)를 바탕으로 배열에 원본 데이터를 쓸지, 패딩 값을 쓸지 분기합니다.
  if (in_bounds) {
    // 변수 output 갱신 (원본)
    // 역할: 출력 배열에 입력 배열의 데이터를 그대로 복사합니다.
    output[idx] = input[in_idx];
  } else {
    // 변수 output 갱신 (패딩)
    // 역할: 출력 배열에 미리 설정해 둔 패딩 상수 값(pad_val)을 삽입합니다.
    output[idx] = params.pad_val;
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/permute.wgsl.ts`

```typescript
/**
 * 생성일: 확인 불가 (Git 기록 없음 혹은 커밋 대기 상태)
 * 수정 이력:
 * - 특이사항 없음
 */
export const PERMUTE_WGSL = `
// 구조체: Params
// 역할 (WHAT): Permute(전치/축 교환) 연산에 필요한 차원, 형상(Shape) 및 보폭(Stride) 정보를 담은 구조체입니다.
// 목적 (WHY): 입력 텐서의 축을 지정된 순서대로 재배열하여 출력 텐서의 메모리 레이아웃을 계산하기 위해 유니폼 데이터를 전달합니다.
// 동작 방식 (HOW): rank와 총 요소 수를 제공하고, 최대 8차원을 지원하기 위해 vec4 두 개를 이어서 strides와 shape 정보를 제공합니다.
struct Params {
  // 변수: rank
  // 역할: 텐서가 가진 총 차원의 수
  rank: u32,
  // 변수: numElements
  // 역할: 텐서 내 존재하는 전체 데이터 요소의 개수
  numElements: u32,
  // 변수: workgroups_x
  // 역할: X축 방향으로 할당된 워크그룹(workgroup)의 총 개수
  workgroups_x: u32,
  // 변수: pad2
  // 역할: 16바이트 메모리 정렬을 위한 패딩 변수
  pad2: u32,
  // 변수: in_strides
  // 역할: 입력 텐서의 첫 4차원(0~3)에 대한 메모리 보폭
  in_strides: vec4<u32>,
  // 변수: in_strides_ext
  // 역할: 입력 텐서의 확장 4차원(4~7)에 대한 메모리 보폭
  in_strides_ext: vec4<u32>,
  out_shape: vec4<u32>,
  // 변수: out_shape_ext
  // 역할: 출력 텐서의 확장 4차원(4~7)에 대한 크기(Shape)
  out_shape_ext: vec4<u32>,
  // 변수: out_strides
  // 역할: 출력 텐서의 첫 4차원(0~3)에 대한 메모리 보폭
  out_strides: vec4<u32>,
  // 변수: out_strides_ext
  // 역할: 출력 텐서의 확장 4차원(4~7)에 대한 메모리 보폭
  out_strides_ext: vec4<u32>,
};

// 변수: params
// 역할: 셰이더 전역에서 접근 가능한 Permute 연산용 메타데이터 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 역할: 원본 데이터가 들어 있는 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 역할: 축이 변환된 최종 데이터가 저장될 쓰기 가능한 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 함수: main
// 역할 (WHAT): 출력 인덱스를 기반으로 다차원 좌표를 복원하고, 이를 입력 텐서의 보폭과 매칭하여 축 교환된 값을 저장합니다.
// 목적 (WHY): 텐서의 차원 순서를 바꾸는 연산(예: 행렬 전치, 채널 축 변경)을 GPU를 활용하여 병렬로 빠르게 수행하기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 2D 기반 ID를 통해 출력 인덱스(out_idx)를 얻고, 반복문을 통해 각 차원별 인덱스를 분리해내어, 원본 보폭과 곱하여 in_idx를 도출해 값을 복사합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: out_idx
  // 역할: 2D 그리드 디스패치(dispatch)를 지원하기 위해 글로벌 ID x, y를 결합하여 만든 1차원 출력 인덱스
  // 주석: Compute global index supporting 2D grid dispatch
  let out_idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  
  // 조건문: 데이터 범위 초과 검사
  // 역할: 스레드의 계산된 인덱스가 전체 요소 크기를 넘어서는지 판단하여 초과 시 연산을 중지합니다.
  if (out_idx >= params.numElements) {
    return;
  }
  
  // 변수: out_idx_remaining
  // 역할: 각 차원의 좌표를 구하기 위해 나누기/나머지 연산을 하면서 변해가는 임시 나머지 인덱스 값
  var out_idx_remaining = out_idx;
  // 변수: in_idx
  // 역할: 입력 텐서에서 실제 데이터를 읽어올 1차원 메모리 인덱스의 누적 값
  var in_idx = 0u;
  
  // 반복문: for 루프 (모든 차원에 대한 순회)
  // 역할 (WHAT): 최상위 차원부터 시작하여 현재 차원에 해당하는 좌표를 구하고, 이를 바탕으로 원래 입력 배열의 인덱스를 계산합니다.
  // 목적 (WHY): 다차원 구조가 평면 배열(flat array)로 선형화되어 있으므로, 출력의 구조를 풀어 입력의 구조로 맵핑해야 하기 때문입니다.
  // 동작 방식 (HOW): 0부터 rank-1까지 순회하면서 차원(i)에 맞는 보폭(Stride) 값을 가져오고, 좌표(coord)를 구한 후 입력 인덱스를 누적합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    // 변수: out_stride
    // 역할: 현재 루프 차원(i)에 해당하는 출력 텐서의 보폭
    var out_stride = 0u;
    // 변수: in_stride
    // 역할: 현재 루프 차원(i)에 해당하는 입력 텐서의 보폭
    var in_stride = 0u;
    
    // 조건문: 차원(i) 확인 및 보폭 할당
    // 역할 (WHAT): 루프 인덱스 i 값에 따라 vec4에 묶여 있는 각 차원의 보폭 값을 가져옵니다.
    // 목적 (WHY): WGSL에서는 배열 인덱싱을 지원하지 않는 vec4 구조체 필드에 동적으로 접근하기 위해 하드코딩 된 조건 분기가 필요하기 때문입니다.
    // 동작 방식 (HOW): i가 0~7 중 어느 것인지 확인하고, 해당하는 x, y, z, w 컴포넌트 값을 보폭 변수에 저장합니다.
    if (i == 0u) { out_stride = params.out_strides.x; in_stride = params.in_strides.x; }
    else if (i == 1u) { out_stride = params.out_strides.y; in_stride = params.in_strides.y; }
    else if (i == 2u) { out_stride = params.out_strides.z; in_stride = params.in_strides.z; }
    else if (i == 3u) { out_stride = params.out_strides.w; in_stride = params.in_strides.w; }
    else if (i == 4u) { out_stride = params.out_strides_ext.x; in_stride = params.in_strides_ext.x; }
    else if (i == 5u) { out_stride = params.out_strides_ext.y; in_stride = params.in_strides_ext.y; }
    else if (i == 6u) { out_stride = params.out_strides_ext.z; in_stride = params.in_strides_ext.z; }
    else if (i == 7u) { out_stride = params.out_strides_ext.w; in_stride = params.in_strides_ext.w; }
    
    // 변수: coord
    // 역할: 남은 1차원 인덱스를 출력 보폭으로 나누어 얻은 현재 차원(i)의 논리적 좌표값
    let coord = out_idx_remaining / out_stride;
    
    // 변수: out_idx_remaining 갱신
    // 역할: 다음 차원 계산을 위해 현재 차원에서 처리된 부분을 제외한 나머지(나머지 연산)를 저장합니다.
    out_idx_remaining = out_idx_remaining % out_stride;
    
    // 변수: in_idx 누적
    // 역할: 도출된 논리적 좌표(coord)에 원래 텐서의 보폭(in_stride)을 곱해, 원본 텐서에서 데이터를 읽어올 정확한 1차원 메모리 주소를 누적해 나갑니다.
    in_idx = in_idx + coord * in_stride;
  }
  
  // 변수 output 배열 쓰기
  // 역할: 매핑된 입력 텐서의 1차원 인덱스 위치(in_idx)에 있는 데이터를 읽어와 출력 텐서 위치(out_idx)에 복사하여 위치 바꿈(permute)을 완료합니다.
  output[out_idx] = input[in_idx];
}
`;

```

---

## `packages/forge/src/tensor/kernels/relu.wgsl.ts`

```typescript
/**
 * 생성일: 2026-08-12T12:14:52+09:00
 * 수정 이력:
 * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const RELU_WGSL = `
// 구조체: Params
// 역할 (WHAT): ReLU(Rectified Linear Unit) 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼(uniform) 데이터를 효율적으로 전달하고 메모리 정렬을 맞추기 위해 사용됩니다.
// 동작 방식 (HOW): 각 스레드가 처리할 전체 요소 개수와 2D 워크그룹 할당 정보를 메모리에서 읽어옵니다.
struct Params {
  // 변수: size
  // 역할: 입력 텐서가 가진 총 데이터 요소의 수
  size: u32,
  // 변수: workgroups_x
  // 역할: X축 방향으로 할당된 작업 그룹(workgroup)의 개수
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 16바이트 정렬을 맞추기 위한 패딩 변수
  pad2: u32,
  pad3: u32,
};

// 변수: params
// 역할: Params 구조체 값을 담고 있는 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 역할: ReLU 활성화 함수가 적용될 원본 데이터를 가진 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 역할: ReLU 연산 결과가 기록될 읽기/쓰기 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 역할 (WHAT): 입력 텐서의 각 요소에 대해 ReLU 활성화 함수(max(0, x))를 적용합니다.
// 목적 (WHY): 딥러닝 모델의 비선형성을 부여하기 위한 ReLU 연산을 GPU에서 병렬로 고속 처리하기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 전역 ID를 활용하여 자신의 1D 인덱스를 계산한 후, 해당 인덱스에 있는 x의 값과 0 중 더 큰 값을 y에 저장합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 역할: 처리할 배열의 전체 요소 수를 유니폼 버퍼로부터 가져옵니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 글로벌 ID의 2D 인덱스를 1D 인덱스로 변환하기 위해 X축 워크그룹 수를 가져옵니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: idx
  // 역할: 현재 스레드가 처리해야 하는 1차원 데이터의 절대 인덱스
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: 배열 크기 검사
  // 역할: 계산된 인덱스가 실제 데이터 범위(num_elements)를 벗어나는 경우 쓰레드 실행을 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 변수 y 갱신
  // 역할: x 배열의 해당 위치 값을 0.0과 비교해 큰 값(음수는 0, 양수는 그대로)을 y 배열에 저장합니다.
  y[idx] = max(x[idx], 0.0);
}
`;

```

---

## `packages/forge/src/tensor/kernels/relu_backward.wgsl.ts`

```typescript
/**
 * 생성일: 2026-08-12T12:14:52+09:00
 * 수정 이력:
 * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const RELU_BACKWARD_WGSL = `
// 구조체: Params
// 역할 (WHAT): ReLU 역전파 연산에 필요한 메타데이터 정보를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼 데이터를 전달하여 전체 연산 크기 등을 파악하게 합니다.
// 동작 방식 (HOW): 전체 크기(size)와 2D 그리드 변환을 위한 workgroups_x 인자를 넘겨줍니다.
struct Params {
  // 변수: size
  // 역할: 처리할 데이터의 전체 요소 수
  size: u32,
  // 변수: workgroups_x
  // 역할: x축 워크그룹의 총 개수
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 16바이트 메모리 정렬을 위한 패딩
  pad2: u32,
  pad3: u32,
}

// 변수: params
// 역할: 연산에 필요한 메타데이터가 담긴 유니폼 버퍼
@group(0) @binding(0) var<uniform> params : Params;

// 변수: X
// 역할: 순전파 시 입력되었던 원본 데이터를 담은 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> X : array<f32>;

// 변수: gradOutput
// 역할: 이전 레이어에서 흘러들어온 그래디언트(Gradient) 값을 담은 읽기 전용 버퍼
@group(0) @binding(2) var<storage, read> gradOutput : array<f32>;

// 변수: gradInput
// 역할: ReLU 연산의 역전파 결과로 계산된 그래디언트를 저장할 읽기/쓰기 버퍼
@group(0) @binding(3) var<storage, read_write> gradInput : array<f32>;

// 함수: main
// 역할 (WHAT): ReLU 역전파 그래디언트를 계산하는 컴퓨트 셰이더 메인 함수입니다.
// 목적 (WHY): 역전파 과정에서 X > 0 인 위치에만 그래디언트를 통과시키기 위해 존재합니다.
// 동작 방식 (HOW): 각 스레드는 1D 인덱스를 계산하고, X[index]의 값이 양수일 경우 gradOutput을 그대로 gradInput에 복사하고, 0 이하일 경우 0.0을 저장합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  // 변수: num_elements
  // 역할: 처리할 배열 요소의 총 개수를 저장합니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 2D 인덱스를 1D 인덱스로 풀기 위해 가로 워크그룹 크기를 저장합니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: index
  // 역할: 현재 스레드의 작업을 가리키는 1차원 배열 위치 인덱스
  let index = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: 데이터 경계 확인
  // 역할: 유효한 데이터 인덱스(num_elements 내부)인 경우에만 연산을 수행합니다.
  if (index < num_elements) {
    // 조건문: ReLU 미분 조건 (X > 0)
    // 역할: 원본 입력 값(X)이 양수인지 판단합니다.
    if (X[index] > 0.0) {
      // 변수 gradInput 갱신 (통과)
      // 역할: X가 양수였으므로 미분값이 1이 되어, 들어온 그래디언트를 그대로 전달합니다.
      gradInput[index] = gradOutput[index];
    } else {
      // 변수 gradInput 갱신 (차단)
      // 역할: X가 0 이하이므로 미분값이 0이 되어, 그래디언트 흐름을 0으로 차단합니다.
      gradInput[index] = 0.0;
    }
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/scatter.wgsl.ts`

```typescript
/**
 * 생성일: 2026-08-12T12:23:09+09:00
 * 수정 이력:
 * - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 */
export const SCATTER_WGSL = `
// 구조체: Params
// 역할 (WHAT): 스캐터(Scatter) 연산에 필요한 차원, 보폭 및 타겟 축 정보를 저장하는 구조체입니다.
// 목적 (WHY): 입력 데이터 텐서와 인덱스 텐서를 조합하여 출력 텐서의 어느 위치에 값을 기록할지 결정하기 위함입니다.
// 동작 방식 (HOW): 각 차원에 대한 크기(rank), 흩뿌릴 차원(dim), 인덱스/입력의 보폭 정보를 읽어와 좌표를 계산합니다.
struct Params {
  // 변수: num_elements
  // 역할: 처리할 입력 요소들의 전체 개수
  num_elements: u32,
  // 변수: dim
  // 역할: 인덱스 값으로 대체되어 흩뿌려질 대상 차원 축
  dim: u32,
  // 변수: rank
  // 역할: 텐서가 갖는 전체 차원 수
  rank: u32,
  // 변수: workgroups_x
  // 역할: 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수
  workgroups_x: u32,
  // 변수: x_strides
  // 역할: 출력 배열(입력과 동일한 형상을 가지는 베이스)의 각 차원별 메모리 보폭 배열
  x_strides: array<u32, 8>,
  // 변수: idx_strides
  // 역할: 인덱스 텐서의 각 차원별 메모리 보폭 배열
  idx_strides: array<u32, 8>,
  // 변수: x_shape
  // 역할: 출력 텐서의 각 차원별 크기 배열 (인덱스 바운드 검사용)
  x_shape: array<u32, 8>,
};

// 변수: params
// 역할: 스캐터 연산의 메타데이터를 담은 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: index
// 역할: 흩뿌릴 위치 정보를 가지고 있는 인덱스 배열(읽기 전용 스토리지 버퍼)
@group(0) @binding(1) var<storage, read> index: array<f32>;

// 변수: src
// 역할: 출력 배열에 복사할 원본 값을 가지고 있는 소스 배열(읽기 전용 스토리지 버퍼)
@group(0) @binding(2) var<storage, read> src: array<f32>;

// 변수: output
// 역할: 원본 값들이 인덱스 배열의 지시에 따라 흩뿌려진 최종 결과물이 저장될 스토리지 버퍼
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 함수: main
// 역할 (WHAT): 주어진 인덱스 텐서의 값에 따라 소스 데이터를 출력 텐서의 특정 위치에 저장합니다.
// 목적 (WHY): 특정 차원의 값을 인덱스로 치환하여(Scatter-Elements) 텐서 내 원하는 위치에 데이터를 쓰기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 1차원 ID를 다차원 좌표로 변환하고, 지정된 축(dim)에 대해서만 원래 좌표 대신 인덱스 텐서의 값을 좌표로 사용하여 출력 위치를 정합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: idx
  // 역할: 소스 데이터와 인덱스 데이터의 1차원 메모리 인덱스
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  
  // 조건문: 데이터 경계 검사
  // 역할: 할당된 스레드의 인덱스가 전체 크기(num_elements)를 초과하는지 검사합니다.
  if (idx >= params.num_elements) { return; }

  // 변수: temp
  // 역할: 다차원 좌표로 분리해 나가기 위해 남은 인덱스 수치를 보관하는 임시 변수
  var temp = idx;
  // 변수: out_idx
  // 역할: 최종적으로 계산된 출력 배열의 1차원 메모리 인덱스를 누적할 변수
  var out_idx = 0u;

  // 반복문: for 루프 (모든 차원 순회)
  // 역할 (WHAT): 최상위 차원부터 0번째 차원까지 각 차원의 좌표를 구하고, 이를 이용해 출력 인덱스를 계산합니다.
  // 목적 (WHY): 1차원 인덱스를 다시 N차원 좌표로 풀고, 특정 차원(dim)에 대해서만 값을 교체하기 위해 필요합니다.
  // 동작 방식 (HOW): i가 dim과 같을 경우, 계산된 논리적 좌표 대신 index 배열에 있는 값을 가져와서 보폭을 곱하고, 그 외의 경우 원래 좌표에 보폭을 곱합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    // 변수: coord
    // 역할: 현재 차원(i)에 해당하는 인덱스 텐서 기준의 다차원 논리 좌표
    let coord = temp / params.idx_strides[i];
    
    // 변수: temp 갱신
    // 역할: 다음 차원 계산을 위해 나머지 값을 임시 변수에 업데이트합니다.
    temp = temp % params.idx_strides[i];
    
    // 조건문: 타겟 차원(dim) 여부 검사
    // 역할: 현재 처리 중인 차원이 인덱스 값으로 대체할 타겟 차원인지 판단합니다.
    if (i == params.dim) {
      let raw_idx = index[idx];
      let dim_size = i32(params.x_shape[i]);
      var signed_idx = i32(raw_idx);
      if (signed_idx < 0) {
        signed_idx = signed_idx + dim_size;
      }
      // OOB or NaN check: skip execution if index is out of bounds or NaN to prevent data corruption
      if (signed_idx < 0 || signed_idx >= dim_size || raw_idx != raw_idx) {
        return;
      }
      let valid_idx = u32(signed_idx);
      out_idx = out_idx + valid_idx * params.x_strides[i];
    } else {
      // 변수: out_idx 누적 (일반 축)
      // 역할: 원래 논리적 좌표(coord)에 출력 보폭(x_strides)을 곱해 더합니다.
      out_idx = out_idx + coord * params.x_strides[i];
    }
  }

  // 주석: 엄밀한 원자성(Atomic)은 제공하지 않지만 인덱스가 겹치지 않는 단순 스캐터의 경우 정상 동작함.
  // 변수 output 갱신
  // 역할: 치환이 완료되어 도출된 출력 인덱스 위치(out_idx)에 소스 배열의 데이터(src[idx])를 저장합니다.
  // Not strictly atomic, but for simple scatter where indices are unique it's fine.
  output[out_idx] = src[idx];
}
`;

```

---

## `packages/forge/src/tensor/kernels/sigmoid.wgsl.ts`

```typescript
/**
 * 생성일: 2026-08-12T12:14:52+09:00
 * 수정 이력:
 * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const SIGMOID_WGSL = `
// 구조체: Params
// 역할 (WHAT): 시그모이드(Sigmoid) 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼 인자를 넘겨주어 전체 요소 수 등의 전역 설정을 공유하기 위함입니다.
// 동작 방식 (HOW): 요소 크기와 2D 워크그룹 할당 정보를 메모리 정렬을 맞추어 전달합니다.
struct Params {
  // 변수: size
  // 역할: 처리 대상 배열이 가진 전체 원소의 개수
  size: u32,
  // 변수: workgroups_x
  // 역할: X축 방향의 워크그룹 개수
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 16바이트 메모리 정렬(alignment)용 패딩
  pad2: u32,
  pad3: u32,
};

// 변수: params
// 역할: Params 구조체를 담고 있는 유니폼 버퍼 (바인딩 0)
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 역할: 시그모이드 활성화 함수가 적용될 원본 데이터가 저장된 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 역할: 시그모이드 연산 결과가 기록될 읽기/쓰기 가능한 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 역할 (WHAT): 입력 텐서의 각 요소에 대해 시그모이드 활성화 함수(1 / (1 + exp(-x)))를 적용합니다.
// 목적 (WHY): 신경망의 값을 0과 1 사이로 변환하는 활성화 함수 연산을 GPU에서 병렬로 고속 수행하기 위함입니다.
// 동작 방식 (HOW): 64크기의 워크그룹 내 스레드들이 1D 인덱스를 계산하고, 범위를 초과하지 않으면 수학 함수 exp를 이용해 시그모이드 수식을 계산 후 y 배열에 씁니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 역할: 유니폼 버퍼에서 배열의 총 원소 개수를 읽어옵니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 2차원 워크그룹 인덱스를 1차원 인덱스로 변환하기 위해 X축 워크그룹 수를 읽어옵니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: idx
  // 역할: 현재 쓰레드가 처리해야 할 1차원 데이터 인덱스
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: 배열 경계 확인
  // 역할: 인덱스가 실제 배열의 범위를 벗어날 경우 셰이더 실행을 조기 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 변수 y 갱신
  // 역할: x 배열의 값에 시그모이드 공식을 적용한 결과를 y 배열에 저장합니다.
  y[idx] = 1.0 / (1.0 + exp(-x[idx]));
}
`;

```

---

## `packages/forge/src/tensor/kernels/sigmoid_backward.wgsl.ts`

```typescript
/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const SIGMOID_BACKWARD_WGSL = `
// 구조체: Params
// 목적: WGSL 커널에서 사용할 유니폼 파라미터들을 정의합니다. 메모리 정렬을 위해 패딩 변수가 포함되어 있습니다.
// 작동 방식: size와 workgroups_x 정보를 포함하여 작업 스레드가 자신의 위치를 파악할 수 있게 합니다.
struct Params {
  // 변수: size
  // 목적: 처리해야 할 전체 요소의 총 개수를 저장합니다.
  // 작동 방식: 배열의 범위를 초과하는 접근을 방지하는 기준값으로 사용됩니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 방향의 워크그룹 개수를 저장합니다.
  // 작동 방식: 2차원 워크그룹 인덱스를 1차원 전역 인덱스로 변환할 때 곱해지는 계수로 사용됩니다.
  workgroups_x: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: GPU 메모리 접근 성능을 최적화하고 데이터 구조의 규격을 맞추는 역할을 합니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: GPU 메모리 접근 성능을 최적화하고 데이터 구조의 규격을 맞추는 역할을 합니다.
  pad3: u32,
};

// 변수: params
// 목적: 외부에서 전달되는 설정값들을 저장하는 유니폼 버퍼(Uniform buffer) 변수입니다.
// 작동 방식: 바인딩 그룹 0, 바인딩 0에 매핑되어 워크그룹 실행 시 필요한 메타데이터를 제공합니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: grad
// 목적: 역전파(Backpropagation) 단계에서 이전 층(layer)으로부터 전달받은 손실(loss)의 기울기(gradient)를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 0, 바인딩 1에 매핑되며, 최종 기울기를 계산할 때 곱해지는 입력값으로 쓰입니다.
@group(0) @binding(1) var<storage, read> grad: array<f32>;

// 변수: sigmoid_output
// 목적: 순전파(Forward propagation) 단계에서 미리 계산되었던 Sigmoid 함수의 출력 결과를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 0, 바인딩 2에 매핑되며, Sigmoid 미분 공식을 적용하기 위한 상태값으로 사용됩니다.
@group(0) @binding(2) var<storage, read> sigmoid_output: array<f32>;

// 변수: output
// 목적: 계산된 Sigmoid 함수의 역전파 기울기 결과를 저장할 읽기/쓰기 가능 버퍼입니다.
// 작동 방식: 바인딩 0, 바인딩 3에 매핑되며, 각 스레드에서 계산된 최종 미분값이 이곳에 기록됩니다.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 함수: main
// 목적: Sigmoid 함수의 역전파(Backward) 연산을 병렬로 수행하는 메인 컴퓨트 셰이더(Compute Shader) 진입점입니다.
// 작동 방식: Sigmoid 미분 공식인 'sigmoid_output * (1 - sigmoid_output)'을 사용하여 이전 기울기 'grad'와 곱한 뒤 최종 기울기를 계산합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 연산해야 할 총 원소의 개수를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 size 필드를 읽어와 저장합니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 워크그룹의 크기를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 workgroups_x 필드를 읽어와 저장합니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 실행 중인 스레드가 담당할 1차원 데이터 인덱스를 계산합니다.
  // 작동 방식: 3차원인 global_id 값을 바탕으로, Y축 인덱스에 (X축 워크그룹 개수 * 워크그룹 크기 64)를 곱하고 X축 인덱스를 더해 평면화(flatten)된 인덱스를 구합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 유효한 데이터 범위를 벗어난 스레드가 실행되는 것을 방지합니다.
  // 작동 방식: 계산된 인덱스(idx)가 처리해야 할 전체 요소 수(num_elements) 이상인지 확인합니다.
  if (idx >= num_elements) {
    // 유효 범위를 초과하면 아무 연산도 수행하지 않고 함수를 종료합니다.
    return;
  }

  // 연산: output[idx] 갱신
  // 목적: 최종적으로 입력 노드에 전달할 기울기(Gradient) 값을 도출하여 저장합니다.
  // 작동 방식: 체인 룰(Chain rule)에 의해 '상류에서 온 기울기(grad[idx])' * '로컬 미분값(sigmoid_output[idx] * (1.0 - sigmoid_output[idx]))'을 연산한 후 배열에 기록합니다.
  output[idx] = grad[idx] * sigmoid_output[idx] * (1.0 - sigmoid_output[idx]);
}
`;

```

---

## `packages/forge/src/tensor/kernels/sub.wgsl.ts`

```typescript
/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *   - 2026-08-18 14:10:00 +0900: Feat: Full 8D Multi-Dimensional Stride Broadcasting Decoder
 */
export const SUB_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  rank: u32,
  pad0: u32,
  dim0: u32, dim1: u32, dim2: u32, dim3: u32,
  dim4: u32, dim5: u32, dim6: u32, dim7: u32,
  stride_a0: u32, stride_a1: u32, stride_a2: u32, stride_a3: u32,
  stride_a4: u32, stride_a5: u32, stride_a6: u32, stride_a7: u32,
  stride_b0: u32, stride_b1: u32, stride_b2: u32, stride_b3: u32,
  stride_b4: u32, stride_b5: u32, stride_b6: u32, stride_b7: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    var temp = idx;
    let c7 = temp % params.dim7; temp = temp / params.dim7;
    let c6 = temp % params.dim6; temp = temp / params.dim6;
    let c5 = temp % params.dim5; temp = temp / params.dim5;
    let c4 = temp % params.dim4; temp = temp / params.dim4;
    let c3 = temp % params.dim3; temp = temp / params.dim3;
    let c2 = temp % params.dim2; temp = temp / params.dim2;
    let c1 = temp % params.dim1; temp = temp / params.dim1;
    let c0 = temp;

    let idx_a = c0 * params.stride_a0 + c1 * params.stride_a1 + c2 * params.stride_a2 + c3 * params.stride_a3 +
                c4 * params.stride_a4 + c5 * params.stride_a5 + c6 * params.stride_a6 + c7 * params.stride_a7;
    let idx_b = c0 * params.stride_b0 + c1 * params.stride_b1 + c2 * params.stride_b2 + c3 * params.stride_b3 +
                c4 * params.stride_b4 + c5 * params.stride_b5 + c6 * params.stride_b6 + c7 * params.stride_b7;

    out[idx] = a[idx_a] - b[idx_b];
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/sum.wgsl.ts`

```typescript
/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * - 2026-08-18 00:30:00: Fix(SCRUM-157/VULN-05): 2D workgroup linear index reconstruction for >65535 reductions
 */
export const SUM_WGSL = `
// 구조체: Params
// 목적: 합계(Sum) 연산에 필요한 메타데이터와 패딩을 정의합니다.
// 작동 방식: 전체 원소 개수와 2D 디스패치 분할을 위한 X축 워크그룹 수를 제공하며 16바이트 정렬을 준수합니다.
struct Params {
  // 변수: numElements
  // 목적: 더해야 할 입력 배열의 전체 원소 개수를 나타냅니다.
  // 작동 방식: 전역 인덱스가 유효 범위를 벗어나는지 검사하는 용도로 사용됩니다.
  numElements: u32,
  // 변수: workgroups_x
  // 목적: 65,535 초과 시 2D 그리드로 분할된 X축 워크그룹 개수입니다.
  // 작동 방식: workgroup_id.y * workgroups_x + workgroup_id.x 수식을 통해 1D 선형 워크그룹 인덱스를 복원합니다.
  workgroups_x: u32,
  // 변수: pad1
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  pad1: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  pad2: u32,
};

// 변수: params
// 목적: 유니폼 버퍼를 통해 워크그룹 외부에서 메타데이터를 주입받습니다.
// 작동 방식: 바인딩 0에 할당되어 전체 요소 개수와 그리드 크기를 모든 스레드에 제공합니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 목적: 합계를 구할 대상이 되는 데이터를 담은 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 1에 할당되며, 각 스레드가 자신의 위치에 해당하는 값을 읽어옵니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 목적: 각 워크그룹 내에서의 부분 합계(Partial sum)를 저장할 버퍼입니다.
// 작동 방식: 바인딩 2에 할당되며, 최종적으로 워크그룹 개수만큼의 결과가 저장됩니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 변수: s_data
// 목적: 워크그룹 내 스레드들이 공유하는 로컬 메모리(Shared memory)입니다.
// 작동 방식: 256 크기의 배열로 할당되어 빠른 Reduction(축소) 연산을 위한 캐시 역할을 합니다.
var<workgroup> s_data: array<f32, 256>;

// 함수: main
// 목적: 배열 요소들의 총합을 구하기 위한 병렬 Reduction(축소) 알고리즘을 수행합니다.
// 작동 방식: 2D 워크그룹 좌표에서 선형 인덱스를 복원하고 공유 메모리를 사용하여 트리(Tree) 구조로 단계별 덧셈을 수행합니다.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  // 변수: wg_linear
  // 목적: 2D 그리드로 분할된 워크그룹의 고유 1차원 선형 인덱스를 복원합니다.
  // 작동 방식: workgroup_id.y * params.workgroups_x + workgroup_id.x
  let wg_linear = workgroup_id.y * params.workgroups_x + workgroup_id.x;

  // 변수: gid
  // 목적: 전체 스레드 중 현재 스레드의 고유 1차원 전역 인덱스입니다.
  // 작동 방식: 선형 워크그룹 번호와 로컬 ID를 조합하여 계산합니다.
  let gid = wg_linear * 256u + local_id.x;

  // 변수: lid
  // 목적: 현재 워크그룹 내에서의 로컬 인덱스(0~255)입니다.
  // 작동 방식: local_id.x 값을 가져와 공유 메모리 접근 및 축소 연산 인덱스로 사용합니다.
  let lid = local_id.x;

  // 변수: wid
  // 목적: 부분 합 결과를 저장할 출력 버퍼 위치 인덱스입니다.
  let wid = wg_linear;
  
  // 제어문: if-else
  // 목적: 입력 데이터를 로컬 공유 메모리에 복사하면서, 범위를 벗어난 공간을 0으로 초기화합니다.
  // 작동 방식: gid가 유효한 원소 범위 안에 있으면 input[gid]를, 벗어나면 0.0을 s_data에 할당합니다.
  if (gid < params.numElements) {
    s_data[lid] = input[gid];
  } else {
    s_data[lid] = 0.0;
  }
  
  // 동기화: workgroupBarrier()
  // 목적: 워크그룹 내의 모든 스레드가 공유 메모리에 데이터를 쓸 때까지 대기합니다.
  // 작동 방식: 초기 데이터 적재(Load)가 완전히 끝난 뒤에만 다음 Reduction 단계를 진행하도록 보장합니다.
  workgroupBarrier();
  
  // 반복문: for
  // 목적: 워크그룹 내 256개의 요소를 트리 기반 병렬 Reduction 방식으로 더합니다.
  // 작동 방식: s 변수를 128부터 시작하여 0보다 클 때까지 절반으로 줄여가며 (비트 시프트 연산) 부분 합을 구합니다.
  for (var s = 128u; s > 0u; s >>= 1u) {
    // 제어문: if
    // 목적: 유효한 절반의 스레드들만 덧셈 연산에 참여하도록 제한합니다.
    // 작동 방식: 로컬 인덱스가 현재 단계의 s보다 작을 경우에만 s_data[lid]와 s_data[lid + s]를 더합니다.
    if (lid < s) {
      s_data[lid] = s_data[lid] + s_data[lid + s];
    }
    // 동기화: workgroupBarrier()
    // 목적: 다음 단계의 Reduction으로 넘어가기 전, 현재 단계의 덧셈이 모든 스레드에서 완료되었는지 확인합니다.
    // 작동 방식: 모든 스레드가 동기화 지점에 도달할 때까지 실행을 일시 중단합니다.
    workgroupBarrier();
  }
  
  // 제어문: if
  // 목적: 워크그룹 내 최종 합산 결과(s_data[0])를 전역 출력 버퍼에 단 한 번만 기록합니다.
  // 작동 방식: 로컬 인덱스가 0번인 스레드만 대표로 output[wid]에 s_data[0] 값을 할당합니다.
  if (lid == 0u) {
    output[wid] = s_data[0];
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/sum_axis.wgsl.ts`

```typescript
/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * - 2026-08-18 00:30:00 +0900: Fix(SCRUM-154/VULN-02): Generic 3-parameter reduction for 3D/4D tensors
 */
export const SUM_AXIS_WGSL = `
/**
 * 이 구조체(Params)는 임의 축에 대한 텐서 축소(Sum Along Axis) 연산에 필요한 메타데이터를 담고 있습니다.
 * 3차원 이상의 고차원 텐서에서도 일반화된 (outer_size, reduction_size, inner_stride) 3-파라미터 체계를 지원합니다.
 */
struct Params {
  outer_size: u32,     // 축소 축 이전의 외부 배치/차원들의 곱
  reduction_size: u32, // 축소할 대상 축의 원소 개수 (Reduction Dimension Size)
  inner_stride: u32,   // 축소 축 이후의 내부 차원들의 스트라이드 곱
  output_numel: u32,   // 결과 텐서의 총 원소 개수 (outer_size * inner_stride)
  workgroups_x: u32,   // 2D 디스패치 분할을 위한 X축 워크그룹 수
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 전달되는 축소 메타데이터 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // 축소 연산을 수행할 원본 입력 텐서입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 축소된 결과가 저장될 출력 텐서입니다.

/**
 * main 함수는 출력 텐서의 각 원소에 대해 입력 텐서의 reduction_size개 원소들을 순회하며 합산합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let workgroups_x = params.workgroups_x;
  let out_idx = global_id.x + global_id.y * workgroups_x * 64u;

  if (out_idx >= params.output_numel) {
    return;
  }

  let inner_stride = params.inner_stride;
  let reduction_size = params.reduction_size;
  let outer_idx = out_idx / inner_stride;
  let inner_idx = out_idx % inner_stride;
  let slice_stride = reduction_size * inner_stride;
  let base_offset = outer_idx * slice_stride + inner_idx;

  var sum = 0.0;
  for (var r = 0u; r < reduction_size; r = r + 1u) {
    sum += input[base_offset + r * inner_stride];
  }
  output[out_idx] = sum;
}
`;

```

---

## `packages/forge/src/tensor/kernels/tanh.wgsl.ts`

```typescript
/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const TANH_WGSL = `
// 구조체: Params
// 목적: Tanh 커널 연산 시 필요한 설정값을 제공합니다.
// 작동 방식: 배열의 전체 요소 수와 워크그룹의 X축 크기를 포함합니다.
struct Params {
  // 변수: size
  // 목적: 전체 연산 대상 원소의 개수를 저장합니다.
  // 작동 방식: 범위를 초과하는 메모리 접근을 방지하는 기준으로 쓰입니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 워크그룹 수를 저장합니다.
  // 작동 방식: 3차원 스레드 ID를 1차원 인덱스로 변환할 때 곱해집니다.
  workgroups_x: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 유니폼 구조체의 메모리 오프셋 규칙을 준수합니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 유니폼 구조체의 메모리 오프셋 규칙을 준수합니다.
  pad3: u32,
};

// 변수: params
// 목적: 커널의 설정값을 가지고 있는 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0을 통해 GPU에 전달되어 읽기 전용으로 참조됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 목적: Tanh 함수를 적용할 입력 텐서 데이터를 담고 있는 버퍼입니다.
// 작동 방식: 바인딩 1에 매핑되며 원본 데이터를 제공합니다.
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 목적: Tanh 함수의 계산 결과를 저장할 출력 버퍼입니다.
// 작동 방식: 바인딩 2에 매핑되며 계산된 활성화 값이 각 인덱스에 저장됩니다.
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 목적: 병렬 스레드를 이용하여 배열의 각 요소에 대해 Tanh(쌍곡탄젠트) 함수를 계산합니다.
// 작동 방식: 내장 함수인 tanh()를 호출하여 y 배열에 저장합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 배열의 전체 요소 개수입니다.
  // 작동 방식: 유니폼 버퍼에서 size를 읽어옵니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 워크그룹의 크기입니다.
  // 작동 방식: 유니폼 버퍼에서 workgroups_x를 읽어옵니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 스레드가 처리할 1차원 배열의 인덱스입니다.
  // 작동 방식: global_id 정보를 바탕으로 평면화된 인덱스를 계산합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 유효한 데이터 인덱스 범위를 초과한 스레드가 실행되는 것을 막습니다.
  // 작동 방식: idx가 num_elements 이상일 경우 함수를 조기 종료(return)합니다.
  if (idx >= num_elements) {
    return;
  }

  // 연산: y[idx] 기록
  // 목적: 특정 요소의 Tanh 값을 계산하여 저장합니다.
  // 작동 방식: WGSL 내장 함수 tanh(x[idx])를 호출하고 그 결과를 y[idx]에 씁니다.
  y[idx] = tanh(x[idx]);
}
`;

```

---

## `packages/forge/src/tensor/kernels/tanh_backward.wgsl.ts`

```typescript
/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const TANH_BACKWARD_WGSL = `
// 구조체: Params
// 목적: WGSL 커널에서 사용할 유니폼 파라미터들을 정의합니다. 메모리 정렬을 위해 패딩 변수가 포함되어 있습니다.
// 작동 방식: 배열 크기(size)와 2차원 워크그룹의 X축 크기(workgroups_x)를 제공합니다.
struct Params {
  // 변수: size
  // 목적: 연산할 전체 배열 요소의 개수입니다.
  // 작동 방식: 배열 범위를 초과하는 인덱스 접근을 차단하기 위한 경계값으로 쓰입니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 방향 워크그룹의 개수입니다.
  // 작동 방식: 3D 워크그룹 인덱스를 1D 전역 인덱스로 변환할 때 필요합니다.
  workgroups_x: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트의 배수로 맞춰 GPU 메모리 접근 오류를 방지합니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트의 배수로 맞춰 GPU 메모리 접근 오류를 방지합니다.
  pad3: u32,
};

// 변수: params
// 목적: 커널의 설정값을 가지고 있는 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0을 통해 GPU에 전달되어 읽기 전용으로 참조됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: grad
// 목적: 역전파 시 이전 층(상류)으로부터 전달받은 손실 기울기를 저장하는 버퍼입니다.
// 작동 방식: 바인딩 1에 매핑되어 최종 미분값 계산에 곱해지는 입력값으로 쓰입니다.
@group(0) @binding(1) var<storage, read> grad: array<f32>;

// 변수: tanh_output
// 목적: 순전파 단계에서 이미 계산되었던 Tanh 함수의 출력 결과를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 2에 매핑되며, Tanh 미분 공식을 적용하기 위한 상태값으로 사용됩니다.
@group(0) @binding(2) var<storage, read> tanh_output: array<f32>;

// 변수: output
// 목적: 계산된 Tanh 함수의 역전파 기울기 결과를 저장할 읽기/쓰기 가능 버퍼입니다.
// 작동 방식: 바인딩 3에 매핑되며, 각 스레드에서 계산된 최종 미분값이 이곳에 기록됩니다.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 함수: main
// 목적: Tanh 함수의 역전파(Backward) 연산을 병렬로 수행하는 메인 컴퓨트 셰이더 진입점입니다.
// 작동 방식: Tanh 미분 공식 '1 - tanh_output^2'에 이전 기울기 'grad'를 곱하여 최종 기울기를 계산합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 연산해야 할 총 원소의 개수를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 size 필드를 읽어옵니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 워크그룹의 크기를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 workgroups_x 필드를 읽어옵니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 실행 중인 스레드가 담당할 1차원 데이터 인덱스를 계산합니다.
  // 작동 방식: 3차원인 global_id 값을 바탕으로, Y축 인덱스에 (X축 워크그룹 개수 * 64)를 곱하고 X축 인덱스를 더해 평면화합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 유효한 데이터 범위를 벗어난 스레드가 실행되는 것을 방지합니다.
  // 작동 방식: idx가 num_elements 이상인지 확인하여 맞으면 함수를 종료합니다.
  if (idx >= num_elements) {
    return;
  }

  // 연산: output[idx] 갱신
  // 목적: 최종적으로 입력 노드에 전달할 기울기(Gradient) 값을 도출하여 저장합니다.
  // 작동 방식: 체인 룰(Chain rule)에 의해 '상류에서 온 기울기(grad[idx])' * '로컬 미분값(1.0 - tanh_output[idx] * tanh_output[idx])'을 계산해 기록합니다.
  output[idx] = grad[idx] * (1.0 - tanh_output[idx] * tanh_output[idx]);
}
`;

```

---

## `packages/forge/src/tensor/kernels/transpose.wgsl.ts`

```typescript
/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:59:35: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization) (67c4ce9901dbb7caf2710e9ad03514f48956cfa6)
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const TRANSPOSE_WGSL = `
// 구조체: Params
// 목적: 행렬 전치(Transpose) 연산에 필요한 차원 정보를 전달합니다.
// 작동 방식: 행(M), 열(N), 배치(B) 크기를 받아 다차원 배열의 인덱스를 계산할 수 있게 합니다.
struct Params {
  // 변수: M
  // 목적: 변환 전 원본 행렬의 행(Row) 개수입니다.
  // 작동 방식: 전치 후에는 이 값이 열의 개수가 됩니다.
  M: u32,
  // 변수: N
  // 목적: 변환 전 원본 행렬의 열(Column) 개수입니다.
  // 작동 방식: 전치 후에는 이 값이 행의 개수가 됩니다.
  N: u32,
  // 변수: B
  // 목적: 배치(Batch) 크기를 의미합니다.
  // 작동 방식: 여러 개의 독립적인 행렬(배치)을 동시에 전치할 수 있게 합니다.
  B: u32,
};

// 변수: params
// 목적: 셰이더 실행 시 필요한 차원(M, N, B) 정보를 담은 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0에 매핑되어 인덱스 계산의 기준값으로 사용됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 목적: 전치하기 전의 원본 다차원 배열(배치 포함 3차원 구조) 데이터입니다.
// 작동 방식: 바인딩 1에 읽기 전용 스토리지 버퍼로 바인딩됩니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: out
// 목적: 행과 열이 뒤바뀐 전치 결과를 저장할 출력 버퍼입니다.
// 작동 방식: 바인딩 2에 할당되어 계산된 데이터가 저장됩니다.
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// 함수: main
// 목적: 배치 차원을 유지한 채로 행(Row)과 열(Column)의 위치를 바꾸는 전치 연산을 수행합니다.
// 작동 방식: 3차원 글로벌 인덱스(x, y, z)를 각각 (row, col, batch)로 매핑하고 변환 공식을 적용합니다.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: row
  // 목적: 원본 행렬 기준에서의 행 인덱스를 나타냅니다.
  // 작동 방식: 3D 스레드 ID의 x 성분(global_id.x)을 사용합니다.
  let row = global_id.x;

  // 변수: col
  // 목적: 원본 행렬 기준에서의 열 인덱스를 나타냅니다.
  // 작동 방식: 3D 스레드 ID의 y 성분(global_id.y)을 사용합니다.
  let col = global_id.y;

  // 변수: batch
  // 목적: 현재 처리 중인 배치의 인덱스를 나타냅니다.
  // 작동 방식: 3D 스레드 ID의 z 성분(global_id.z)을 사용합니다.
  let batch = global_id.z;
  
  // 제어문: if
  // 목적: 패딩이나 워크그룹 크기 맞춤으로 인해 실제 데이터 범위를 초과한 스레드가 실행되는 것을 방지합니다.
  // 작동 방식: row, col, batch가 각각 M, N, B보다 작은지 확인합니다.
  if (row < params.M && col < params.N && batch < params.B) {
    // 변수: in_idx
    // 목적: 1차원으로 평면화된 원본 배열에서 읽어올 요소의 인덱스를 계산합니다.
    // 작동 방식: '배치 오프셋 + 행 오프셋 + 열' (batch * M * N + row * N + col) 공식을 사용합니다.
    let in_idx = batch * (params.M * params.N) + row * params.N + col;

    // 변수: out_idx
    // 목적: 전치된 결과를 저장할 출력 배열의 1차원 평면화 인덱스를 계산합니다.
    // 작동 방식: 행과 열의 기준 크기가 바뀌므로 '배치 오프셋 + 새로운 행 오프셋 + 새로운 열' (batch * M * N + col * M + row)로 계산합니다.
    let out_idx = batch * (params.M * params.N) + col * params.M + row;

    // 연산: out[out_idx] 할당
    // 목적: 계산된 위치에 원본 데이터를 복사하여 전치를 완료합니다.
    // 작동 방식: 원본 위치(in_idx)의 값을 읽어 목표 위치(out_idx)에 기록합니다.
    out[out_idx] = input[in_idx];
  }
}
`;

```

---

## `packages/forge/src/tensor/kernels/where.wgsl.ts`

```typescript
/**
 * 파일 생성: 2026-08-12 12:23:09
 * 수정 내역:
 * - 2026-08-12 12:23:09: Docs: Build Apache-style docs and unify tests (fc28607f9d46845175a9bdaf0e9e8c44bace5ecb)
 */
export const WHERE_WGSL = `
// 구조체: Params
// 목적: 조건부 분기(Where) 연산에 사용되는 메타데이터를 저장합니다.
// 작동 방식: 처리할 전체 요소 크기(size)와 패딩 값들을 통해 16바이트 정렬된 메모리 구조를 형성합니다.
struct Params {
  // 변수: size
  // 목적: 배열의 전체 요소 수를 지정합니다.
  // 작동 방식: 커널에서 각 스레드가 유효한 범위 내에 있는지 확인하는 데 사용됩니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 방향 워크그룹의 총 개수입니다.
  // 작동 방식: 글로벌 인덱스 변환 시 X축 길이를 곱하는 계수로 사용됩니다.
  workgroups_x: u32,
  // 변수: pad2 ~ pad7
  // 목적: 메모리 정렬(Alignment)을 맞추기 위한 여유 공간(패딩)들입니다.
  // 작동 방식: WGSL 유니폼 버퍼의 레이아웃 규칙에 맞추기 위해 사용됩니다.
  pad2: u32,
  pad3: u32,
  pad4: u32,
  pad5: u32,
  pad6: u32,
  pad7: u32,
};

// 변수: params
// 목적: 외부에서 제공되는 파라미터 구조체를 바인딩합니다.
// 작동 방식: 바인딩 0에 읽기 전용으로 매핑됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: cond
// 목적: 요소별로 어느 값을 선택할지 결정하는 조건(Condition) 배열입니다.
// 작동 방식: 값이 0보다 크면 참(True), 그렇지 않으면 거짓(False)으로 평가됩니다.
@group(0) @binding(1) var<storage, read> cond: array<f32>;

// 변수: x
// 목적: 조건이 참(True)일 때 선택될 데이터 배열입니다.
// 작동 방식: 바인딩 2에 할당됩니다.
@group(0) @binding(2) var<storage, read> x: array<f32>;

// 변수: y
// 목적: 조건이 거짓(False)일 때 선택될 데이터 배열입니다.
// 작동 방식: 바인딩 3에 할당됩니다.
@group(0) @binding(3) var<storage, read> y: array<f32>;

// 변수: out
// 목적: 조건에 따라 x 또는 y에서 선택된 결과가 저장될 배열입니다.
// 작동 방식: 바인딩 4에 할당되어 계산 결과를 기록합니다.
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

// 함수: main
// 목적: cond 배열의 값에 따라 병렬로 x 또는 y의 요소를 선택하여 out 배열에 씁니다 (TensorFlow/PyTorch의 where 함수와 유사).
// 작동 방식: 각 스레드가 조건 평가를 거쳐 선택한 값을 기록합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 연산할 전체 배열 요소의 개수입니다.
  // 작동 방식: 유니폼 버퍼에서 읽어옵니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 방향의 워크그룹 개수입니다.
  // 작동 방식: 유니폼 버퍼에서 읽어옵니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 스레드가 담당하는 배열 요소의 1차원 인덱스입니다.
  // 작동 방식: 2차원 워크그룹 배열 인덱스를 1차원으로 평면화하여 계산합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 배열의 유효 범위를 넘어가는 스레드가 메모리에 접근하지 않게 합니다.
  // 작동 방식: 인덱스가 전체 크기 이상이면 함수를 끝냅니다.
  if (idx >= num_elements) {
    return;
  }

  // 제어문: if-else
  // 목적: 조건에 맞게 x 배열과 y 배열 중 하나의 값을 선택합니다.
  // 작동 방식: cond[idx]가 0보다 크면 x[idx]를, 아니면 y[idx]를 out[idx]에 할당합니다.
  if (cond[idx] > 0.0) {
    out[idx] = x[idx];
  } else {
    out[idx] = y[idx];
  }
}
`;

```

---

## `packages/forge/src/tensor/tensorRegistry.ts`

```typescript
/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * tensorRegistry.ts — GPU 텐서 생명주기 레지스트리
 *
 * C-06 Fix: dispose() 시 _globalQuotaManager.markPendingRelease() 즉시 호출.
 * NC-07 Fix: dynamic import() 제거 → 정적 import 사용 + device.destroy() 보장.
 * NL-03 Fix: Date.now() 제거 → 단조증가 ID만 사용 (타이밍 정보 노출 방지).
 */

import { TensorHandle, TensorRecord } from "../types";
import { AMEVAForgeDisposedError } from "../errors";
import { freeBuffer } from "../webgpu/buffers";
import { _globalQuotaManager } from "../webgpu/quota";
import { getDevice } from "../webgpu/device";

/**
 * WHAT: GPU 텐서의 생명주기를 관리하는 레지스트리 클래스입니다.
 * WHY: 생성된 텐서의 메타데이터와 WebGPU 버퍼를 중앙에서 추적하고 메모리 누수를 방지하기 위해 존재합니다.
 * HOW: Map 객체를 사용하여 고유한 핸들(TensorHandle)을 키로, 텐서 레코드(TensorRecord)를 값으로 저장 및 관리합니다.
 */
export class TensorRegistry {
  private records: Map<TensorHandle, TensorRecord> = new Map();
  private nextId: number = 1;

  snapshotHandles(): string[] {
    const list: string[] = [];
    for (const [handle, record] of this.records.entries()) {
      if (!record.disposed) list.push(handle);
    }
    return list;
  }

  registerRecord(record: Omit<TensorRecord, 'createdAt' | 'disposed'>): TensorHandle {
    const fullRecord: TensorRecord = {
      ...record,
      disposed: false,
      createdAt: this.nextId - 1,
    };
    this.records.set(record.handle, fullRecord);
    this.nextId++;
    return record.handle;
  }

  /**
   * WHAT: 새로운 텐서를 레지스트리에 등록하고 고유 핸들을 반환하는 함수입니다.
   * WHY: WebGPU 버퍼 및 메타데이터를 프레임워크가 추적할 수 있도록 레지스트리에 기록하기 위함입니다.
   * HOW: 예측 불가능한 UUID 기반의 핸들을 생성하고, 입력받은 정보와 함께 내부 records 맵에 저장합니다.
   */
  register(
    recordOmitHandle: Omit<TensorRecord, 'handle' | 'disposed' | 'createdAt'>
  ): TensorHandle {
    // F-015 Fix: 예측 가능한 핸들 생성을 막기 위해 암호학적 난수 기반 식별자 사용
    /**
     * WHAT: 암호학적으로 안전한 무작위 식별자(UUID) 문자열입니다.
     * WHY: 악의적인 사용자가 다른 텐서의 핸들을 추측하여 접근하는 것을 방지하기 위해 생성됩니다.
     * HOW: crypto.randomUUID가 사용 가능하면 이를 호출하고, 그렇지 않으면 Math.random()을 기반으로 임시 문자열을 생성합니다.
     */
    const uuid = typeof crypto !== 'undefined' && crypto.randomUUID 
      ? crypto.randomUUID() 
      : Math.random().toString(36).substring(2, 15);
    
    /**
     * WHAT: 텐서를 고유하게 식별하기 위한 최종 핸들 문자열입니다.
     * WHY: 외부에서 텐서를 참조할 때 이 문자열을 사용하여 안전하게 접근할 수 있도록 제공됩니다.
     * HOW: "tensor_" 접두사와 위에서 생성한 uuid 문자열을 결합하여 생성됩니다.
     */
    const handle = `tensor_${uuid}`;
    
    /**
     * WHAT: 레지스트리에 저장될 텐서의 모든 메타데이터를 포함하는 레코드 객체입니다.
     * WHY: WebGPU 버퍼, 모양(shape), 데이터 타입, 생성 순서 등을 한 곳에서 관리하기 위함입니다.
     * HOW: 전달된 recordOmitHandle 객체에 handle, disposed=false, createdAt(단조증가 ID)을 병합하여 생성합니다.
     */
    const record: TensorRecord = {
      ...recordOmitHandle,
      handle,
      disposed: false,
      createdAt: this.nextId - 1  // NL-03 Fix: monotonic counter, not timestamp
    };
    
    this.records.set(handle, record);
    this.nextId++;
    return handle;
  }

  /**
   * WHAT: 주어진 핸들을 사용하여 텐서 레코드를 조회하는 함수입니다.
   * WHY: 연산을 수행할 때 필요한 텐서의 메타데이터와 실제 WebGPU 버퍼를 가져오기 위해 존재합니다.
   * HOW: 내부 records 맵에서 핸들을 키로 조회하며, 존재하지 않거나 이미 폐기된 경우 에러를 발생시킵니다.
   */
  get(handle: TensorHandle): TensorRecord {
    /**
     * WHAT: 레지스트리에서 핸들로 조회한 텐서 레코드입니다.
     * WHY: 텐서가 실제로 존재하는지 검증하고 접근하기 위해 임시 변수에 저장합니다.
     * HOW: this.records.get(handle)을 통해 값을 가져옵니다.
     */
    const record = this.records.get(handle);
    
    if (!record) {
      throw new AMEVAForgeDisposedError(`Tensor not found: ${handle}`);
    }
    if (record.disposed) {
      throw new AMEVAForgeDisposedError(`Attempted to access disposed tensor: ${handle}`);
    }
    return record;
  }

  /**
   * WHAT: 특정 핸들의 텐서가 유효하게 존재하는지 확인하는 함수입니다.
   * WHY: 텐서가 해제(dispose)되었는지 예외 없이 안전하게 체크하기 위해 사용됩니다.
   * HOW: 핸들로 레코드를 조회하여 undefined가 아니고 disposed 상태가 아닌지(boolean)를 반환합니다.
   */
  has(handle: TensorHandle): boolean {
    /**
     * WHAT: 조회된 텐서 레코드 변수입니다.
     * WHY: 존재 여부 및 disposed 상태를 판별하기 위해 사용합니다.
     * HOW: records 맵에서 핸들로 가져옵니다.
     */
    const record = this.records.get(handle);
    return record !== undefined && !record.disposed;
  }

  /**
   * WHAT: 지정된 핸들의 텐서를 폐기하고 GPU 메모리를 해제하는 함수입니다.
   * WHY: 사용이 끝난 텐서의 메모리를 반환하여 OOM(Out of Memory)을 방지하고 자원 누수를 막기 위함입니다.
   * HOW: 레코드를 disposed로 표시하고 맵에서 제거한 뒤, QuotaManager와 WebGPU 큐를 통해 버퍼를 해제합니다.
   */
  dispose(handle: TensorHandle): void {
    if (!this.records.has(handle)) {
        return; // TS-H04: 이중 dispose 방어 — 이미 해제된 핸들 무시
    }
    
    const record = this.records.get(handle);
    if (!record || record.disposed) return;

    record.disposed = true;
    this.records.delete(handle);

    // C-06 Fix: 즉시 "해제 예약" 표시
    _globalQuotaManager.markPendingRelease(record.token);

    // NC-07 Fix: 정적 import된 getDevice() 사용 (dynamic import 제거)
    try {
      /**
       * WHAT: 현재 활성화된 WebGPU 디바이스 인스턴스입니다.
       * WHY: GPU에 제출된 모든 명령이 끝난 후 안전하게 버퍼를 파괴하기 위해 필요합니다.
       * HOW: getDevice() 유틸리티 함수를 호출하여 가져옵니다.
       */
      const device = getDevice();
      device.queue.onSubmittedWorkDone().then(() => {
        freeBuffer(record.buffer, record.token);
      }).catch(() => {
        // GPU 큐 실패 → 즉시 소각
        _safeDestroyBuffer(record);
      });
    } catch {
      // device가 없거나 lost → 즉시 quota 해제 + buffer 소각
      _safeDestroyBuffer(record);
    }
  }

  // F-016 Fix: 비동기 에러 발생 시 해당 핸들에 에러를 마킹
  markFailed(handle: TensorHandle, errorMsg: string): void {
    const record = this.records.get(handle);
    if (record && !record.disposed) {
      record.error = errorMsg;
    }
  }

  /**
   * WHAT: 레지스트리에 등록된 모든 텐서를 일괄 폐기하는 함수입니다.
   * WHY: 컨텍스트 초기화나 애플리케이션 종료 시 모든 GPU 자원을 확실하게 정리하기 위해 존재합니다.
   * HOW: 아직 폐기되지 않은 모든 레코드를 수집하고, 맵을 비운 뒤 GPU 큐가 비워지면 버퍼를 순차적으로 해제합니다.
   */
  clear(): void {
    /**
     * WHAT: 아직 해제되지 않아 메모리를 점유하고 있는 텐서 레코드들의 배열입니다.
     * WHY: 맵(Map)이 초기화된 후에도 이 객체들의 버퍼를 파괴하기 위해 참조를 유지해야 합니다.
     * HOW: this.records.values()를 배열로 변환하고 disposed가 false인 것만 필터링합니다.
     */
    const recordsToFree = Array.from(this.records.values()).filter(r => !r.disposed);
    this.records.clear();

    if (recordsToFree.length === 0) return;

    /**
     * WHAT: 해제 대상 텐서들을 순회하는 루프입니다.
     * WHY: 모든 할당된 텐서에 대해 해제 대기 상태임을 QuotaManager에 알리기 위함입니다.
     * HOW: for...of 구문을 사용하여 recordsToFree 배열을 순회합니다.
     */
    for (const record of recordsToFree) {
      _globalQuotaManager.markPendingRelease(record.token);
    }

    try {
      /**
       * WHAT: WebGPU 명령 큐의 상태를 확인하기 위한 디바이스 객체입니다.
       * WHY: 큐에 대기 중인 작업이 텐서를 참조할 수 있으므로, 작업 완료 후 안전하게 해제하기 위해 사용됩니다.
       * HOW: getDevice()를 통해 인스턴스를 얻어옵니다.
       */
      const device = getDevice();
      device.queue.onSubmittedWorkDone().then(() => {
        /**
         * WHAT: GPU 작업이 완료된 후 각 버퍼를 해제하는 루프입니다.
         * WHY: 실제 VRAM과 QuotaManager의 할당량을 반환하기 위해 필요합니다.
         * HOW: for...of 루프를 돌며 freeBuffer를 호출합니다.
         */
        for (const record of recordsToFree) {
          freeBuffer(record.buffer, record.token);
        }
      }).catch(() => {
        /**
         * WHAT: GPU 큐 대기 실패 시 강제 해제하는 루프입니다.
         * WHY: 큐에러가 발생해도 메모리 누수를 방지하기 위해 존재합니다.
         * HOW: _safeDestroyBuffer 헬퍼를 직접 호출합니다.
         */
        for (const record of recordsToFree) {
          _safeDestroyBuffer(record);
        }
      });
    } catch {
      // device already lost
      /**
       * WHAT: 디바이스 유실 시 텐서 버퍼를 강제 해제하는 루프입니다.
       * WHY: 디바이스가 유실되어 큐 대기를 할 수 없으므로 남은 리소스를 정리하기 위해 필요합니다.
       * HOW: for...of 루프를 돌며 _safeDestroyBuffer를 호출하고, 이후 쿼터를 초기화합니다.
       */
      for (const record of recordsToFree) {
        _safeDestroyBuffer(record);
      }
      _globalQuotaManager.reset();
    }
  }
}

/**
 * WHAT: 텐서의 WebGPU 버퍼를 파괴하고 할당 토큰을 해제하는 유틸리티 함수입니다.
 * WHY: 디바이스 유실이나 큐 실패 상황에서도 예외 발생 없이 버퍼 자원을 반환하기 위해 존재합니다.
 * HOW: try-catch 블록 안에서 buffer.destroy()를 호출하고, _globalQuotaManager.releaseToken()을 호출합니다.
 */
function _safeDestroyBuffer(record: TensorRecord): void {
  try {
    record.buffer.destroy();
  } catch {
    // buffer가 이미 destroyed
  }
  _globalQuotaManager.releaseToken(record.token);
}

/**
 * WHAT: 전역적으로 사용되는 텐서 레지스트리의 단일 인스턴스(싱글톤)입니다.
 * WHY: 애플리케이션 전체에서 동일한 텐서 관리 상태를 공유하기 위해 생성됩니다.
 * HOW: TensorRegistry 클래스의 새 인스턴스를 생성하여 내보냅니다(export).
 */
export const _globalRegistry = new TensorRegistry();

```

---

## `packages/forge/src/tensor/validateDType.ts`

```typescript
/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
import { AMEVAForgeDTypeError } from "../errors";
import { DType } from "../types";

/**
 * WHAT: 입력된 데이터 타입(dtype)이 프레임워크에서 지원하는 타입인지 검증하는 함수입니다.
 * WHY: 지원하지 않는 데이터 타입이 사용될 경우 발생할 수 있는 메모리 계산 오류 및 WebGPU 셰이더 오류를 사전에 방지하기 위해 존재합니다.
 * HOW: 입력된 dtype 문자열이 "float32"인지 비교하고, 일치하지 않으면 AMEVAForgeDTypeError 예외를 발생시킵니다. asserts 키워드를 사용하여 타입스크립트 컴파일러에게 dtype이 DType임을 보장합니다.
 */
export function validateDType(dtype: string): asserts dtype is DType {
  // WHAT: dtype이 "float32"가 아닌지 확인하는 조건문입니다.
  // WHY: 현재 WebGPU 연산 파이프라인에서 float32 데이터 타입만 완벽하게 지원하므로 이를 검증하기 위함입니다.
  // HOW: 일치 연산자(!==)를 통해 입력 문자열이 정확히 "float32"와 다른지 확인합니다.
  if (dtype !== "float32") {
    throw new AMEVAForgeDTypeError(`Unsupported dtype: ${dtype}. Only float32 is supported.`);
  }
}

```

---

## `packages/forge/src/tensor/validateShape.ts`

```typescript
/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
import { AMEVAForgeShapeError, AMEVAForgeDTypeError } from "../errors";
import { DType } from "../types";

/** 
 * WHAT: dtype별 바이트 크기를 매핑하는 상수 딕셔너리입니다.
 * WHY: 텐서의 전체 바이트 크기를 계산할 때, 데이터 타입마다 차지하는 바이트 수가 다르기 때문에 이를 정확히 계산하기 위해 존재합니다.
 * HOW: Record 유틸리티 타입을 사용하여 DType 문자열을 키로, 바이트 수(number)를 값으로 갖는 객체를 정의합니다.
 */
const BYTES_PER_ELEMENT: Record<DType, number> = {
  "float32": 4,
  // float16: 2 — 셰이더 구현 완료 후 추가 예정
  // int32: 4 — 셰이더 구현 완료 후 추가 예정
};

/**
 * WHAT: 텐서가 가질 수 있는 최대 원소 수를 정의하는 상수입니다.
 * WHY: 메모리 초과(OOM) 오류를 방지하고 시스템의 안정성을 유지하기 위해 하드 리미트를 설정합니다.
 * HOW: 256MB 크기의 float32 버퍼에 맞추어 256 * 1024 * 1024로 값을 할당합니다.
 */
const MAX_ELEMENTS = 256 * 1024 * 1024;

/**
 * WHAT: 텐서 shape의 최대 랭크(차원 수)를 정의하는 상수입니다.
 * WHY: WebGPU에서 처리할 수 있는 차원의 한계를 설정하고, 과도하게 복잡한 다차원 텐서의 생성을 방지합니다.
 * HOW: 스칼라(rank 0)부터 시작하여 최대 8차원까지 허용하도록 숫자 8을 할당합니다.
 */
const MAX_RANK = 8; // NM-06: 스칼라(rank 0) 포함하여 0~8까지 허용

/**
 * WHAT: 텐서 shape의 유효성을 검증하고 총 원소 수를 반환하는 함수입니다.
 * WHY: 잘못된 텐서 형태나 예상치 못한 크기의 메모리 할당을 사전에 차단하여 안전한 연산을 보장하기 위함입니다.
 * HOW: 입력된 shape가 배열인지, 랭크 제한을 넘지 않는지 확인한 후, 각 차원의 값을 곱해 총 원소 수를 구합니다. 예상 바이트 크기가 주어진 경우 이를 함께 검증합니다.
 *
 * M-01 Fix: dtype별 바이트 크기를 BYTES_PER_ELEMENT 맵으로 정확히 계산.
 * NM-06 Fix: rank 0 스칼라 텐서 허용 (PyTorch/JAX/TF 표준).
 *   rank 0 = shape=[], elements=1, byteLength=4 (단일 float32 스칼라)
 */
export function validateShape(
  shape: number[],
  dtype: DType,
  expectedByteLength?: number
): number {
  if (!Array.isArray(shape)) {
    throw new AMEVAForgeShapeError("Shape must be an array.");
  }
  // NM-06 Fix: rank 0 (shape=[]) 허용 — 스칼라 텐서
  if (shape.length > MAX_RANK) {
    throw new AMEVAForgeShapeError(
      `Shape rank must be between 0 and ${MAX_RANK}, got ${shape.length}.`
    );
  }

  /**
   * WHAT: 텐서의 총 원소 수를 누적하여 저장하는 변수입니다.
   * WHY: shape 배열의 각 차원을 곱하여 텐서 데이터가 차지할 실제 원소의 총 개수를 알아내기 위해 필요합니다.
   * HOW: 스칼라(rank 0)의 경우를 처리하기 위해 1로 초기화됩니다.
   */
  let elements = 1;

  /**
   * WHAT: shape 배열의 각 차원 값을 순회하며 원소 수를 계산하고 유효성을 검사하는 루프입니다.
   * WHY: 모든 차원의 값이 양의 정수인지 확인하고, 안전한 정수 범위를 벗어나는 오버플로우를 감지하기 위해 존재합니다.
   * HOW: 인덱스 i를 0부터 shape.length - 1까지 증가시키며 dim 값을 추출해 검증하고 elements에 누적 곱셈을 수행합니다.
   */
  for (let i = 0; i < shape.length; i++) {
    /**
     * WHAT: 현재 검사 중인 텐서의 특정 차원(dimension)의 크기를 나타내는 변수입니다.
     * WHY: 이 값이 유효한 양의 정수인지 검사하기 위해 루프 내에서 임시로 저장합니다.
     * HOW: shape 배열에서 i번째 인덱스의 값을 참조하여 가져옵니다.
     */
    const dim = shape[i];
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new AMEVAForgeShapeError(`shape[${i}] must be positive, got ${dim}`);
    }
    if (dim > Number.MAX_SAFE_INTEGER / elements) {
      throw new AMEVAForgeShapeError("Shape product overflows safe integer limit.");
    }
    elements *= dim;
  }

  if (elements > MAX_ELEMENTS) {
    throw new AMEVAForgeShapeError(
      `Tensor size exceeds max elements limit: ${elements} > ${MAX_ELEMENTS}`
    );
  }

  if (expectedByteLength !== undefined) {
    /**
     * WHAT: 입력된 dtype이 차지하는 단일 원소의 바이트 크기를 저장하는 변수입니다.
     * WHY: 전체 텐서의 예상 바이트 크기를 계산하기 위해 요소당 크기를 알아야 합니다.
     * HOW: BYTES_PER_ELEMENT 상수 맵에서 dtype을 키로 사용하여 값을 조회합니다.
     */
    const bytesPerElement = BYTES_PER_ELEMENT[dtype];
    if (bytesPerElement === undefined) {
      throw new AMEVAForgeDTypeError(
        `Unsupported dtype for byte size calculation: "${dtype}". ` +
        `Supported: ${Object.keys(BYTES_PER_ELEMENT).join(', ')}`
      );
    }
    /**
     * WHAT: shape와 dtype을 바탕으로 계산된 텐서의 실제 필요 바이트 크기를 담는 변수입니다.
     * WHY: 사용자가 제시한 expectedByteLength와 비교하여 데이터 정합성을 검증하기 위해 계산합니다.
     * HOW: 누적된 총 원소 수(elements)에 원소당 바이트 크기(bytesPerElement)를 곱하여 구합니다.
     */
    const calculatedBytes = elements * bytesPerElement;
    if (calculatedBytes !== expectedByteLength) {
      throw new AMEVAForgeShapeError(
        `Shape/data size mismatch: shape ${JSON.stringify(shape)} (${dtype}) ` +
        `implies ${calculatedBytes} bytes, but data is ${expectedByteLength} bytes.`
      );
    }
  }

  return elements;
}

```

---

## `packages/forge/src/types.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * 
 * types.ts — 핵심 타입 정의
 *
 * H-06 Fix: DType을 실제 지원되는 "float32"만으로 제한.
 *   기존에 float16/int32가 타입에 있었지만 셰이더와 검증 로직이 float32 전용이라
 *   타입 에러 없이 잘못된 셰이더에 전달되는 버그가 있었다.
 *   → float16/int32 추가는 셰이더 커널 구현과 동시에 이루어져야 한다.
 */

/**
 * WHAT: 텐서를 고유하게 식별하기 위한 핸들(Handle) 타입입니다.
 * WHY: 시스템 내부적으로 파이썬과 자바스크립트 간에 무겁고 복잡한 객체를 통째로 넘기지 않고, 문자열 형태의 참조값만 주고받음으로써 통신 오버헤드를 최소화하기 위해 고안되었습니다.
 * HOW: 타입스크립트의 타입 앨리어스(Type Alias)를 사용하여 string을 TensorHandle로 추상화 명명합니다.
 */
export type TensorHandle = string;

/** 
 * WHAT: 텐서의 데이터 타입을 정의하는 문자열 리터럴 타입입니다. (H-06: 현재 구현이 실제로 지원하는 dtype만 허용)
 * WHY: 현재 작성된 커스텀 셰이더와 연산 검증 로직이 32비트 부동소수점만 지원하므로, 잘못된 타입이 주입되어 런타임 버그나 셰이더 크래시가 발생하는 것을 컴파일 타임에 엄격히 제한하고 방지하기 위함입니다.
 * HOW: 타입스크립트의 리터럴 타입을 활용해 오직 "float32"라는 문자열 값만 허용하도록 고정합니다.
 */
export type DType = "float32";

import { AllocationToken } from "./webgpu/quota";

/**
 * WHAT: 메모리에 할당된 단일 텐서에 대한 전체 상태와 리소스 정보를 담고 있는 핵심 레코드 인터페이스입니다.
 * WHY: 텐서 레지스트리(Tensor Registry)가 텐서의 수명 주기(생성, 사용, 해제)를 완벽하게 추적하고 관리하기 위한 중앙 정보 저장소 역할을 제공하기 위해서 존재합니다.
 * HOW: GPUBuffer 리소스 자체와 차원(Shape), 데이터 타입(DType), 그리고 할당 토큰 등 다양한 메타데이터를 하나의 구조화된 객체 형태로 묶어서 표현합니다.
 */
export interface TensorRecord {
  /** WHAT: 텐서의 고유 식별자 문자열. WHY: 레지스트리에서 이 텐서를 특정하기 위해. HOW: UUID나 고유 해시 형태의 텍스트 저장. */
  handle: TensorHandle;
  
  /** WHAT: 텐서의 차원별 크기를 담은 숫자 배열. WHY: 다차원 데이터 구조(배치, 높이, 너비, 채널 등)를 해석하고 메모리 오프셋을 계산하기 위해. HOW: [정수, 정수, ...] 형태의 배열로 저장. */
  shape: number[];
  
  /** WHAT: 텐서 내 원소들의 데이터 타입. WHY: 데이터를 바이트로 변환하거나 셰이더에서 올바르게 읽을 수 있게 보장하기 위해. HOW: DType 타입(현재 "float32" 전용)으로 제한. */
  dtype: DType;
  
  /** WHAT: 버퍼가 차지하는 총 바이트 단위 크기. WHY: GPU 메모리 할당량을 계산하고 버퍼 복사 시 정확한 크기를 설정하기 위해. HOW: 숫자로 저장. */
  byteLength: number;
  
  /** WHAT: WebGPU 인스턴스의 실제 메모리 버퍼 객체 참조. WHY: 하드웨어 가속 연산 및 메모리 읽기/쓰기 작업을 직접 수행하기 위해. HOW: 네이티브 GPUBuffer 객체를 직접 레퍼런싱. */
  buffer: GPUBuffer;
  
  /** WHAT: QuotaManager로부터 발급받은 메모리 할당량을 추적하는 토큰. WHY: 시스템 전체 메모리 한도를 관리하고 자원 해제 시 할당량을 정확히 반납하기 위해. HOW: 발급된 토큰 객체를 참조로 가짐. */
  token: AllocationToken;
  
  /** WHAT: 해당 텐서의 메모리가 이미 해제(disposed)되었는지를 나타내는 플래그입니다. WHY: Use-After-Free 같은 비정상적인 메모리 접근을 사전에 차단하기 위해. HOW: 불리언(boolean) 값으로 상태를 저장하고 체크합니다. */
  disposed: boolean;
  
  /** 
   * WHAT: 단조 증가하는 등록 순서값입니다. (Monotonic registration order)
   * WHY: 타임스탬프의 경우 시스템 시간에 따라 값이 역행하거나 중복될 가능성이 있으므로, 순서 보장이 필요한 로직(예: 디버깅, LRU 캐싱 등)에서 완벽한 선후 관계를 판별하기 위해서입니다.
   * HOW: 내부 카운터를 증가시키며 얻은 고유 숫자를 할당합니다.
   */
  createdAt: number;

  /** WHAT: 트랜잭션 롤백 시 이 핸들에 매핑된 에러 메시지입니다. WHY: F-016 수정의 일환으로 개별 에러를 추적하기 위함. HOW: 문자열로 기록. */
  error?: string;
}

/**
 * WHAT: 외부에 텐서의 메타데이터(크기, 형태, 상태)만을 제공하기 위한 읽기 전용 형태의 인터페이스입니다.
 * WHY: 실제 GPUBuffer 객체나 내부 할당 토큰 같은 민감한 하드웨어 리소스를 숨기고, 파이썬(Pyodide) 쪽이나 외부에 텐서의 정보만 확인할 때 필요한 최소한의 데이터만 안전하게 노출하여 은닉성(Encapsulation)을 보장하기 위함입니다.
 * HOW: TensorRecord에서 민감한 속성을 제외한 서브셋(subset) 필드만으로 구성된 구조체를 정의합니다.
 */
export interface TensorInfo {
  /** WHAT: 텐서의 고유 식별자. WHY: 이 정보가 어떤 텐서의 것인지 매핑하기 위해. HOW: 식별자 복사. */
  handle: TensorHandle;
  /** WHAT: 텐서의 차원별 크기 배열. WHY: 외부에서 텐서의 모양을 파악하기 위해. HOW: 숫자 배열 반환. */
  shape: number[];
  /** WHAT: 텐서 데이터 타입. WHY: 외부에서 데이터 형식을 파악하기 위해. HOW: DType 값 반환. */
  dtype: DType;
  /** WHAT: 총 바이트 크기. WHY: 데이터 전송 크기 등을 예측하기 위해. HOW: 숫자 값 반환. */
  byteLength: number;
  /** WHAT: 해제(Disposed) 여부 확인 플래그. WHY: 외부에서 유효한 텐서인지 검사하기 위해. HOW: 불리언 값 반환. */
  disposed: boolean;
}

```

---

## `packages/forge/src/webgpu/buffers.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * buffers.ts — GPU 버퍼 할당, 읽기 인터페이스
 *
 * C-05 Fix: _stagingBuffers 전역 Map 제거 → mapBufferAsync가 staging buffer 직접 반환.
 * H-05 / NH-05 Fix: "Zero-Copy" 주석 수정 — GPU→CPU 전송은 1번 copy가 불가피.
 *   WebGPU 스펙상 GPU 메모리를 WASM 힙과 직접 공유할 수 없다 (CUDA pinned memory와 달리).
 *   최소 1번의 copy는 WebGPU의 구조적 한계이며 Dawn, wgpu, TensorFlow.js도 동일.
 * ARC-01 Fix: createBuffer() OOM은 device.pushErrorScope()로만 감지 가능 — 문서화.
 */

import { getDevice, getQueue } from "./device";
import { _globalQuotaManager, AllocationKind, AllocationToken } from "./quota";
import { AMEVAForgeValidationError } from "../errors";

/**
 * WHAT: 지정된 크기와 용도에 맞게 GPU 버퍼를 할당합니다.
 * WHY: WebGPU의 버퍼 생성을 추상화하고 전역 할당량(Quota) 관리 시스템과 통합하여 메모리 부족(OOM)을 방지하기 위해 존재합니다.
 * HOW: QuotaManager를 통해 `byteLength`만큼의 메모리를 예약한 후, `device.createBuffer`를 호출하여 버퍼를 생성합니다. 실패 시 예약된 메모리 토큰을 반환(release)하고 에러를 던집니다.
 */
export function allocateBuffer(
  byteLength: number,
  usage: GPUBufferUsageFlags,
  kind: AllocationKind = 'tensor',
  ownerGraph: string | null = null
): { buffer: GPUBuffer, token: AllocationToken } {
  const token = _globalQuotaManager.reserveToken(byteLength, kind, ownerGraph);
  try {
    const buffer = getDevice().createBuffer({ size: byteLength, usage });
    return { buffer, token };
  } catch (e) {
    _globalQuotaManager.releaseToken(token);
    throw e;
  }
}

/**
 * WHAT: 주어진 GPU 버퍼에 Float32Array 데이터를 씁니다.
 * WHY: CPU 측의 데이터를 GPU 버퍼로 복사하여 GPU 연산에 사용할 수 있도록 하기 위해 필요합니다.
 * HOW: WebGPU 큐(`device.queue.writeBuffer`)를 사용하여 주어진 데이터의 전체 크기만큼 지정된 버퍼의 오프셋 0부터 복사합니다.
 */
export function writeFloat32Array(buffer: GPUBuffer, data: Float32Array): void {
  if (data.byteLength > buffer.size) {
    throw new AMEVAForgeValidationError(
      `writeFloat32Array overflow: data size (${data.byteLength}B) exceeds buffer capacity (${buffer.size}B)`
    );
  }
  getQueue().writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
}

import { _globalUniformPool } from "./uniformPool";

interface StagingPoolEntry {
  buffer: GPUBuffer;
  token: AllocationToken;
}

export const _stagingPool: Map<number, StagingPoolEntry[]> = new Map();
const STAGING_POOL_MAX_PER_SIZE = 4;

export function clearStagingPool(): void {
  for (const entries of _stagingPool.values()) {
    for (const { buffer, token } of entries) {
      try { freeBuffer(buffer, token); } catch {}
    }
  }
  _stagingPool.clear();
  try { _globalUniformPool.clear(); } catch {}
}

export async function flushGC(): Promise<void> {
  try {
    const device = getDevice();
    await device.queue.onSubmittedWorkDone();
    await _globalUniformPool.retireSubmitted(device);
  } catch {}
  clearStagingPool();
  try { _globalUniformPool.clear(); } catch {}
}

export function acquireStagingBuffer(byteLength: number): { buffer: GPUBuffer, token: AllocationToken } {
  const pool = _stagingPool.get(byteLength);
  if (pool && pool.length > 0) {
    return pool.pop()!;
  }
  const usage = typeof GPUBufferUsage !== 'undefined' ? (GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST) : (0x0001 | 0x0008);
  const { buffer, token } = allocateBuffer(
    byteLength,
    usage,
    'staging',
    'StagingPool'
  );
  return { buffer, token };
}

export function releaseStagingBuffer(
  buffer: GPUBuffer,
  token: AllocationToken,
  byteLength: number,
  isCorrupted: boolean = false
): void {
  if (isCorrupted) {
    try { buffer.destroy(); } catch {}
    if (token) {
      try { _globalQuotaManager.releaseToken(token); } catch {}
    }
    return;
  }

  const pool = _stagingPool.get(byteLength) ?? [];
  if (pool.length < STAGING_POOL_MAX_PER_SIZE) {
    pool.push({ buffer, token });
    _stagingPool.set(byteLength, pool);
  } else {
    try { freeBuffer(buffer, token); } catch {}
  }
}

/**
 * WHAT: GPU 버퍼의 데이터를 읽어서 CPU 메모리 상의 Float32Array로 반환합니다.
 * WHY: GPU에서 처리된 결과 데이터를 CPU로 가져와서 애플리케이션 수준에서 활용(예: 출력, 저장)하기 위해 존재합니다.
 * HOW: 
 *   1. 복사를 위한 중간 버퍼(Staging Buffer)를 MAP_READ와 COPY_DST 용도로 할당합니다.
 *   2. CommandEncoder를 사용해 원본 버퍼의 데이터를 Staging Buffer로 복사하고 큐에 제출합니다.
 *   3. Staging Buffer를 비동기적으로 맵핑(mapAsync)하여 CPU에서 읽을 수 있게 합니다.
 *   4. 데이터를 읽어 Float32Array로 복사한 후 버퍼를 해제(unmap, destroy)하고 토큰을 반환합니다.
 */
export async function readBufferToFloat32Array(
  buffer: GPUBuffer,
  byteLength: number
): Promise<Float32Array> {
  const device = getDevice();
  const { buffer: stagingBuffer, token } = acquireStagingBuffer(byteLength);
  let isCorrupted = false;

  try {
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
    device.queue.submit([commandEncoder.finish()]);
    await stagingBuffer.mapAsync(GPUMapMode.READ);

    try {
      const arrayBuffer = stagingBuffer.getMappedRange();
      return new Float32Array(arrayBuffer.slice(0));
    } finally {
      stagingBuffer.unmap();
    }
  } catch (err) {
    isCorrupted = true;
    throw err;
  } finally {
    releaseStagingBuffer(stagingBuffer, token, byteLength, isCorrupted);
  }
}

/**
 * WHAT: GPU 버퍼의 내용을 읽기 위해 Staging Buffer를 생성하고 비동기적으로 맵핑합니다.
 * WHY: 대용량 데이터 전송 시 메모리 맵핑을 직접 제어하거나 제로 카피(Zero-Copy) 메커니즘과 유사한 최적화를 구현하기 위해 필요합니다.
 * HOW: `MAP_READ | COPY_DST` 속성의 Staging 버퍼를 새로 할당하고, 원본 버퍼의 내용을 복사하기 위한 커맨드를 큐에 제출한 뒤, `mapAsync`를 호출하여 맵핑된 버퍼와 할당 토큰을 반환합니다.
 */
export async function mapBufferAsync(
  buffer: GPUBuffer,
  byteLength: number
): Promise<{ stagingBuffer: GPUBuffer, token: AllocationToken, byteLength: number }> {
  const device = getDevice();
  const { buffer: stagingBuffer, token } = acquireStagingBuffer(byteLength);

  try {
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
    device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    return { stagingBuffer, token, byteLength };
  } catch (e) {
    releaseStagingBuffer(stagingBuffer, token, byteLength, true);
    throw e;
  }
}

/**
 * WHAT: 맵핑이 완료된 Staging 버퍼의 데이터를 외부에서 제공된 Float32Array 배열에 직접 복사합니다.
 * WHY: 새로운 배열 객체를 생성하지 않고 기존 메모리(Pre-allocated buffer)를 재사용하여 메모리 할당 및 가비지 컬렉션(GC) 부하를 줄이기 위해 사용됩니다.
 * HOW: Staging 버퍼의 맵핑 범위를 가져와서 전달된 `outArray`에 `set` 메서드로 데이터를 덮어쓴 후, unmap 후 Staging Pool로 반환합니다.
 */
export function readMappedInto(
  stagingBuffer: GPUBuffer,
  token: AllocationToken,
  outArray: Float32Array
): void {
  const byteLength = outArray.byteLength;
  let isCorrupted = false;
  try {
    const arrayBuffer = stagingBuffer.getMappedRange();
    const mapped = new Float32Array(arrayBuffer);
    if (outArray.length !== mapped.length) {
      throw new RangeError(`readMappedInto destination length mismatch: expected ${mapped.length}, got ${outArray.length}`);
    }
    outArray.set(mapped);
  } catch (err) {
    isCorrupted = true;
    throw err;
  } finally {
    try { stagingBuffer.unmap(); } catch {}
    releaseStagingBuffer(stagingBuffer, token, byteLength, isCorrupted);
  }
}

/**
 * WHAT: 할당된 GPU 버퍼를 메모리에서 해제하고, 관련된 할당량 토큰(AllocationToken)을 반환합니다.
 * WHY: WebGPU 리소스 누수를 방지하고, 전역 쿼타 매니저(Quota Manager)에 반환하여 다른 작업에서 가용 메모리를 사용할 수 있도록 하기 위해 존재합니다.
 * HOW: `buffer.destroy()`를 호출하여 실제 GPU 리소스를 해제한 다음, `_globalQuotaManager.releaseToken(token)`을 통해 예약된 메모리 용량을 반환합니다.
 */
export function freeBuffer(buffer: GPUBuffer, token: AllocationToken): void {
  buffer.destroy();
  _globalQuotaManager.releaseToken(token);
}

```

---

## `packages/forge/src/webgpu/device.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * device.ts — WebGPU 싱글톤 디바이스 래퍼
 *
 * H-04 Fix: getAdapter() export 추가 → gpuCore.ts에서 adapter.limits 조회 가능
 * L-03 Fix: device lost 시 onDeviceLostCallback을 통해 pipelineCache도 무효화
 */

import { AMEVAForgeWebGPUUnavailableError, AMEVAForgeDeviceError } from "../errors";
import { _globalQuotaManager } from "./quota";

declare var process: any;

/**
 * WHAT: 개발 환경이나 디버그 모드에서만 시스템 메시지를 출력하는 안전한 로깅 함수입니다.
 * WHY: 불필요한 콘솔 출력을 프로덕션 환경에서 방지하고, 에러 없이 안전하게 로그를 남기기 위해 사용됩니다.
 * HOW: 현재 실행 환경이 개발 모드(NODE_ENV, AMEVA_DEBUG, __DEV__, Vite env 등)인지 확인하고 조건을 만족할 때만 `globalThis.log`를 통해 메시지를 출력합니다. 예외가 발생해도 시스템이 멈추지 않도록 try-catch로 감쌉니다.
 */
let _isLogging = false;
let _consecutiveLoggingErrors = 0;
let _lastSelfHealTimestamp = 0;

export function _safeLog(msg: string) {
  if (_isLogging) return;
  _isLogging = true;

  try {
    const isDev = 
      (typeof process !== 'undefined' && process.env && process.env.NODE_ENV !== 'production') ||
      (typeof (globalThis as any).AMEVA_DEBUG !== 'undefined' && (globalThis as any).AMEVA_DEBUG) ||
      (typeof (globalThis as any).__DEV__ !== 'undefined' && (globalThis as any).__DEV__);

    if (!isDev) return;

    if (typeof (globalThis as any).log === 'function') {
      (globalThis as any).log(msg, 'system');
    }
  } catch (err) {
    _consecutiveLoggingErrors++;
    const now = Date.now();
    
    // 자가 치유(Self-Healing): 3회 연속 에러 시 쿼터 상태 자동 정합 및 치료
    if (_consecutiveLoggingErrors >= 3 && now - _lastSelfHealTimestamp > 5000) {
      _lastSelfHealTimestamp = now;
      _consecutiveLoggingErrors = 0;
      try {
        _globalQuotaManager.sanitizePendingBytes();
      } catch (sanitizeErr) {
        // Safe fallback
      }
    }

    if (typeof console !== 'undefined' && typeof console.debug === 'function') {
      console.debug('[AMEVA-SafeLog-Fallback]', msg, err);
    }
  } finally {
    _isLogging = false;
  }
}

/**
 * WHAT: 초기화된 논리적 WebGPU 디바이스(GPUDevice) 인스턴스를 저장하는 내부 변수입니다.
 * WHY: 모듈 내에서 싱글톤(singleton) 패턴을 유지하여 여러 번 초기화되지 않도록 상태를 관리합니다.
 * HOW: initWebGPU 함수 내에서 생성된 디바이스가 할당되며, 디바이스 손실(device lost) 시 다시 null로 초기화됩니다.
 */
let device: GPUDevice | null = null;
/**
 * WHAT: WebGPU 기능 및 하드웨어 한계를 나타내는 물리적 GPU 어댑터(GPUAdapter) 인스턴스를 저장하는 내부 변수입니다.
 * WHY: 디바이스 생성 전 스펙(limits)을 검사하거나 생성 후 하드웨어 정보를 참조하기 위해 캐싱해 둡니다.
 * HOW: initWebGPU 실행 시 requestAdapter()로 요청받아 할당되며, 디바이스 손실 시 함께 정리됩니다.
 */
let adapter: GPUAdapter | null = null;
/**
 * WHAT: 디바이스 손실(device lost) 발생 시 실행될 콜백 함수를 저장하는 변수입니다.
 * WHY: GPU 오류나 컨텍스트 초기화 상황이 발생했을 때 상위 애플리케이션으로 이벤트를 위임하기 위해 필요합니다.
 * HOW: setDeviceLostCallback 함수를 통해 설정되며, device.lost Promise가 해결(resolve)될 때 내부적으로 호출됩니다.
 */
let onDeviceLostCallback: (() => void) | null = null;

/**
 * WHAT: 시스템 환경에서 WebGPU 디바이스 및 어댑터를 비동기적으로 초기화합니다.
 * WHY: WebGPU API를 사용하기 위해 필수적인 하드웨어 어댑터(adapter)와 논리적 디바이스(device) 인스턴스를 확보하고 전역에서 접근할 수 있도록 캐싱하기 위해 존재합니다.
 * HOW: 
 *   1. navigator.gpu 객체가 존재하는지 확인하고, requestAdapter()로 물리적 GPU 어댑터를 요청합니다.
 *   2. 어댑터가 지원하는 최대 버퍼 크기 등의 한계를 파악하여 requestDevice()로 디바이스를 생성합니다.
 *   3. 디바이스 손실(device.lost) 이벤트를 수신하여 리소스를 정리하고 등록된 콜백을 실행하도록 설정합니다.
 */
export async function initWebGPU(options?: GPURequestAdapterOptions): Promise<void> {
  _safeLog(`[device.ts] initWebGPU started. current device=${device ? 'SET' : 'NULL'}`);
  if (device) return;

  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new AMEVAForgeWebGPUUnavailableError(
      "WebGPU is not available in this environment. " +
      "Ensure you are running in a supported browser with WebGPU enabled."
    );
  }

  adapter = await navigator.gpu.requestAdapter(options);
  if (!adapter) {
    // Try software fallback
    adapter = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true });
    if (adapter) {
      _safeLog('[AMEVA] WARNING: Using software fallback adapter. Performance will be severely degraded.');
    }
  }
  
  if (!adapter) {
    throw new AMEVAForgeWebGPUUnavailableError(
      "Failed to request a WebGPU adapter. " +
      "Your GPU may not support WebGPU, or the browser has disabled it."
    );
  }

  const requiredLimits: any = {};
  if (adapter.limits) {
    requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
    requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
  }
  
  device = await adapter.requestDevice({ requiredLimits });

  if (device.limits && device.limits.maxStorageBufferBindingSize) {
    const maxBinding = device.limits.maxStorageBufferBindingSize;
    const adaptedHard = Math.max(1024 * 1024 * 1024, maxBinding * 2);
    const adaptedSoft = Math.max(768 * 1024 * 1024, maxBinding);
    try {
      _globalQuotaManager.setLimits(adaptedHard, adaptedSoft);
    } catch (e) { /* intentionally empty: if quota limit set fails, rely on default limits rather than crashing initialization */ }
  }

  _safeLog(`[device.ts] initWebGPU finished. device successfully created.`);

  device.lost.then((info) => {
    const msg = `[AMEVA] WebGPU Device Lost: ${info.message} (reason: ${info.reason})`;
    console.error(msg);
    _safeLog(msg);
    device = null;
    // (globalThis as any).__AMEVA_DEVICE__ = null;
    adapter = null;
    if (onDeviceLostCallback) {
      onDeviceLostCallback();
    }
  });
}

/**
 * WHAT: 전역에 캐시된 WebGPU 디바이스 인스턴스를 반환합니다.
 * WHY: 애플리케이션의 여러 모듈에서 동일한 단일 디바이스 인스턴스에 접근하여 버퍼 및 텍스처를 생성할 수 있도록 제공하기 위함입니다.
 * HOW: 내부 `device` 변수가 초기화되어 있는지 확인하고, 없을 경우 예외(AMEVAForgeDeviceError)를 발생시키며, 존재할 경우 그대로 반환합니다.
 */
export function getDevice(): GPUDevice {
  if (!device) {
    const globalExists = typeof globalThis.amevaForge !== "undefined";
    throw new AMEVAForgeDeviceError(
      `WebGPU device is not initialized. (globalThis.amevaForge exists: ${globalExists}). Call await init() first.`
    );
  }
  return device;
}

export function _setDeviceForTesting(d: any): void {
  device = d;
}

/**
 * WHAT: 전역에 캐시된 WebGPU 어댑터(Adapter) 인스턴스를 반환합니다.
 * WHY: GPU의 하드웨어 스펙(limits, features 등)을 조회하거나 디바이스 기능 제약 조건을 파악하기 위해 외부 모듈에서 어댑터에 접근할 수 있게 합니다.
 * HOW: 내부 `adapter` 변수를 그대로 반환합니다. 아직 초기화되지 않았다면 null이 반환될 수 있습니다.
 */
export function getAdapter(): GPUAdapter | null {
  return adapter;
}

/**
 * WHAT: 초기화된 WebGPU 디바이스와 연결된 커맨드 큐(GPUQueue)를 반환합니다.
 * WHY: 데이터를 버퍼로 전송(writeBuffer)하거나 렌더링/컴퓨트 커맨드(submit)를 실행할 수 있도록 접근 지점을 제공합니다.
 * HOW: `getDevice()` 함수를 호출해 디바이스를 얻은 후 `device.queue` 속성을 반환합니다.
 */
export function getQueue(): GPUQueue {
  return getDevice().queue;
}

/**
 * WHAT: WebGPU 디바이스가 현재 성공적으로 초기화되어 사용 가능한지 여부를 반환합니다.
 * WHY: 기능 호환성 검사나 런타임 조건부 로직 실행 전, WebGPU 사용 가능 여부를 안전하게 확인하기 위해 제공됩니다.
 * HOW: 내부에 저장된 `device` 변수가 null이 아닌지 불리언(Boolean) 값으로 평가하여 반환합니다.
 */
export function isAvailable(): boolean {
  return device !== null;
}

export function _resetDeviceForTesting(): void {
  device = null;
  adapter = null;
  if (onDeviceLostCallback) {
    onDeviceLostCallback();
  }
}

/**
 * WHAT: GPU 디바이스 연결이 끊어졌을 때(device lost) 호출될 콜백 함수를 등록합니다.
 * WHY: 예기치 못한 GPU 충돌이나 컨텍스트 상실 시 상위 계층(예: 파이프라인 캐시 무효화, 재초기화 로직)에 이를 알리기 위해 존재합니다.
 * HOW: 전달받은 함수(callback)를 모듈 레벨 변수인 `onDeviceLostCallback`에 할당하여 이후 디바이스 손실 이벤트 발생 시 실행될 수 있도록 합니다.
 */
export function setDeviceLostCallback(callback: () => void): void {
  onDeviceLostCallback = callback;
}

```

---

## `packages/forge/src/webgpu/pipelineCache.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * pipelineCache.ts — WGSL 컴파일 파이프라인 캐시
 *
 * L-03 Fix: clear() 메서드를 통해 device lost 시 캐시 무효화.
 * NL-02 Fix: 캐시 키에 WGSL 해시를 포함하여 동일 op명으로 다른 WGSL 지원.
 */

import { getDevice } from "./device";

/**
 * WHAT: 문자열 데이터를 기반으로 고유한 해시(32-bit 정수형 기반 16진수 문자열)를 생성하는 간단한 해시 함수입니다.
 * WHY: WGSL 소스 코드 문자열을 해시화하여 파이프라인 캐시 키(cache key)에 추가함으로써, 같은 이름의 연산(op)이라도 다른 WGSL 코드가 주어질 경우 충돌을 방지하기 위함입니다.
 * HOW: djb2 해시 알고리즘 변형을 사용합니다.
 *      초기값 5381에서 시작하여 문자열의 각 문자 코드를 순회(for 루프)하면서,
 *      비트 이동 및 XOR 연산을 통해 해시 값을 누적한 후 부호 없는 32비트 정수를 16진수 문자열로 변환하여 반환합니다.
 */
export function hashString(str: string): string {
  let hash = 5381;
  /**
   * WHAT: 문자열의 각 문자를 순회하며 해시값을 갱신하는 반복문입니다.
   * WHY: 문자열 전체의 데이터를 기반으로 고유성을 보장하는 해시값을 계산하기 위해 필요합니다.
   * HOW: 인덱스 i를 0부터 문자열 끝까지 증가시키며 현재 문자의 유니코드 값을 가져와 비트 연산으로 기존 해시값과 혼합합니다.
   */
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) ^ str.charCodeAt(i);
    hash = hash & hash; // 32bit integer
  }
  return (hash >>> 0).toString(16);
}

/**
 * WHAT: WebGPU 컴퓨트 파이프라인(GPUComputePipeline)과 셰이더 모듈(GPUShaderModule)을 저장하고 재사용하는 캐시 관리 클래스입니다.
 * WHY: WGSL 코드를 매번 파싱하고 컴파일하는 비용(오버헤드)을 줄여 GPU 연산 초기화 성능을 극대화하기 위해 존재합니다.
 * HOW: 내부적으로 Map 인스턴스를 유지하여 연산명(key)과 WGSL 해시의 조합을 캐시 키로 사용하고, 컴파일된 객체를 메모리에 저장 및 반환합니다.
 */
class PipelineCache {
  /**
   * WHAT: 컴파일 완료된 셰이더 모듈과 컴퓨트 파이프라인 객체를 키(문자열)에 매핑하여 보관하는 내부 저장소입니다.
   * WHY: 반복적인 연산 요청 시 동일한 코드가 주어지면 이전에 컴파일된 객체를 빠르게 찾아 반환하기 위해 필요합니다.
   * HOW: JavaScript 내장 Map 구조를 사용하여 생성되며, 캐시 적중(Cache Hit) 시 저장된 값을 제공하고, 누락(Cache Miss) 시 새 객체를 추가합니다.
   */
  private cache: Map<string, { shader: GPUShaderModule; pipeline: GPUComputePipeline }> =
    new Map();

  /**
   * WHAT: 주어진 연산 이름(key)과 WGSL 소스 코드를 바탕으로 컴파일된 파이프라인 객체와 셰이더 모듈을 반환합니다.
   * WHY: 기존에 컴파일된 캐시가 있다면 즉시 반환하여 성능을 최적화하고, 없다면 즉석에서(Synchronously) 새로 컴파일하기 위해 사용됩니다.
   * HOW: 연산 이름과 WGSL 해시를 조합해 캐시 키를 만든 후, 내부 캐시 맵에서 조회합니다. 없을 경우 WebGPU 디바이스에 셰이더 모듈과 파이프라인을 생성 요청하고, 결과를 캐시에 저장한 뒤 반환합니다.
   */
  getPipeline(
    key: string,
    wgslCode: string
  ): { shader: GPUShaderModule; pipeline: GPUComputePipeline } {
    const cacheKey = `${key}:${hashString(wgslCode)}`;
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    const device = getDevice();
    const shader = device.createShaderModule({ code: wgslCode });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: shader, entryPoint: "main" },
    });

    const entry = { shader, pipeline };
    this.cache.set(cacheKey, entry);
    return entry;
  }

  /**
   * WHAT: 애플리케이션 초기화 단계에서 지정된 여러 파이프라인을 비동기적으로 미리 컴파일하여 캐싱하는 웜업(Warmup) 기능입니다.
   * WHY: 첫 번째 GPU 연산 실행 시 발생하는 동기적 컴파일로 인한 UI 프리징(Freeze) 혹은 끊김(Stuttering) 현상을 방지하기 위해 존재합니다.
   * HOW: 입력받은 배열을 순회하며 아직 캐시되지 않은 항목만 추려낸 뒤, `createComputePipelineAsync`를 사용해 비동기로 병렬 컴파일을 수행(Promise.allSettled)하고 결과를 캐시에 저장합니다.
   */
  async warmup(entries: Array<{ key: string; wgslCode: string }>): Promise<void> {
    const device = getDevice();
    const pendingEntries = entries.filter(e => !this.cache.has(`${e.key}:${hashString(e.wgslCode)}`));
    const promises = pendingEntries.map(async (e) => {
        const cacheKey = `${e.key}:${hashString(e.wgslCode)}`;
        const shader = device.createShaderModule({ code: e.wgslCode });
        const pipeline = await device.createComputePipelineAsync({
          layout: "auto",
          compute: { module: shader, entryPoint: "main" },
        });
        this.cache.set(cacheKey, { shader, pipeline });
      });
    const results = await Promise.allSettled(promises);
    results.forEach((res, i) => {
      if (res.status === 'rejected') {
        console.warn(`[AMEVA] Warmup failed for ${pendingEntries[i].key}:`, res.reason);
      }
    });
  }

  /**
   * WHAT: 파이프라인 캐시 내에 저장된 모든 항목을 삭제하여 완전히 초기화합니다.
   * WHY: GPU 디바이스가 유실(device lost)되었거나 초기화가 다시 발생할 때, 이전 디바이스 컨텍스트를 가리키는 더 이상 유효하지 않은 파이프라인 참조를 제거하기 위함입니다.
   * HOW: 내부 맵(Map) 객체의 내장 메서드인 `clear()`를 호출하여 모든 키-값 쌍을 비웁니다.
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * WHAT: 현재 캐시에 저장된 파이프라인 객체의 총 개수를 반환하는 프로퍼티 접근자(Getter)입니다.
   * WHY: 메모리 사용량 모니터링이나 디버깅 시 캐시의 누적 상태를 파악하기 위해 제공됩니다.
   * HOW: 내부 캐시 맵의 `size` 프로퍼티 값을 그대로 반환합니다.
   */
  get size(): number {
    return this.cache.size;
  }
}

/**
 * WHAT: 전역에서 공유되는 단일 PipelineCache 인스턴스입니다.
 * WHY: 여러 텐서 연산 모듈이 개별 캐시를 만들지 않고 하나의 중앙 집중형 캐시를 재사용하여 메모리와 컴파일 비용을 최소화하기 위해 사용됩니다.
 * HOW: PipelineCache 클래스의 인스턴스를 하나 생성하여 모듈 외부로 노출(export)합니다.
 */
export const _globalPipelineCache = new PipelineCache();

```

---

## `packages/forge/src/webgpu/quota.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-18 13:45:00 +0900: Fix: Orphaned token state & QuotaSnapshot accounting
 *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 *   - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * quota.ts — VRAM 할당 쿼터 관리자
 *
 * C-06 Fix: quota release 타이밍 불일치 해결 — markPendingRelease + release 2단계.
 * H-04 Fix: setLimits()로 런타임에 동적 쿼터 설정 가능.
 * NH-04 Fix: markPendingRelease() 이중 dispose 시 카운터 음수 방지.
 */

import { AMEVAForgeQuotaExceededError } from "../errors";

/**
 * WHAT: GPU 메모리 할당의 목적이나 종류를 나타내는 리터럴 타입입니다.
 * WHY: 메모리가 어떤 용도로(tensor 데이터, staging 버퍼, uniform 등) 할당되었는지 추적하여 디버깅 및 프로파일링에 활용하기 위해 정의되었습니다.
 * HOW: 문자열 유니온 타입으로 선언되어 'tensor', 'staging', 'uniform', 'temporary' 중 하나의 값을 가집니다.
 */
export type AllocationKind = 'tensor' | 'staging' | 'uniform' | 'temporary';

/**
 * WHAT: 할당된 메모리 토큰의 현재 생명주기 상태를 나타내는 리터럴 타입입니다.
 * WHY: 메모리가 아직 사용 중인지(active), 해제가 예약되었는지(pending_release), 완전히 해제되었는지(released), 파괴 실패로 고아화되었는지(orphaned) 구분하여 안전한 리소스 관리를 보장하기 위함입니다.
 * HOW: 상태 전이를 명확하게 나타내기 위해 문자열 유니온을 사용합니다.
 */
export type AllocationState = 'active' | 'pending_release' | 'released' | 'orphaned';

/**
 * WHAT: 할당된 GPU 메모리 블록을 추적하는 메타데이터 객체(토큰) 클래스입니다.
 * WHY: 실제 GPU 버퍼 리소스와 매핑되어 해당 할당의 크기, 종류, 소유자 등을 식별하고 QuotaManager를 통한 반환을 제어하기 위해 존재합니다.
 * HOW: 생성자에서 유니크 ID, 크기(size), 종류, 그래프 소유자(ownerGraph) 및 생성 세대(generation)를 주입받아 초기화합니다.
 */
export class AllocationToken {
  public id: string;
  public size: number;
  public kind: AllocationKind;
  public state: AllocationState;
  public ownerGraph: string | null;
  public generation: number;

  /**
   * WHAT: AllocationToken 인스턴스를 초기화하는 생성자입니다.
   * WHY: 새로운 메모리 할당이 예약될 때 필요한 상태 및 식별 데이터를 객체에 부여하기 위해 호출됩니다.
   * HOW: 전달받은 파라미터로 멤버 변수를 초기화하며, 초기 상태는 'active'로 설정합니다.
   */
  constructor(id: string, size: number, kind: AllocationKind, ownerGraph: string | null, generation: number) {
    this.id = id;
    this.size = size;
    this.kind = kind;
    this.state = 'active';
    this.ownerGraph = ownerGraph;
    this.generation = generation;
  }
}

/**
 * WHAT: 시스템 전체의 GPU VRAM 할당량을 중앙 집중적으로 관리하는 클래스입니다.
 * WHY: WebGPU 애플리케이션에서 발생할 수 있는 Out-Of-Memory(OOM) 오류를 미연에 방지하고, 메모리 누수를 추적하며 동적으로 한계(Limits)를 설정하기 위해 필요합니다.
 * HOW: 소프트 제한(soft limit)과 하드 제한(hard limit)을 기반으로 메모리 할당 요청을 검사하고 허용하거나 거부하며, 할당된 토큰을 Map을 통해 상태별로 관리합니다.
 */
export class QuotaManager {
  /**
   * WHAT: 현재 활성 상태로 할당된 총 메모리 크기(바이트)입니다.
   * WHY: 사용 중인 리소스를 합산하여 쿼터를 초과하지 않는지 감시하기 위해 유지합니다.
   * HOW: reserveToken 시 증가하고, releaseToken 시 감소합니다.
   */
  public allocatedBytes: number = 0;
  /**
   * WHAT: 해제가 예약되었으나 아직 완전히 반환되지 않은 메모리 크기(바이트)입니다.
   * WHY: 비동기 작업 중 잠시 유지되는 메모리를 계산하여 여유 한계를 보다 정확하게 산정하기 위함입니다.
   * HOW: markPendingRelease 호출 시 증가하고, 완전히 해제될 때 감소합니다.
   */
  public pendingReleaseBytes: number = 0;
  /**
   * WHAT: destroy 실패 후 반환되지 못한 고아(Orphaned) 메모리 크기(바이트)입니다.
   * WHY: 장부 조작 없이 유령 누수 발생 시 명확히 회계에 기록하기 위함입니다.
   */
  public orphanedBytes: number = 0;
  /**
   * WHAT: 메모리 할당이 절대로 초과할 수 없는 최대 허용치(바이트)입니다.
   * WHY: 이 값을 초과하는 할당 요청을 즉시 차단하여 치명적인 시스템 충돌(OOM)을 막기 위해 설정됩니다.
   * HOW: 생성자에서 주입되거나 setLimits를 통해 설정됩니다.
   */
  public hardLimitBytes: number;
  /**
   * WHAT: 경고를 발생시키는 메모리 사용량의 임계점(바이트)입니다.
   * WHY: 하드 리밋에 도달하기 전 시스템에 과부하가 올 수 있음을 경고(warn)하기 위해 사용됩니다.
   * HOW: 실제 사용량(allocated - pending)이 이 값을 초과할 때 콘솔에 경고 로그를 출력합니다.
   */
  public softLimitBytes: number;
  
  /**
   * WHAT: 발급된 모든 AllocationToken을 고유 식별자(ID)로 관리하는 맵(Map)입니다.
   * WHY: 토큰의 무결성 검증, 이중 해제(Double Free) 방지 및 전체 할당 현황 조회를 위해 존재합니다.
   * HOW: 토큰 발급 시 추가하고 해제 시 삭제합니다.
   */
  private tokens = new Map<string, AllocationToken>();
  /**
   * WHAT: 새로 생성되는 메모리 토큰에 부여할 고유 식별자 카운터입니다.
   * WHY: 각 할당 토큰을 구별하고 맵에서 충돌 없이 관리하기 위해 필요합니다.
   * HOW: 새로운 토큰이 생성될 때마다 1씩 증가합니다.
   */
  private nextId = 1;
  /**
   * WHAT: 현재 할당 주기를 나타내는 세대(Generation) 카운터입니다.
   * WHY: 그래프 재컴파일 등 대규모 변경이 일어날 때 이전 세대의 토큰들을 구분하고 메모리 누수를 진단하기 위해 도입되었습니다.
   * HOW: incrementGeneration() 호출 시 증가하며 토큰 생성 시 부여됩니다.
   */
  private currentGeneration = 1;

  /**
   * WHAT: QuotaManager 클래스의 인스턴스를 초기화하는 생성자입니다.
   * WHY: 객체 생성 시 초기 하드 리밋과 소프트 리밋 용량을 설정하기 위해 호출됩니다.
   * HOW: 전달된 바이트 값을 각각의 클래스 프로퍼티에 할당합니다.
   */
  constructor(
    hardLimitBytes: number = 1 * 1024 * 1024 * 1024,
    softLimitBytes: number = 768 * 1024 * 1024
  ) {
    this.hardLimitBytes = hardLimitBytes;
    this.softLimitBytes = softLimitBytes;
  }

  /**
   * WHAT: 메모리 할당의 하드 리밋과 소프트 리밋을 동적으로 변경합니다.
   * WHY: 애플리케이션 실행 중 디바이스 환경에 따라 가용 메모리 한계를 유연하게 재조정하기 위해 사용됩니다.
   * HOW: 전달된 값이 유효한 양수인지, 소프트 리밋이 하드 리밋보다 작거나 같은지 검증한 후 내부 프로퍼티를 갱신합니다.
   */
  setLimits(hardLimitBytes: number, softLimitBytes: number): void {
    if (!Number.isSafeInteger(hardLimitBytes) || hardLimitBytes <= 0) {
      throw new AMEVAForgeQuotaExceededError(`Invalid hard limit: ${hardLimitBytes}`);
    }
    if (!Number.isSafeInteger(softLimitBytes) || softLimitBytes <= 0) {
      throw new AMEVAForgeQuotaExceededError(`Invalid soft limit: ${softLimitBytes}`);
    }
    if (softLimitBytes > hardLimitBytes) {
      throw new AMEVAForgeQuotaExceededError("softLimitBytes must be <= hardLimitBytes");
    }
    this.hardLimitBytes = hardLimitBytes;
    this.softLimitBytes = softLimitBytes;
  }

  /**
   * WHAT: 주어진 크기의 메모리 할당을 예약하고 추적용 토큰을 반환합니다.
   * WHY: 버퍼 생성 전에 쿼터 초과 여부를 먼저 검사하여, 한계 초과 시 안전하게 예외(AMEVAForgeQuotaExceededError)를 발생시키기 위함입니다.
   * HOW: 크기 무결성 검사 후 현재 가용 한계 내인지 확인하고, `allocatedBytes`를 증가시킨 뒤 소프트 리밋을 넘었는지 확인하여 경고합니다. 그 후 새 토큰 객체를 만들어 맵에 등록하고 반환합니다.
   */
  reserveToken(byteLength: number, kind: AllocationKind, ownerGraph: string | null = null): AllocationToken {
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
      throw new AMEVAForgeQuotaExceededError(`Invalid allocation size: ${byteLength}`);
    }
    if (byteLength > this.hardLimitBytes - this.allocatedBytes) {
      throw new AMEVAForgeQuotaExceededError(
        `Quota Exceeded: Cannot allocate ${byteLength} bytes. ` +
        `Current: ${this.allocatedBytes} (${this.pendingReleaseBytes} pending release), ` +
        `Limit: ${this.hardLimitBytes}`
      );
    }
    
    this.allocatedBytes += byteLength;
    if (this.allocatedBytes - this.pendingReleaseBytes > this.softLimitBytes) {
      console.warn(
        `[AMEVA] VRAM soft quota exceeded: ` +
        `${((this.allocatedBytes - this.pendingReleaseBytes) / 1e9).toFixed(2)}GB / ` +
        `${(this.softLimitBytes / 1e9).toFixed(2)}GB`
      );
    }
    
    const id = `alloc_${this.nextId++}`;
    const token = new AllocationToken(id, byteLength, kind, ownerGraph, this.currentGeneration);
    this.tokens.set(id, token);
    return token;
  }

  /**
   * WHAT: 특정 메모리 토큰을 곧 해제될 것('pending_release')으로 표시합니다.
   * WHY: GPU의 비동기 커맨드 실행이 완료되기 전까지는 버퍼를 파괴할 수 없으므로, 해당 시기를 유예(delay)하면서도 논리적으로는 해제 절차에 들어갔음을 명시하기 위해 존재합니다.
   * HOW: 토큰의 존재와 상태를 검증한 후 상태를 변경하고, `pendingReleaseBytes`에 토큰 크기를 합산합니다.
   */
  markPendingRelease(token: AllocationToken): void {
    if (!token || token.state !== 'active') return;
    
    // Verify token exists and belongs to us
    if (!this.tokens.has(token.id)) return;

    token.state = 'pending_release';
    
    const newPending = this.pendingReleaseBytes + token.size;
    this.pendingReleaseBytes = Math.min(newPending, this.allocatedBytes);
  }

  /**
   * WHAT: 메모리 토큰이 차지하던 용량을 쿼터 매니저에 완전히 반환하고 토큰을 해제('released') 상태로 바꿉니다.
   * WHY: GPU 리소스가 실제로 해제되었음을 반영하여 가용 메모리(allocatedBytes)를 줄이고 새로운 할당 요청을 수용할 수 있게 하기 위해 필요합니다.
   * HOW: 토큰 상태에 따라 `pendingReleaseBytes`와 `allocatedBytes`를 감소시키고, 토큰을 맵에서 제거합니다.
   */
  releaseToken(token: AllocationToken): void {
    if (!token || token.state === 'released') return;
    
    if (!this.tokens.has(token.id)) return;

    if (token.state === 'pending_release') {
      this.pendingReleaseBytes = Math.max(0, this.pendingReleaseBytes - token.size);
    }
    this.allocatedBytes = Math.max(0, this.allocatedBytes - token.size);
    token.state = 'released';
    this.tokens.delete(token.id);
  }

  /**
   * WHAT: 메모리 토큰을 고아(orphaned) 상태로 마킹합니다.
   * WHY: GPUBuffer.destroy() 실패 등으로 인해 장부상 회계 불일치(ghost leak)가 발생하지 않도록 명시적 기록을 남깁니다.
   */
  markOrphaned(token: AllocationToken, reason?: string): void {
    if (!token || token.state === 'orphaned' || token.state === 'released') return;
    if (!this.tokens.has(token.id)) return;

    if (token.state === 'pending_release') {
      this.pendingReleaseBytes = Math.max(0, this.pendingReleaseBytes - token.size);
    }
    this.orphanedBytes += token.size;
    token.state = 'orphaned';
    console.warn(`[QuotaManager] Token marked orphaned: ${token.id} (${token.size} bytes). Reason: ${reason || 'unknown'}`);
  }

  /**
   * WHAT: 현재 메모리 할당량, 대기량, 유효 사용량, 한계치 등의 쿼터 사용 현황을 묶어 반환합니다.
   * WHY: 프로파일러, 디버깅 도구 또는 UI에서 시스템의 메모리 점유 상태를 실시간으로 모니터링하기 위해 제공됩니다.
   * HOW: 클래스 내부에 유지 중인 통계 값들을 객체 형태로 복사하여 리턴합니다.
   */
  getUsage(): {
    allocatedBytes: number;
    pendingReleaseBytes: number;
    orphanedBytes: number;
    effectiveBytes: number;
    hardLimitBytes: number;
    softLimitBytes: number;
    activeTokens: number;
  } {
    return {
      allocatedBytes: this.allocatedBytes,
      pendingReleaseBytes: this.pendingReleaseBytes,
      orphanedBytes: this.orphanedBytes,
      effectiveBytes: this.allocatedBytes - this.pendingReleaseBytes,
      hardLimitBytes: this.hardLimitBytes,
      softLimitBytes: this.softLimitBytes,
      activeTokens: this.tokens.size
    };
  }

  /**
   * WHAT: 메모리 할당 관리의 세대(Generation) 카운터를 1 증가시킵니다.
   * WHY: 실행 그래프나 환경이 크게 전환되는 시점을 마킹하여, 이전 세대에서 생성되었으나 아직 해제되지 않은 누수(Leak) 토큰을 식별하기 위함입니다.
   * HOW: `currentGeneration` 변수에 1을 더합니다.
   */
  incrementGeneration(): void {
    this.currentGeneration++;
  }
  
  /**
   * WHAT: 현재 할당 관리의 세대 카운터 값을 반환합니다.
   * WHY: 외부 모듈에서 최신 세대 번호를 조회하여 할당 로직이나 상태 리포팅에 활용하기 위해 제공됩니다.
   * HOW: `currentGeneration` 프로퍼티 값을 반환합니다.
   */
  getGeneration(): number {
    return this.currentGeneration;
  }

  /**
   * WHAT: 모든 쿼터 통계치와 관리 중인 토큰을 초기 상태로 되돌립니다.
   * WHY: 테스트 사이의 격리(Isolation)를 보장하거나, 디바이스 초기화 시 이전 상태를 안전하게 파기하기 위해 존재합니다.
   * HOW: 바이트 카운터들을 0으로 설정하고, 토큰 맵을 비웁니다(clear).
   */
  reset(): void {
    this.allocatedBytes = 0;
    this.pendingReleaseBytes = 0;
    this.orphanedBytes = 0;
    this.tokens.clear();
  }

  /**
   * WHAT: 자가 치유(Self-Healing) 함수: 실제 살아있는 토큰들의 상태를 스캔하여 쿼터 통계를 완벽히 정합시킵니다.
   * WHY: 비동기 작업 예외나 누수로 인해 쿼터 카운터가 어긋났을 때 자동으로 카운터를 보정하기 위함입니다.
   */
  sanitizePendingBytes(): { repairedAllocated: number; repairedPending: number } {
    let actualAllocated = 0;
    let actualPending = 0;
    for (const [id, token] of this.tokens.entries()) {
      if (token.state === 'released') {
        this.tokens.delete(id);
      } else {
        actualAllocated += token.size;
        if (token.state === 'pending_release') {
          actualPending += token.size;
        }
      }
    }
    this.allocatedBytes = actualAllocated;
    this.pendingReleaseBytes = actualPending;
    return { repairedAllocated: actualAllocated, repairedPending: actualPending };
  }
}

/**
 * WHAT: 전역에서 사용할 수 있는 QuotaManager의 싱글톤 인스턴스입니다.
 * WHY: 애플리케이션 내의 다양한 모듈(버퍼 관리자, 텐서 객체 등)이 하나의 통일된 메모리 한계를 공유하고 갱신하도록 강제하기 위해 생성되었습니다.
 * HOW: QuotaManager를 기본값(1GB/768MB)으로 인스턴스화하여 내보냅니다(export).
 */
export const _globalQuotaManager = new QuotaManager();

export interface QuotaSnapshot {
  usedBytes: number;
  pendingBytes: number;
  orphanedBytes: number;
  maxBytes: number;
  activeTokens: number;
}

export function getQuotaSnapshot(): QuotaSnapshot {
  const usage = _globalQuotaManager.getUsage();
  return Object.freeze({
    usedBytes: usage.allocatedBytes,
    pendingBytes: usage.pendingReleaseBytes,
    orphanedBytes: usage.orphanedBytes,
    maxBytes: usage.hardLimitBytes,
    activeTokens: usage.activeTokens,
  });
}

```

---

## `packages/forge/src/webgpu/shaderGuard.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * shaderGuard.ts — WGSL 셰이더 보안 가드
 *
 * H-07 Fix: 화이트리스트에 모든 구현된 op 추가.
 * NH-07 Fix: 이 파일의 assertAllowedKernelName()을 graphExecutor.ts와 gpuCore.ts에서
 *   실제로 import하여 사용한다 (이전에는 데드 코드였음).
 */

import { AMEVAForgeSecurityError } from "../errors";

/**
 * WHAT: 셰이더 내에서 사용될 식별자(함수명, 변수명)가 안전한 문자열인지 검사합니다.
 * WHY: 영숫자와 밑줄(_) 이외의 문자가 주입되어 비정상적인 코드 실행이나 컴파일 에러를 유도하는 셰이더 인젝션 공격을 예방하기 위해 존재합니다.
 * HOW: 정규 표현식(/^[a-zA-Z_][a-zA-Z0-9_]*$/)을 사용하여 문자열 패턴을 검증하고, 실패 시 AMEVAForgeSecurityError를 던집니다.
 */
export function assertSafeShaderIdentifier(identifier: string): void {
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(identifier)) {
    throw new AMEVAForgeSecurityError(
      `Invalid shader identifier: "${identifier}". Only alphanumeric and underscore allowed.`
    );
  }
}

/**
 * WHAT: 셰이더에 주입되는 상수(숫자) 값이 유한한(finite) 숫자인지 확인합니다.
 * WHY: Infinity나 NaN과 같은 유효하지 않은 값이 셰이더 소스에 포함되어 GPU 연산 오류를 유발하는 것을 막기 위함입니다.
 * HOW: `Number.isFinite(value)`로 검사하고, 유한하지 않은 경우 예외를 발생시킵니다.
 */
export function assertAllowedShaderConstant(value: number): void {
  if (!Number.isFinite(value)) {
    throw new AMEVAForgeSecurityError(
      `Invalid shader constant: ${value}. Must be a finite number.`
    );
  }
}

/**
 * WHAT: 전달된 셰이더 소스 문자열에 동적 템플릿 리터럴 구문("${" 또는 "`")이 포함되어 있는지 검사합니다.
 * WHY: 신뢰할 수 없는 데이터가 셰이더 코드로 동적으로 삽입되는 인젝션 공격(Template Literal Injection)을 철저히 차단하기 위해 필요합니다.
 * HOW: 문자열의 `includes` 메서드를 통해 해당 패턴의 존재 여부를 검사하고, 발견될 경우 에러를 던집니다.
 */
export function assertStaticShaderSourceOnly(source: string): void {
  if (source.includes("${") || source.includes("`")) {
    throw new AMEVAForgeSecurityError(
      "Dynamic shader source interpolation is forbidden. Use uniform buffers for runtime values."
    );
  }
}

/**
 * WHAT: 보안 상 실행이 허용된 커널(연산) 이름들을 저장하는 화이트리스트(Set)입니다.
 * WHY: 허가되지 않은 임의의 커널 이름이 실행 경로로 주입되어 예상치 못한 셰이더 모듈이 생성되거나 호출되는 보안 취약점을 방어하기 위해 존재합니다.
 * HOW: Set 자료구조로 초기화하여 허용 목록을 빠르게 조회(has)할 수 있도록 합니다.
 * 
 * H-07/NH-07 Fix: 모든 구현된 커널 이름을 화이트리스트에 포함.
 * graphExecutor.ts의 ALLOWED_OPS와 반드시 동기화 유지.
 * 이 함수는 gpuCore.ts와 graphExecutor.ts에서 실제로 호출된다.
 */
let ALLOWED_KERNEL_NAMES = new Set([
  "matmul",
  "batched_matmul",
  "relu",
  "relu_backward",
  "add",
  "mul",
  "transpose",
  // v2.0: 학습 기능에 필요한 커널 추가 (VUL-001 Fix)
  "sub",
  "neg",
  "div",
  "exp",
  "log",
  "sigmoid",
  "tanh",
  "sigmoid_backward",
  "tanh_backward",
  "fill",
  "sum",
  "max",
  "sum_axis",
  "max_axis",
  "max_axis_backward",
  "axpy",
  "pad",
  "gather",
  "scatter",
  "cat",
  "where",
  "dropout",
  "maxpool2d",
  "avgpool2d",
  "im2col",
  "col2im",
  "permute",
  "matmul_bias_relu",
]);

/**
 * WHAT: 화이트리스트에 허용된 커널 이름들을 새롭게 등록(덮어쓰기)합니다.
 * WHY: 애플리케이션 초기화 단계 또는 플러그인 로드 시 동적으로 안전한 커널 목록을 확장하고 갱신할 수 있도록 유연성을 제공하기 위함입니다.
 * HOW: 제공된 Iterable 인터페이스(예: 배열)를 받아 새로운 Set 객체를 생성하고 `ALLOWED_KERNEL_NAMES` 변수를 갱신합니다.
 */
export function registerKernelNames(names: Iterable<string>): void {
  for (const name of names) {
    assertSafeShaderIdentifier(name);
    ALLOWED_KERNEL_NAMES.add(name);
  }
}

/**
 * WHAT: 요청된 커널 이름이 허용된 화이트리스트(ALLOWED_KERNEL_NAMES)에 존재하는지 검사합니다.
 * WHY: 그래프 실행기(graphExecutor)나 GPU 코어 모듈이 연산을 수행하기 직전, 허가되지 않은 커널 호출을 차단하기 위해 사용됩니다.
 * HOW: `Set.has(name)` 메서드를 사용하여 포함 여부를 확인하고 없으면 보안 예외(SecurityError)를 발생시킵니다.
 */
export function assertAllowedKernelName(name: string): void {
  if (!ALLOWED_KERNEL_NAMES.has(name)) {
    throw new AMEVAForgeSecurityError(
      `Unknown kernel name: "${name}". Allowed: ${[...ALLOWED_KERNEL_NAMES].join(", ")}`
    );
  }
}

/**
 * WHAT: 현재 설정된 커널 이름 화이트리스트(Set)의 읽기 전용 참조를 반환합니다.
 * WHY: 외부 모듈에서 화이트리스트의 구성을 확인할 수 있게 하면서도 직접적인 데이터 변조는 방지하기 위해 존재합니다.
 * HOW: 모듈 내부의 `ALLOWED_KERNEL_NAMES` 변수를 ReadonlySet 타입으로 캐스팅하여 그대로 반환합니다.
 */
export function getAllowedKernelNames(): ReadonlySet<string> {
  return ALLOWED_KERNEL_NAMES;
}

```

---

## `packages/forge/src/webgpu/uniformPool.ts`

```typescript
/**
 * uniformPool.ts - Transient Uniform Buffer Pool for GraphExecutor & Direct Ops
 * 
 * WHAT: 소형 유니폼 버퍼(Uniform Buffer, 16B~256B)를 고성능으로 재사용하는 전용 버퍼 풀입니다.
 * WHY: 그래프 실행 시 수십 개의 유니폼 버퍼를 매번 allocate/free하면서 onSubmittedWorkDone 지연으로 인해 발생하는 '가짜 OOM(Fake OOM)'을 원천 차단합니다.
 * HOW: 크기별 버킷(16, 32, 64, 112, 144, 256)으로 버퍼를 관리하며, GPU 작업 제출 후 fence 카운터를 통해 안전하게 재사용합니다.
 */
import { getDevice } from "./device";
import { allocateBuffer, freeBuffer } from "./buffers";
import { AllocationToken } from "./quota";

const UNIFORM_BUCKETS = [16, 32, 64, 112, 144, 256, 512, 1024];

export type UniformEntry = {
  buffer: GPUBuffer;
  token: AllocationToken;
  byteLength: number;
  inFlight: boolean;
  fenceId: number;
};

export class UniformBufferPool {
  private pools = new Map<number, UniformEntry[]>();
  private inFlight: UniformEntry[] = [];
  private fenceCounter = 0;

  acquire(byteLength: number): UniformEntry {
    const bucket = this.bucket(byteLength);
    const pool = this.pools.get(bucket) ?? [];

    const reusable = pool.pop();
    if (reusable) {
      reusable.inFlight = true;
      reusable.fenceId = this.fenceCounter;
      return reusable;
    }

    const usage = typeof GPUBufferUsage !== 'undefined'
      ? (GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      : (0x0040 | 0x0008);

    const { buffer, token } = allocateBuffer(
      bucket,
      usage,
      'uniform',
      'UniformBufferPool'
    );

    return {
      buffer,
      token,
      byteLength: bucket,
      inFlight: true,
      fenceId: this.fenceCounter,
    };
  }

  releaseAfterSubmit(entry: UniformEntry): void {
    this.inFlight.push(entry);
  }

  releaseSync(entry: UniformEntry): void {
    entry.inFlight = false;
    try {
      freeBuffer(entry.buffer, entry.token);
    } catch {}
  }

  inFlightBytes(): number {
    return this.inFlight.reduce((acc, e) => acc + e.byteLength, 0);
  }

  async retireSubmitted(device: GPUDevice): Promise<void> {
    const currentFence = ++this.fenceCounter;
    try {
      await device.queue.onSubmittedWorkDone();
    } catch {}

    const stillInFlight: UniformEntry[] = [];
    for (const entry of this.inFlight) {
      if (entry.fenceId < currentFence) {
        entry.inFlight = false;
        const pool = this.pools.get(entry.byteLength) ?? [];
        if (pool.length < 256) {
          pool.push(entry);
          this.pools.set(entry.byteLength, pool);
        } else {
          try { freeBuffer(entry.buffer, entry.token); } catch {}
        }
      } else {
        stillInFlight.push(entry);
      }
    }
    this.inFlight = stillInFlight;
  }

  clear(): void {
    for (const entries of this.pools.values()) {
      for (const entry of entries) {
        try { freeBuffer(entry.buffer, entry.token); } catch {}
      }
    }
    this.pools.clear();
    for (const entry of this.inFlight) {
      try { freeBuffer(entry.buffer, entry.token); } catch {}
    }
    this.inFlight = [];
  }

  private bucket(n: number): number {
    for (const b of UNIFORM_BUCKETS) {
      if (n <= b) return b;
    }
    return Math.ceil(n / 256) * 256;
  }
}

export const _globalUniformPool = new UniformBufferPool();

```

---

## `packages/forge/src/webgpu/validateWasmRange.ts`

```typescript
/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * 
 * WHAT: WASM(WebAssembly) 메모리 영역에 접근할 때 오프셋과 길이의 유효성을 검증하는 모듈입니다.
 * WHY: 잘못된 메모리 주소나 범위를 참조하여 발생하는 버퍼 오버플로우, 세그멘테이션 폴트 및 잠재적 보안 취약점을 차단하기 위해 필요합니다.
 */
import { AMEVAForgeSecurityError } from "../errors";

/**
 * WHAT: 주어진 오프셋과 데이터 길이가 WASM 선형 메모리 힙(heap)의 유효한 범위 내에 있는지 안전하게 검사합니다.
 * WHY: CPU-GPU 간 데이터 전송이나 공유 메모리 접근 시 악의적이거나 잘못된 크기 요청으로 인한 메모리 침범을 방어하기 위해 호출됩니다.
 * HOW: `Number.isSafeInteger`와 비음수(non-negative) 조건을 통해 입력 인자의 데이터 타입을 엄격히 검증한 후, `offset + byteLength`가 총 WASM 메모리 크기를 초과하지 않는지 계산하여 확인합니다. 위반 시 보안 예외를 던집니다.
 */
export function assertWasmRange(offset: number, byteLength: number, wasmByteLength: number): void {
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new AMEVAForgeSecurityError("Invalid offset: must be a non-negative safe integer.");
  }
  if (!Number.isSafeInteger(byteLength) || byteLength < 0) {
    throw new AMEVAForgeSecurityError("Invalid byteLength: must be a non-negative safe integer.");
  }
  if (!Number.isSafeInteger(wasmByteLength) || wasmByteLength < 0) {
    throw new AMEVAForgeSecurityError("Invalid wasmByteLength: must be a non-negative safe integer.");
  }

  if (offset > wasmByteLength || byteLength > wasmByteLength - offset) {
    throw new AMEVAForgeSecurityError("WASM memory range out of bounds");
  }
}

```

---

