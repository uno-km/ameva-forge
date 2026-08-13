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
)

# WHAT: 자동 미분(autograd) 구현을 위한 베이스 클래스들을 임포트합니다.
# WHY: 각 수학 연산이 순전파와 역전파를 지원하는 연산 노드로 동작하게 만들기 위함입니다.
# HOW: 모든 연산 클래스는 Function을 상속받고 forward/backward에서 Context(ctx)를 사용합니다.
from .autograd import Function, Context

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
            return Tensor(shape=out_shape, dtype="float32", device="gpu",
                          op='add', parents=(a, b))
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
            return Tensor(shape=out_shape, dtype="float32", device="gpu",
                          op='mul', parents=(a, b))
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
            return Tensor(shape=out_shape, dtype='float32', device='gpu', op='sub', parents=(a, b))
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
            # WHAT: GPU에서는 별도의 mean 커널 없이 기존 sum과 div를 조합하여 처리합니다.
            # WHY: 중복된 커널 구현을 피하고 기존 최적화된 연산들을 재사용하기 위함입니다.
            # HOW: sum_op 결과를 원소 수 n 텐서로 나눕니다.
            s = sum_op(x)
            return div(s, tensor(np.array([float(n)], dtype=np.float32), device=x.device))
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
        # WHAT: 대상 축과 원래 형태를 저장합니다.
        # WHY: 역전파 시 줄어든 차원을 다시 복구(unsqueeze)하여 기울기를 흩뿌려주기 위함입니다.
        # HOW: 컨텍스트 객체에 할당합니다.
        ctx.axis = axis
        ctx.input_shape = x.shape
        
        if x.device == 'gpu':
            # WHAT: 결과 텐서의 모양(shape)을 계산합니다.
            # WHY: 축소된 차원이 제거된 새로운 shape 튜플을 만들기 위함입니다.
            # HOW: 리스트 변환 후 해당 인덱스를 지우고 다시 튜플로 만듭니다.
            new_shape = list(x.shape)
            del new_shape[axis]
            new_shape = tuple(new_shape) if new_shape else ()
            
            # WHAT: GPU 커널이 사용할 다차원 합산 파라미터(stride)를 계산합니다.
            # WHY: 1차원 선형 배열 형태인 VRAM 데이터를 특정 차원 간격으로 순회하며 합쳐야 하기 때문입니다.
            # HOW: 축 바깥쪽 크기(outer_size)와 축 안쪽 간격(inner_stride)을 구하여 op_params로 넘깁니다.
            outer_size = 1
            for i in range(axis):
                outer_size *= x.shape[i]
            reduction_size = x.shape[axis]
            inner_stride = 1
            for i in range(axis + 1, len(x.shape)):
                inner_stride *= x.shape[i]
                
            return Tensor(shape=new_shape, dtype='float32', device='gpu',
                         op='sum_axis', parents=(x,), op_params=[reduction_size, inner_stride])
        else:
            data = np.sum(_require_cpu_data(x, 'x'), axis=axis)
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
        ctx.axis = axis
        data = _require_cpu_data(x, "x")
        # WHAT: 주어진 축을 따라 최댓값을 구합니다.
        # WHY: 역전파 시 마스킹 비교에 사용하기 위함입니다.
        # HOW: np.max를 적용하고 컨텍스트에 기록합니다.
        m = np.max(data, axis=axis)
        ctx.max_val = m
        return Tensor(shape=m.shape, dtype="float32", device="cpu", data=m)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        data = _require_cpu_data(x, "x")
        grad = _require_cpu_data(grad_output, "grad")
        
        # WHAT: 브로드캐스팅을 위해 차원을 하나 늘립니다.
        # WHY: 축소되었던 차원(axis)을 다시 복원하여 원본 텐서와 비교 연산을 하기 위함입니다.
        # HOW: np.expand_dims를 사용합니다.
        m_exp = np.expand_dims(ctx.max_val, axis=ctx.axis)
        grad_exp = np.expand_dims(grad, axis=ctx.axis)
        
        # WHAT: 최댓값이 있던 위치의 마스크를 생성하고 동일 값 처리(평균화)를 합니다.
        # WHY: 중복된 최대값에 대해 기울기가 올바르게 분산되도록 하기 위함입니다.
        # HOW: np.sum으로 개수를 세고, np.divide로 나눕니다(0으로 나누는 것 방지).
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
        ctx.save_for_backward(x)
        if _should_use_gpu(x):
            # WHAT: GPU 에러 처리입니다.
            # WHY: 아직 GPU에서 제곱근 커널이 구현되지 않았기 때문입니다.
            # HOW: AMEVAForgeDeviceError를 발생시킵니다.
            from .errors import AMEVAForgeDeviceError
            raise AMEVAForgeDeviceError("GPU sqrt (pow) kernel not registered.")
        else:
            data = _require_cpu_data(x, "x")
            res = np.sqrt(data)
            return Tensor(shape=x.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # WHAT: 제곱근의 미분값 계산입니다.
        # WHY: f(x) = sqrt(x) 이면 df/dx = 1 / (2 * sqrt(x)) 이기 때문입니다.
        # HOW: 값이 2인 텐서를 만들어 분모를 구성하고 나눕니다.
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
        
        if grad_output.device == 'cpu':
            g = _require_cpu_data(grad_output, 'grad_output')
        else:
            # WHAT: 슬라이싱을 위해 임시로 CPU로 폴백(Fallback)하는 처리입니다.
            # WHY: 아직 GPU 상의 슬라이싱/cat 백워드 커널이 최적화되지 않았기 때문입니다.
            # HOW: numpy()를 호출해 데이터를 내립니다.
            g = grad_output.numpy()
            
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
        
        if grad_output.device == 'cpu':
            return (Tensor(shape=a.shape, dtype="float32", device="cpu", data=ga),
                    Tensor(shape=b.shape, dtype="float32", device="cpu", data=gb))
        else:
            return (tensor(ga, device="gpu"), tensor(gb, device="gpu"))

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

# WHAT: 조건 텐서에 따라 참이면 x, 거짓이면 y 요소를 선택하는 연산입니다.
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
            
            # WHAT: GPU 커널이 사용할 파라미터 리스트를 고정된 포맷(8차원 패딩)으로 만듭니다.
            # WHY: C++ 기반 커널이 일관된 구조로 데이터를 파싱할 수 있게 하기 위함입니다.
            # HOW: 리스트 안에 stride와 원래 형태 정보를 이어 붙입니다.
            op_params = [
                0, dim, rank, 0,
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
        # WHAT: 수집(Gather)의 역전파 연산입니다.
        # WHY: 추출해온 위치에 미분값을 정확히 되돌려놓아야 하기 때문입니다.
        # HOW: scatter 연산을 이용해 x와 같은 크기의 0 텐서에 grad_output을 뿌립니다.
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
# HOW: ScatterFunction.apply를 호출합니다.
def scatter(x: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
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
        x, = ctx.saved_tensors
        # WHAT: GPU 텐서이더라도 현재 백엔드는 CPU로 넘겨 계산합니다.
        # WHY: GPU 전용 풀링 역전파 커널이 미구현 상태이므로, Fallback 전략을 쓰기 때문입니다.
        # HOW: numpy() 코루틴이 아닌 일반 메서드를 호출해 CPU 메모리로 복사합니다.
        grad_out_np = grad_output.numpy()
        x_np = x.numpy()
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
        x, = ctx.saved_tensors
        # WHAT: 평균 풀링의 CPU 기반 역전파입니다.
        # WHY: 평균 풀링은 미분 시 각 출력에 대한 그래디언트를 원래 윈도우의 모든 픽셀에 동일하게(1/면적 비율로) 나눠주어야 하기 때문입니다.
        # HOW: grad_out_np를 넓이로 나누어 윈도우 크기로 복사해 누적 더하기를 수행합니다.
        grad_out_np = grad_output.numpy()
        x_np = x.numpy()
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
        # HOW: 마스크를 None으로 설정하고 원본을 리턴합니다.
        if not training or p == 0.0:
            ctx.save_for_backward(x)
            ctx.mask = None
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
# HOW: DropoutFunction.apply를 호출합니다.
def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return DropoutFunction.apply(x, p, training)

# WHAT: 임베딩(Embedding) 룩업을 수행하는 클래스입니다.
# WHY: 단어 인덱스 같은 정수 배열을 받아 밀집 벡터(Dense Vector) 공간의 실수 가중치로 변환하기 위함입니다.
# HOW: numpy 팬시 인덱싱(data_w[data_i])을 사용하고, 미분 시 np.add.at을 통해 추출된 위치에 기울기를 누적합니다.
class EmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, weight: Tensor, index: Tensor) -> Tensor:
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
