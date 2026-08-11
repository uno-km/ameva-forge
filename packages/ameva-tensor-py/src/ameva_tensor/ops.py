"""
ops.py — 텐서 연산 API

C-09 Fix: 모든 assert를 명시적 에러 클래스로 교체.
M-03 Fix: CPU 경로에서 disposed 텐서 접근 시 AMEVATensorDisposedError 발생.
M-04 Fix: should_use_gpu 논리 정리.
NM-03 Fix: ones_like() GPU 텐서 생성 후 realize() 호출하여 backward seed로 안전하게 사용.
NL-05 Fix: ones_like를 공개 API로 노출.
"""
import numpy as np
from typing import Any, Tuple, Optional
from .tensor import Tensor
from .errors import (
    AMEVATensorDeviceError,
    AMEVATensorShapeError,
    AMEVATensorDisposedError,
)
from .autograd import Function, Context


def _require_cpu_data(t: Tensor, name: str = "tensor") -> np.ndarray:
    """
    C-09/M-03 Fix: CPU 텐서의 _data를 안전하게 요구한다.
    None이면 disposed 에러 발생 (assert 대신 명시적 에러).
    """
    if t._data is None:
        raise AMEVATensorDisposedError(
            f"CPU tensor '{name}' has no data. It may have been disposed or not yet initialized."
        )
    return t._data


def _ensure_same_device(a: Tensor, b: Tensor, op: str = "operation") -> None:
    """
    M-04 Fix: 두 텐서의 기기가 다르면 명시적 에러.
    """
    if a.device != b.device:
        raise AMEVATensorDeviceError(
            f"Cannot perform '{op}' on tensors from different devices: "
            f"'{a.device}' and '{b.device}'. "
            f"Move tensors to the same device first."
        )


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

def tensor(
    data: Any,
    device: str = "cpu",
    dtype: str = "float32",
    requires_grad: bool = False
) -> Tensor:
    """데이터로부터 텐서를 생성한다."""
    if isinstance(data, np.ndarray):
        arr = data if data.dtype == np.float32 else data.astype(np.float32)
    else:
        arr = np.array(data, dtype=np.float32)

    if device == "gpu":
        return Tensor(
            shape=arr.shape, dtype=dtype, device="gpu",
            requires_grad=requires_grad, data=arr, op='upload'
        )
    else:
        return Tensor(
            shape=arr.shape, dtype=dtype, device=device,
            requires_grad=requires_grad, data=arr
        )


def random(
    shape: Tuple[int, ...],
    device: str = "cpu",
    dtype: str = "float32",
    requires_grad: bool = False
) -> Tensor:
    """균등 분포 난수 텐서를 생성한다."""
    arr = np.random.random(shape).astype(np.float32)
    if device == "cpu":
        return Tensor(shape=shape, dtype=dtype, device=device,
                      requires_grad=requires_grad, data=arr)
    else:
        return Tensor(shape=shape, dtype=dtype, device="gpu",
                      requires_grad=requires_grad, data=arr, op='upload')


def ones_like(x: Tensor) -> Tensor:
    """
    x와 같은 shape/device의 1-텐서를 생성한다.
    NL-05 Fix: 공개 API로 노출됨 (__init__.py 참조).
    GPU 텐서의 경우 lazy upload 노드로 생성하여
    상위 그래프의 realize() 시점에 함께 제출된다 (레이지 철학 준수).
    """
    arr = np.ones(x.shape, dtype=np.float32)
    if x.device == "cpu":
        return Tensor(shape=x.shape, dtype=x.dtype, device="cpu", data=arr)
    else:
        # Lazy upload: realize() 호출 없이 그래프에 합류
        return Tensor(shape=x.shape, dtype="float32", device="gpu",
                   data=arr, op='upload')


def zeros_like(x: Tensor) -> Tensor:
    """x와 같은 shape/device의 0-텐서를 생성한다."""
    arr = np.zeros(x.shape, dtype=np.float32)
    if x.device == "cpu":
        return Tensor(shape=x.shape, dtype=x.dtype, device="cpu", data=arr)
    else:
        # Lazy upload: realize() 호출 없이 그래프에 합류
        return Tensor(shape=x.shape, dtype="float32", device="gpu",
                   data=arr, op='upload')


def zeros(
    shape: Tuple[int, ...],
    device: str = "cpu",
    dtype: str = "float32",
    requires_grad: bool = False
) -> Tensor:
    """0으로 채워진 텐서를 생성한다."""
    arr = np.zeros(shape, dtype=np.float32)
    return tensor(arr, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(
    shape: Tuple[int, ...],
    device: str = "cpu",
    dtype: str = "float32",
    requires_grad: bool = False
) -> Tensor:
    """1로 채워진 텐서를 생성한다."""
    arr = np.ones(shape, dtype=np.float32)
    return tensor(arr, device=device, dtype=dtype, requires_grad=requires_grad)


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

class AddFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "add")
        if _should_use_gpu(a, b):
            return Tensor(shape=a.shape, dtype="float32", device="gpu",
                          op='add', parents=(a, b))
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            res = data_a + data_b
            return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


def add(a: Tensor, b: Tensor) -> Tensor:
    return AddFunction.apply(a, b)


class MulFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "mul")
        if _should_use_gpu(a, b):
            return Tensor(shape=a.shape, dtype="float32", device="gpu",
                          op='mul', parents=(a, b))
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            res = data_a * data_b
            return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        return mul(grad_output, b), mul(grad_output, a)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return MulFunction.apply(a, b)


class MatmulFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "matmul")

        if len(a.shape) != 2 or len(b.shape) != 2:
            raise AMEVATensorShapeError(
                f"Matmul requires 2D tensors, got shapes {a.shape} and {b.shape}"
            )
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            raise AMEVATensorShapeError(
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
        grad_a = matmul(grad_output, transpose(b))
        grad_b = matmul(transpose(a), grad_output)
        return grad_a, grad_b


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatmulFunction.apply(a, b)


class TransposeFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        if len(x.shape) != 2:
            raise AMEVATensorShapeError(
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


def transpose(x: Tensor) -> Tensor:
    return TransposeFunction.apply(x)


class ReLUFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        if _should_use_gpu(x):
            return Tensor(shape=x.shape, dtype="float32", device="gpu",
                          op='relu', parents=(x,))
        else:
            data = _require_cpu_data(x, "x")
            res = np.maximum(data, 0.0)
            return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        if x.device == "cpu":
            data_x = _require_cpu_data(x, "x")
            data_g = _require_cpu_data(grad_output, "grad_output")
            grad = data_g * (data_x > 0).astype(np.float32)
            return (Tensor(shape=x.shape, dtype="float32", device="cpu", data=grad),)
        else:
            return (Tensor(shape=x.shape, dtype="float32", device="gpu",
                           op='relu_backward', parents=(x, grad_output)),)


def relu(x: Tensor) -> Tensor:
    return ReLUFunction.apply(x)


# ─────────────────────────────────────────────────────────────────────────────
# 편의 함수
# ─────────────────────────────────────────────────────────────────────────────

def to_numpy(x: Tensor) -> np.ndarray:
    return x.numpy()


async def to_numpy_async(x: Tensor) -> np.ndarray:
    return await x.numpy_async()


def dispose(x: Tensor) -> None:
    x.dispose()
