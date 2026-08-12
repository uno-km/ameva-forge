"""
ops.py — 텐서 연산 API

C-09 Fix: 모든 assert를 명시적 에러 클래스로 교체.
M-03 Fix: CPU 경로에서 disposed 텐서 접근 시 AMEVAForgeDisposedError 발생.
M-04 Fix: should_use_gpu 논리 정리.
Lazy Fix: ones_like() GPU 텐서는 lazy upload 노드로 생성 (realize 호출 없음).
NL-05 Fix: ones_like를 공개 API로 노출.
"""
import numpy as np
from typing import Any, Tuple, Optional
from .tensor import Tensor
from .errors import (
    AMEVAForgeDeviceError,
    AMEVAForgeShapeError,
    AMEVAForgeDisposedError,
)
from .autograd import Function, Context

# ─── Debug Mode ──────────────────────────────────────────────────────────────
# VUL-003/004: NumPy/PyTorch 기본 동작 유지, debug mode에서만 경고 활성화
_debug_mode: bool = False

def set_debug_mode(enabled: bool = True) -> None:
    """디버그 모드를 설정한다. 활성화 시 div-by-zero, log-of-non-positive 등을 경고한다."""
    global _debug_mode
    _debug_mode = enabled

def get_debug_mode() -> bool:
    return _debug_mode


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
        out_shape = _broadcast_shapes(a.shape, b.shape)
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
        return _unbroadcast(grad_output, ctx.a_shape), _unbroadcast(grad_output, ctx.b_shape)


def add(a: Tensor, b: Tensor) -> Tensor:
    return AddFunction.apply(a, b)


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
        a, b = ctx.saved_tensors
        return _unbroadcast(mul(grad_output, b), ctx.a_shape), _unbroadcast(mul(grad_output, a), ctx.b_shape)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return MulFunction.apply(a, b)


class MatmulFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "matmul")

        if len(a.shape) != 2 or len(b.shape) != 2:
            raise AMEVAForgeShapeError(
                f"Matmul requires 2D tensors, got shapes {a.shape} and {b.shape}"
            )
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
        return _unbroadcast(grad_output, ctx.a_shape), _unbroadcast(neg(grad_output), ctx.b_shape)


def sub(a: Tensor, b: Tensor) -> Tensor:
    return SubFunction.apply(a, b)


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
        return (neg(grad_output),)


def neg(x: Tensor) -> Tensor:
    return NegFunction.apply(x)


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
        grad_a = div(grad_output, b)
        grad_b = neg(mul(div(grad_output, mul(b, b)), a))  # -grad * a / b^2
        return _unbroadcast(grad_a, ctx.a_shape), _unbroadcast(grad_b, ctx.b_shape)


def div(a: Tensor, b: Tensor) -> Tensor:
    return DivFunction.apply(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# 편의 함수
# ─────────────────────────────────────────────────────────────────────────────

def to_numpy(x: Tensor) -> np.ndarray:
    return x.numpy()


async def to_numpy_async(x: Tensor) -> np.ndarray:
    return await x.numpy_async()


def dispose(x: Tensor) -> None:
    x.dispose()


def _broadcast_shapes(a_shape, b_shape):
    ndim = max(len(a_shape), len(b_shape))
    a_padded = (1,) * (ndim - len(a_shape)) + a_shape
    b_padded = (1,) * (ndim - len(b_shape)) + b_shape
    result = []
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

def _unbroadcast(grad, target_shape):
    if grad.shape == target_shape:
        return grad
    # Sum along broadcast dimensions
    ndim_diff = len(grad.shape) - len(target_shape)
    padded = (1,) * ndim_diff + target_shape
    
    # For CPU: use numpy directly
    if grad.device == 'cpu':
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
        # For GPU: compose with sum_axis ops
        result = grad
        axes = []
        for i, (g, t) in enumerate(zip(grad.shape, padded)):
            if t == 1 and g != 1:
                axes.append(i)
        for i in range(ndim_diff):
            if i not in axes:
                axes.append(i)
        axes = sorted(set(axes))
        
        # GPU sum_axis sequentially reduces an axis, changing the shape.
        # Reduce axes from back to front to avoid index shifting.
        for ax in reversed(axes):
            result = sum_axis(result, axis=ax)
            
        return reshape(result, target_shape) if result.shape != target_shape else result

class SumFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ctx.input_shape = x.shape
        if _should_use_gpu(x):
            return Tensor(shape=(), dtype='float32', device='gpu', op='sum', parents=(x,))
        else:
            return Tensor(shape=(), dtype='float32', device='cpu',
                         data=np.array(np.sum(_require_cpu_data(x, 'x')), dtype=np.float32))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # grad of sum is ones_like(input) * grad_output
        return (mul(grad_output, ones(x.shape, device=x.device)),)

def sum_op(x): return SumFunction.apply(x)


class MeanFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        n = 1
        for d in x.shape:
            n *= d
        ctx.numel = n
        if _should_use_gpu(x):
            # mean = sum / n, compose with existing ops
            s = sum_op(x)
            return div(s, tensor(np.array([float(n)], dtype=np.float32)))
        else:
            return Tensor(shape=(), dtype='float32', device='cpu',
                         data=np.array(np.mean(_require_cpu_data(x, 'x')), dtype=np.float32))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        n = ctx.numel
        return (mul(grad_output, full(x.shape, 1.0/n, device=x.device)),)

def mean_op(x): return MeanFunction.apply(x)


class ExpFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if _should_use_gpu(x):
            result = Tensor(shape=x.shape, dtype='float32', device='gpu', op='exp', parents=(x,))
        else:
            result = Tensor(shape=x.shape, dtype='float32', device='cpu',
                          data=np.exp(_require_cpu_data(x, 'x')))
        ctx.save_for_backward(result)  # save output for backward
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        exp_x, = ctx.saved_tensors
        return (mul(grad_output, exp_x),)  # d/dx exp(x) = exp(x)

def exp_op(x): return ExpFunction.apply(x)


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
        return (div(grad_output, x),)  # d/dx log(x) = 1/x

def log_op(x): return LogFunction.apply(x)


class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if _should_use_gpu(x):
            result = Tensor(shape=x.shape, dtype='float32', device='gpu', op='sigmoid', parents=(x,))
        else:
            data = 1.0 / (1.0 + np.exp(-_require_cpu_data(x, 'x')))
            result = Tensor(shape=x.shape, dtype='float32', device='cpu', data=data)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        sig, = ctx.saved_tensors
        # sigmoid'(x) = sig * (1 - sig)
        one_minus = sub(ones(sig.shape, device=sig.device), sig)
        return (mul(grad_output, mul(sig, one_minus)),)

def sigmoid(x): return SigmoidFunction.apply(x)


class TanhFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if _should_use_gpu(x):
            result = Tensor(shape=x.shape, dtype='float32', device='gpu', op='tanh', parents=(x,))
        else:
            result = Tensor(shape=x.shape, dtype='float32', device='cpu',
                          data=np.tanh(_require_cpu_data(x, 'x')))
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        tanh_x, = ctx.saved_tensors
        # tanh'(x) = 1 - tanh²(x)
        one_minus_sq = sub(ones(tanh_x.shape, device=tanh_x.device), mul(tanh_x, tanh_x))
        return (mul(grad_output, one_minus_sq),)

def tanh_op(x): return TanhFunction.apply(x)


class ReshapeFunction(Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.original_shape = x.shape
        if x.device == 'gpu':
            # For GPU: create a new Tensor with same handle but different shape
            # This is a metadata-only operation
            return Tensor(shape=new_shape, dtype=x.dtype, device='gpu', op='reshape', parents=(x,),
                         op_params=list(new_shape))
        else:
            data = _require_cpu_data(x, 'x').reshape(new_shape)
            return Tensor(shape=new_shape, dtype='float32', device='cpu', data=data)
    
    @staticmethod
    def backward(ctx, grad_output):
        return (reshape(grad_output, ctx.original_shape),)

def reshape(x, new_shape):
    if isinstance(new_shape, list):
        new_shape = tuple(new_shape)
    return ReshapeFunction.apply(x, new_shape)


class SumAxisFunction(Function):
    @staticmethod
    def forward(ctx, x, axis):
        ctx.save_for_backward(x)
        ctx.axis = axis
        ctx.input_shape = x.shape
        
        if x.device == 'gpu':
            # GPU: emit sum_axis lazy node
            new_shape = list(x.shape)
            del new_shape[axis]
            new_shape = tuple(new_shape) if new_shape else ()
            return Tensor(shape=new_shape, dtype='float32', device='gpu',
                         op='sum_axis', parents=(x,), op_params=[x.shape[axis], 
                         1 if len(x.shape) == 1 else (x.shape[1] if axis == 0 else x.shape[0])])
        else:
            data = np.sum(_require_cpu_data(x, 'x'), axis=axis)
            return Tensor(shape=data.shape, dtype='float32', device='cpu', data=data)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Expand grad back to original shape
        # This is the reverse: unsqueeze + broadcast
        if grad_output.device == 'cpu':
            data = _require_cpu_data(grad_output, 'grad')
            expanded = np.expand_dims(data, axis=ctx.axis)
            tiled = np.broadcast_to(expanded, ctx.input_shape).copy()
            return (Tensor(shape=ctx.input_shape, dtype='float32', device='cpu', data=tiled),)
        else:
            # VUL-013 Fix: GPU 경로 — reshape + broadcast로 gradient 확장
            # grad_output shape: input_shape에서 axis 차원이 제거된 shape
            # target: ctx.input_shape
            # 방법: ones(input_shape) * reshape(grad, broadcast-compatible shape)
            axis = ctx.axis
            input_shape = ctx.input_shape
            # grad를 broadcast-compatible shape으로 reshape
            expand_shape = list(input_shape)
            expand_shape[axis] = 1
            grad_reshaped = reshape(grad_output, tuple(expand_shape))
            # ones_like(input) * reshaped_grad → broadcast로 확장
            broadcast_ones = ones(input_shape, device='gpu')
            return (mul(broadcast_ones, grad_reshaped),)

def sum_axis(x, axis=0): return SumAxisFunction.apply(x, axis)

def randn(shape, device='cpu', dtype='float32', requires_grad=False):
    arr = np.random.randn(*shape).astype(np.float32)
    if device == 'cpu':
        return Tensor(shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, data=arr)
    else:
        return Tensor(shape=shape, dtype=dtype, device='gpu', requires_grad=requires_grad, data=arr, op='upload')

def unsqueeze(x: Tensor, dim: int) -> Tensor:
    shape = list(x.shape)
    if dim < 0:
        dim += len(shape) + 1
    shape.insert(dim, 1)
    return reshape(x, tuple(shape))

def squeeze(x: Tensor, dim: Optional[int] = None) -> Tensor:
    shape = list(x.shape)
    if dim is not None:
        if dim < 0:
            dim += len(shape)
        if shape[dim] == 1:
            shape.pop(dim)
    else:
        shape = [s for s in shape if s != 1]
    return reshape(x, tuple(shape))

def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    shape = list(x.shape)
    if end_dim < 0:
        end_dim += len(shape)
    if start_dim < 0:
        start_dim += len(shape)
    if start_dim > end_dim:
        return x
    new_shape = shape[:start_dim]
    flat_size = 1
    for s in shape[start_dim:end_dim+1]:
        flat_size *= s
    new_shape.append(flat_size)
    new_shape.extend(shape[end_dim+1:])
    return reshape(x, tuple(new_shape))

class PermuteFunction(Function):
    @staticmethod
    def forward(ctx, x, dims):
        ctx.save_for_backward(x)
        ctx.dims = dims
        data = _require_cpu_data(x, "x")
        res = np.transpose(data, dims)
        return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        inv_dims = [0] * len(ctx.dims)
        for i, d in enumerate(ctx.dims):
            inv_dims[d] = i
        return (permute(grad_output, tuple(inv_dims)),)

def permute(x: Tensor, dims: tuple) -> Tensor:
    return PermuteFunction.apply(x, dims)

class MaxFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        data = _require_cpu_data(x, "x")
        m = np.max(data)
        ctx.max_val = m
        return Tensor(shape=(), dtype="float32", device="cpu", data=np.array(m, dtype=np.float32))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        data = _require_cpu_data(x, "x")
        grad = _require_cpu_data(grad_output, "grad")
        res_grad = grad * (data == ctx.max_val).astype(np.float32)
        return (Tensor(shape=x.shape, dtype="float32", device="cpu", data=res_grad),)

def max_op(x: Tensor) -> Tensor:
    return MaxFunction.apply(x)

class MaxAxisFunction(Function):
    @staticmethod
    def forward(ctx, x, axis):
        ctx.save_for_backward(x)
        ctx.axis = axis
        data = _require_cpu_data(x, "x")
        m = np.max(data, axis=axis)
        ctx.max_val = m
        return Tensor(shape=m.shape, dtype="float32", device="cpu", data=m)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        data = _require_cpu_data(x, "x")
        grad = _require_cpu_data(grad_output, "grad")
        m_exp = np.expand_dims(ctx.max_val, axis=ctx.axis)
        grad_exp = np.expand_dims(grad, axis=ctx.axis)
        res_grad = grad_exp * (data == m_exp).astype(np.float32)
        return (Tensor(shape=x.shape, dtype="float32", device="cpu", data=res_grad),)

def max_axis(x: Tensor, axis: int) -> Tensor:
    return MaxAxisFunction.apply(x, axis)

def mean_axis(x: Tensor, axis: int) -> Tensor:
    s = sum_axis(x, axis)
    n = x.shape[axis]
    return div(s, tensor(np.array(n, dtype=np.float32), device=x.device))

def var(x: Tensor, axis=None, unbiased=True) -> Tensor:
    if axis is None:
        m = mean_op(x)
        diff = sub(x, m)
        diff_sq = mul(diff, diff)
        s = sum_op(diff_sq)
        n = x.numel()
        denom = n - 1 if unbiased and n > 1 else n
        return div(s, tensor(np.array(denom, dtype=np.float32), device=x.device))
    else:
        m = mean_axis(x, axis)
        m_unsq = unsqueeze(m, axis)
        diff = sub(x, m_unsq)
        diff_sq = mul(diff, diff)
        s = sum_axis(diff_sq, axis)
        n = x.shape[axis]
        denom = n - 1 if unbiased and n > 1 else n
        return div(s, tensor(np.array(denom, dtype=np.float32), device=x.device))

class SqrtFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if _should_use_gpu(x):
            return Tensor(shape=x.shape, dtype="float32", device="gpu", op='pow', parents=(x,), op_params=[0.5])
        else:
            data = _require_cpu_data(x, "x")
            res = np.sqrt(data)
            return Tensor(shape=x.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        two = full(x.shape, 2.0, device=x.device)
        return (div(grad_output, mul(two, sqrt(x))),)

def sqrt(x: Tensor) -> Tensor:
    return SqrtFunction.apply(x)

def std(x: Tensor, axis=None, unbiased=True) -> Tensor:
    return sqrt(var(x, axis=axis, unbiased=unbiased))

class CatFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor, dim: int = 0) -> Tensor:
        ctx.save_for_backward(a, b)
        ctx.dim = dim
        _ensure_same_device(a, b, "cat")
        
        shape_a = list(a.shape)
        shape_b = list(b.shape)
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
            # Fallback to CPU for slicing
            g = grad_output.numpy()
            
        slc_a = [slice(None)] * len(g.shape)
        slc_a[dim] = slice(0, a.shape[dim])
        slc_b = [slice(None)] * len(g.shape)
        slc_b[dim] = slice(a.shape[dim], None)
        
        ga = g[tuple(slc_a)]
        gb = g[tuple(slc_b)]
        
        ga = np.ascontiguousarray(ga)
        gb = np.ascontiguousarray(gb)
        
        if grad_output.device == 'cpu':
            return (Tensor(shape=a.shape, dtype="float32", device="cpu", data=ga),
                    Tensor(shape=b.shape, dtype="float32", device="cpu", data=gb))
        else:
            return (tensor(ga, device="gpu"), tensor(gb, device="gpu"))

def cat(tensors: list, dim: int = 0) -> Tensor:
    if len(tensors) < 1:
        raise ValueError("cat requires at least 1 tensor")
    if len(tensors) == 1:
        return tensors[0]
    res = tensors[0]
    for t in tensors[1:]:
        res = CatFunction.apply(res, t, dim)
    return res

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
            res = np.where(c > 0, data_x, data_y)
            return Tensor(shape=out_shape, dtype="float32", device="cpu", data=res)
            
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, Tensor, Tensor]:
        condition, x, y = ctx.saved_tensors
        zero_grad = zeros_like(grad_output)
        grad_x = where(condition, grad_output, zero_grad)
        grad_y = where(condition, zero_grad, grad_output)
        return (None, grad_x, grad_y)

def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return WhereFunction.apply(condition, x, y)

class PadFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, pad: Tuple[int, ...], mode='constant', value=0.0) -> Tensor:
        ctx.save_for_backward(x)
        ctx.pad = pad
        ctx.mode = mode
        ctx.value = value
        
        out_shape = list(x.shape)
        rank = len(x.shape)
        pad_pairs = []
        for i in range(rank):
            pad_before = pad[-(i * 2 + 2)] if len(pad) >= (i * 2 + 2) else 0
            pad_after = pad[-(i * 2 + 1)] if len(pad) >= (i * 2 + 1) else 0
            pad_pairs.insert(0, (pad_before, pad_after))
            out_shape[i] += pad_before + pad_after
            
        out_shape = tuple(out_shape)
        if _should_use_gpu(x):
            def get_strides(s):
                st = [1]*len(s)
                for i in range(len(s)-2, -1, -1):
                    st[i] = st[i+1]*s[i+1]
                return st
            in_strides = get_strides(x.shape)
            out_strides = get_strides(out_shape)
            pad_before_arr = [p[0] for p in pad_pairs]
            
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
        slices = []
        rank = len(x.shape)
        for i in range(rank):
            pad_before = ctx.pad[-(i * 2 + 2)] if len(ctx.pad) >= (i * 2 + 2) else 0
            slc = slice(pad_before, pad_before + x.shape[i])
            slices.append(slc)
        return (grad_output[tuple(slices)],)

def pad(x: Tensor, pad: Tuple[int, ...], mode='constant', value=0.0) -> Tensor:
    return PadFunction.apply(x, pad, mode, value)

class GatherFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, dim: int, index: Tensor) -> Tensor:
        ctx.save_for_backward(x, index)
        ctx.dim = dim
        _ensure_same_device(x, index, "gather")
        if _should_use_gpu(x, index):
            def get_strides(s):
                st = [1]*len(s)
                for i in range(len(s)-2, -1, -1):
                    st[i] = st[i+1]*s[i+1]
                return st
            x_strides = get_strides(x.shape)
            out_strides = get_strides(index.shape)
            rank = len(x.shape)
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
        grad_x = scatter(zeros_like(x), ctx.dim, index, grad_output)
        return (grad_x, None)

def gather(x: Tensor, dim: int, index: Tensor) -> Tensor:
    return GatherFunction.apply(x, dim, index)

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
            return Tensor(shape=x.shape, dtype=x.dtype, device='gpu', op='scatter', parents=(index, src, x), op_params=op_params)
        else:
            data = _require_cpu_data(x, "x").copy()
            idx = _require_cpu_data(index, "index").astype(int)
            src_data = _require_cpu_data(src, "src")
            np.put_along_axis(data, idx, src_data, axis=dim)
            return Tensor(shape=x.shape, dtype=x.dtype, device='cpu', data=data)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, Tensor]:
        x, index, src = ctx.saved_tensors
        grad_src = gather(grad_output, ctx.dim, index)
        grad_x = scatter(grad_output, ctx.dim, index, zeros_like(src))
        return (grad_x, None, grad_src)

def scatter(x: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    return ScatterFunction.apply(x, dim, index, src)

class SliceFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, key) -> Tensor:
        ctx.save_for_backward(x)
        ctx.key = key
        if x.device == 'gpu':
            data = x.numpy()
            res = data[key]
            return tensor(res, device='gpu')
        else:
            data = _require_cpu_data(x, "x")
            res = data[key]
            return Tensor(shape=res.shape, dtype=x.dtype, device='cpu', data=res)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        if x.device == 'gpu':
            grad_x = np.zeros(x.shape, dtype=np.float32)
            grad_x[ctx.key] = grad_output.numpy()
            return (tensor(grad_x, device='gpu'),)
        ctx.padding = padding
        
        N, C, H, W = x.shape
        C_out, C_in, K_h, K_w = weight.shape
        if C != C_in:
            raise AMEVAForgeShapeError(f"Input channels {C} does not match weight channels {C_in}")
            
        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1
        
        if _should_use_gpu(x, weight):
            x_col = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                           op="im2col", inputs=[x], params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            weight_reshaped = weight.reshape((C_out, C * K_h * K_w))
            weight_t = weight_reshaped.transpose(0, 1)
            
            out_2d = Tensor(shape=(N * H_out * W_out, C_out), dtype=x.dtype, device="gpu", requires_grad=False,
                            op="matmul", inputs=[x_col, weight_t], params=[N * H_out * W_out, C_out, C * K_h * K_w])
                            
            out = out_2d.reshape((N, H_out * W_out, C_out)).transpose(1, 2).reshape((N, C_out, H_out, W_out))
            if bias is not None:
                bias_reshaped = bias.reshape((1, C_out, 1, 1))
                out = out + bias_reshaped
            return out
        else:
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
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(axis=(0, 2, 3)).reshape(bias.shape)
            
        if _should_use_gpu(x, weight):
            grad_out_2d = grad_output.transpose(1, 2).transpose(2, 3).reshape((N * H_out * W_out, C_out))
            x_col = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                           op="im2col", inputs=[x], params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            
            x_col_t = x_col.transpose(0, 1)
            grad_weight_2d = Tensor(shape=(C * K_h * K_w, C_out), dtype=x.dtype, device="gpu", requires_grad=False,
                                    op="matmul", inputs=[x_col_t, grad_out_2d], params=[C * K_h * K_w, C_out, N * H_out * W_out])
            grad_weight = grad_weight_2d.transpose(0, 1).reshape(weight.shape)
            
            weight_reshaped = weight.reshape((C_out, C * K_h * K_w))
            grad_x_col_2d = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                                   op="matmul", inputs=[grad_out_2d, weight_reshaped], params=[N * H_out * W_out, C * K_h * K_w, C_out])
            
            grad_x = Tensor(shape=(N, C, H, W), dtype=x.dtype, device="gpu", requires_grad=False,
                            op="col2im", inputs=[grad_x_col_2d], params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            
            return grad_x, grad_weight, grad_bias, None, None
        else:
            x_data = _require_cpu_data(x)
            weight_data = _require_cpu_data(weight)
            grad_out_data = _require_cpu_data(grad_output)
            
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
            
            grad_weight_data = np.zeros_like(weight_data)
            grad_x_data = np.zeros_like(x_data)
            weight_reshaped = weight_data.reshape((C_out, C * K_h * K_w))
            
            grad_out_2d = grad_out_data.transpose(0, 2, 3, 1).reshape(N, H_out * W_out, C_out)
            
            grad_x_col = np.zeros_like(x_col)
            for n in range(N):
                gw = x_col[n].T @ grad_out_2d[n]
                grad_weight_data += gw.T.reshape(weight.shape)
                
                gxc = grad_out_2d[n] @ weight_reshaped
                grad_x_col[n] = gxc
                
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
            return Tensor(shape=x.shape, dtype=x.dtype, device='gpu', op='scatter', parents=(index, src, x), op_params=op_params)
        else:
            data = _require_cpu_data(x, "x").copy()
            idx = _require_cpu_data(index, "index").astype(int)
            src_data = _require_cpu_data(src, "src")
            np.put_along_axis(data, idx, src_data, axis=dim)
            return Tensor(shape=x.shape, dtype=x.dtype, device='cpu', data=data)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, Tensor]:
        x, index, src = ctx.saved_tensors
        grad_src = gather(grad_output, ctx.dim, index)
        grad_x = scatter(grad_output, ctx.dim, index, zeros_like(src))
        return (grad_x, None, grad_src)

def scatter(x: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    return ScatterFunction.apply(x, dim, index, src)

class SliceFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, key) -> Tensor:
        ctx.save_for_backward(x)
        ctx.key = key
        if x.device == 'gpu':
            data = x.numpy()
            res = data[key]
            return tensor(res, device='gpu')
        else:
            data = _require_cpu_data(x, "x")
            res = data[key]
            return Tensor(shape=res.shape, dtype=x.dtype, device='cpu', data=res)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        if x.device == 'gpu':
            grad_x = np.zeros(x.shape, dtype=np.float32)
            grad_x[ctx.key] = grad_output.numpy()
            return (tensor(grad_x, device='gpu'),)
        else:
            grad_x = np.zeros(x.shape, dtype=np.float32)
            grad_x[ctx.key] = _require_cpu_data(grad_output, "grad")
            return (Tensor(shape=x.shape, dtype=x.dtype, device='cpu', data=grad_x),)

def slice_op(x: Tensor, key) -> Tensor:
    return SliceFunction.apply(x, key)

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
            
        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1
        
        if _should_use_gpu(x, weight):
            x_col = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                           op="im2col", inputs=[x], params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            weight_reshaped = weight.reshape((C_out, C * K_h * K_w))
            weight_t = weight_reshaped.transpose(0, 1)
            
            out_2d = Tensor(shape=(N * H_out * W_out, C_out), dtype=x.dtype, device="gpu", requires_grad=False,
                            op="matmul", inputs=[x_col, weight_t], params=[N * H_out * W_out, C_out, C * K_h * K_w])
                            
            out = out_2d.reshape((N, H_out * W_out, C_out)).transpose(1, 2).reshape((N, C_out, H_out, W_out))
            if bias is not None:
                bias_reshaped = bias.reshape((1, C_out, 1, 1))
                out = out + bias_reshaped
            return out
        else:
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
        if bias is not None and bias.requires_grad:
            g = sum_axis(grad_output, 3)
            g = sum_axis(g, 2)
            g = sum_axis(g, 0)
            grad_bias = g.reshape(bias.shape)
            
        if _should_use_gpu(x, weight):
            grad_out_2d = grad_output.transpose(1, 2).transpose(2, 3).reshape((N * H_out * W_out, C_out))
            x_col = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                           op="im2col", inputs=[x], params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            
            x_col_t = x_col.transpose(0, 1)
            grad_weight_2d = Tensor(shape=(C * K_h * K_w, C_out), dtype=x.dtype, device="gpu", requires_grad=False,
                                    op="matmul", inputs=[x_col_t, grad_out_2d], params=[C * K_h * K_w, C_out, N * H_out * W_out])
            grad_weight = grad_weight_2d.transpose(0, 1).reshape(weight.shape)
            
            weight_reshaped = weight.reshape((C_out, C * K_h * K_w))
            grad_x_col_2d = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                                   op="matmul", inputs=[grad_out_2d, weight_reshaped], params=[N * H_out * W_out, C * K_h * K_w, C_out])
            
            grad_x = Tensor(shape=(N, C, H, W), dtype=x.dtype, device="gpu", requires_grad=False,
                            op="col2im", inputs=[grad_x_col_2d], params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            
            return grad_x, grad_weight, grad_bias, None, None
        else:
            x_data = _require_cpu_data(x)
            weight_data = _require_cpu_data(weight)
            grad_out_data = _require_cpu_data(grad_output)
            
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
            
            grad_weight_data = np.zeros_like(weight_data)
            grad_x_data = np.zeros_like(x_data)
            weight_reshaped = weight_data.reshape((C_out, C * K_h * K_w))
            
            grad_out_2d = grad_out_data.transpose(0, 2, 3, 1).reshape(N, H_out * W_out, C_out)
            
            grad_x_col = np.zeros_like(x_col)
            for n in range(N):
                gw = x_col[n].T @ grad_out_2d[n]
                grad_weight_data += gw.T.reshape(weight.shape)
                
                gxc = grad_out_2d[n] @ weight_reshaped
                grad_x_col[n] = gxc
                
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
            
            return grad_x, grad_weight, grad_bias, None, None

def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1, padding: int = 0) -> Tensor:
    return Conv2dFunction.apply(x, weight, bias, stride, padding)

class MaxPool2dFunction(Function):
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
            return Tensor(shape=(B, C, out_h, out_w), dtype='float32', device='gpu', op='maxpool2d', parents=(x,), op_params=op_params)
        else:
            data = _require_cpu_data(x, "x")
            padded = np.pad(data, ((0,0), (0,0), (ctx.pH, ctx.pH), (ctx.pW, ctx.pW)), constant_values=-np.inf)
            out = np.zeros((B, C, out_h, out_w), dtype=np.float32)
            for h in range(out_h):
                for w in range(out_w):
                    h_start, w_start = h * ctx.sH, w * ctx.sW
                    out[:, :, h, w] = np.max(padded[:, :, h_start:h_start+ctx.kH, w_start:w_start+ctx.kW], axis=(2, 3))
            return Tensor(shape=(B, C, out_h, out_w), dtype='float32', device='cpu', data=out)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        x, = ctx.saved_tensors
        grad_out_np = grad_output.numpy()
        x_np = x.numpy()
        B, C, in_h, in_w = x_np.shape
        out_h = (in_h + 2 * ctx.pH - ctx.kH) // ctx.sH + 1
        out_w = (in_w + 2 * ctx.pW - ctx.kW) // ctx.sW + 1
        
        padded = np.pad(x_np, ((0,0), (0,0), (ctx.pH, ctx.pH), (ctx.pW, ctx.pW)), constant_values=-np.inf)
        grad_padded = np.zeros_like(padded)
        
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
                        
        if ctx.pH > 0 or ctx.pW > 0:
            grad_x_np = grad_padded[:, :, ctx.pH:-ctx.pH if ctx.pH > 0 else None, ctx.pW:-ctx.pW if ctx.pW > 0 else None]
        else:
            grad_x_np = grad_padded
            
        if x.device == 'gpu':
            return (tensor(grad_x_np, device='gpu'), None, None, None)
        else:
            return (Tensor(shape=x.shape, dtype='float32', device='cpu', data=grad_x_np), None, None, None)

def max_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    return MaxPool2dFunction.apply(x, kernel_size, stride, padding)

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
            padded = np.pad(data, ((0,0), (0,0), (ctx.pH, ctx.pH), (ctx.pW, ctx.pW)), constant_values=0)
            out = np.zeros((B, C, out_h, out_w), dtype=np.float32)
            for h in range(out_h):
                for w in range(out_w):
                    h_start, w_start = h * ctx.sH, w * ctx.sW
                    out[:, :, h, w] = np.sum(padded[:, :, h_start:h_start+ctx.kH, w_start:w_start+ctx.kW], axis=(2, 3)) / (ctx.kH * ctx.kW)
            return Tensor(shape=(B, C, out_h, out_w), dtype='float32', device='cpu', data=out)
            
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        x, = ctx.saved_tensors
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
                grad_padded[:, :, h_start:h_start+ctx.kH, w_start:w_start+ctx.kW] += grad_per_element[:, :, h:h+1, w:w+1]
                
        if ctx.pH > 0 or ctx.pW > 0:
            grad_x_np = grad_padded[:, :, ctx.pH:-ctx.pH if ctx.pH > 0 else None, ctx.pW:-ctx.pW if ctx.pW > 0 else None]
        else:
            grad_x_np = grad_padded
            
        if x.device == 'gpu':
            return (tensor(grad_x_np, device='gpu'), None, None, None)
        else:
            return (Tensor(shape=x.shape, dtype='float32', device='cpu', data=grad_x_np), None, None, None)

def avg_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    return AvgPool2dFunction.apply(x, kernel_size, stride, padding)

def col2im(cols: Tensor, output_size: Tuple[int, int], kernel_size: int, stride: int = 1, padding: int = 0) -> Tensor:
    return Col2ImFunction.apply(cols, output_size, kernel_size, stride, padding)

class DropoutFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, p: float, training: bool) -> Tensor:
        if not training or p == 0.0:
            ctx.save_for_backward(x)
            ctx.mask = None
            return x
        
        ctx.p = p
        if _should_use_gpu(x):
            seed = float(np.random.rand())
            ctx.seed = seed
            out = Tensor(shape=x.shape, dtype="float32", device="gpu", op="dropout", parents=(x,), op_params=[seed, float(p)])
            return out
        else:
            data = _require_cpu_data(x, "x")
            mask = np.random.binomial(1, 1 - p, size=data.shape).astype(np.float32)
            res = data * mask * (1.0 / (1.0 - p))
            ctx.mask = mask
            return Tensor(shape=x.shape, dtype="float32", device="cpu", data=res)
            
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, type(None), type(None)]:
        if getattr(ctx, 'mask', None) is None and not hasattr(ctx, 'seed'):
            return grad_output, None, None
            
        if hasattr(ctx, 'seed'):
            seed = ctx.seed
            p = ctx.p
            grad_in = Tensor(shape=grad_output.shape, dtype="float32", device="gpu", op="dropout", parents=(grad_output,), op_params=[seed, float(p)])
            return grad_in, None, None
        else:
            mask = ctx.mask
            p = ctx.p
            data = _require_cpu_data(grad_output, "grad")
            res = data * mask * (1.0 / (1.0 - p))
            return Tensor(shape=grad_output.shape, dtype="float32", device="cpu", data=res), None, None

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return DropoutFunction.apply(x, p, training)

class EmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, weight: Tensor, index: Tensor) -> Tensor:
        ctx.save_for_backward(weight, index)
        data_w = _require_cpu_data(weight, "weight")
        data_i = _require_cpu_data(index, "index").astype(int)
        out_data = data_w[data_i]
        return Tensor(shape=out_data.shape, dtype="float32", device="cpu", data=out_data)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, type(None)]:
        weight, index = ctx.saved_tensors
        data_i = _require_cpu_data(index, "index").astype(int)
        data_g = _require_cpu_data(grad_output, "grad_output")
        grad_w = np.zeros_like(_require_cpu_data(weight, "weight"))
        np.add.at(grad_w, data_i, data_g)
        return (Tensor(shape=weight.shape, dtype="float32", device="cpu", data=grad_w), None)

def embedding(weight: Tensor, index: Tensor) -> Tensor:
    return EmbeddingFunction.apply(weight, index)

class BmmFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "bmm")
        if len(a.shape) != 3 or len(b.shape) != 3:
            raise AMEVAForgeShapeError("bmm requires 3D tensors")
        B, N, M = a.shape
        B2, M2, P = b.shape
        if B != B2 or M != M2:
            raise AMEVAForgeShapeError(f"bmm shape mismatch: {a.shape} and {b.shape}")

        if _should_use_gpu(a, b):
            return Tensor(shape=(B, N, P), dtype="float32", device="gpu",
                          op="batched_matmul", parents=(a, b), op_params=[int(B), int(N), int(P), int(M)])
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            import numpy as np
            res = np.matmul(data_a, data_b)
            return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        grad_a = bmm(grad_output, permute(b, (0, 2, 1)))
        grad_b = bmm(permute(a, (0, 2, 1)), grad_output)
        return grad_a, grad_b

def bmm(a: Tensor, b: Tensor) -> Tensor:
    return BmmFunction.apply(a, b)
