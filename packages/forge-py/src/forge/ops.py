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
