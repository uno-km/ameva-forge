"""
special.py — Special mathematical functions for AMEVA-Forge
Provides PyTorch-compatible torch.special APIs.
"""
from typing import Optional, Tuple, Union
import numpy as np
import math
from .tensor import Tensor
from .autograd import Function
from .ops import _require_cpu_data, tensor
from .errors import AMEVAForgeShapeError, AMEVAForgeDTypeError

# Try importing scipy.special for full numerical precision
try:
    import scipy.special as _sp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _safe_erf(x: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY:
        return _sp.erf(x)
    return np.vectorize(math.erf)(x).astype(np.float32)

def _safe_erfc(x: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY:
        return _sp.erfc(x)
    return np.vectorize(math.erfc)(x).astype(np.float32)

def _safe_erfinv(x: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY:
        return _sp.erfinv(x)
    # Winitzki approximation for erfinv
    a = 0.147
    sgn = np.sign(x)
    log_term = np.log(np.clip(1.0 - x**2, 1e-12, 1.0))
    term1 = 2.0 / (np.pi * a) + log_term / 2.0
    term2 = log_term / a
    return (sgn * np.sqrt(np.sqrt(np.maximum(0.0, term1**2 - term2)) - term1)).astype(np.float32)

def _safe_gammaln(x: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY:
        return _sp.gammaln(x)
    return np.vectorize(math.lgamma)(x).astype(np.float32)

def _safe_digamma(x: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY:
        return _sp.digamma(x)
    eps = 1e-5
    return ((np.vectorize(math.lgamma)(x + eps) - np.vectorize(math.lgamma)(x - eps)) / (2 * eps)).astype(np.float32)


class ErfFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        data_x = _require_cpu_data(x, "x")
        res = _safe_erf(data_x)
        return Tensor(shape=res.shape, dtype=x.dtype, device=x.device, data=res)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        data_x = _require_cpu_data(x, "x")
        data_g = _require_cpu_data(grad_output, "grad_output")
        # d/dx erf(x) = (2 / sqrt(pi)) * exp(-x^2)
        grad_x = data_g * (2.0 / np.sqrt(np.pi)) * np.exp(-data_x**2)
        return (Tensor(shape=grad_x.shape, dtype=grad_output.dtype, device=grad_output.device, data=grad_x.astype(np.float32)),)

def erf(input: Tensor) -> Tensor:
    """Computes the error function of each element."""
    return ErfFunction.apply(input)


class ErfcFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        data_x = _require_cpu_data(x, "x")
        res = _safe_erfc(data_x)
        return Tensor(shape=res.shape, dtype=x.dtype, device=x.device, data=res)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        data_x = _require_cpu_data(x, "x")
        data_g = _require_cpu_data(grad_output, "grad_output")
        # d/dx erfc(x) = -(2 / sqrt(pi)) * exp(-x^2)
        grad_x = -data_g * (2.0 / np.sqrt(np.pi)) * np.exp(-data_x**2)
        return (Tensor(shape=grad_x.shape, dtype=grad_output.dtype, device=grad_output.device, data=grad_x.astype(np.float32)),)

def erfc(input: Tensor) -> Tensor:
    """Computes the complementary error function of each element."""
    return ErfcFunction.apply(input)


class ErfinvFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        data_x = _require_cpu_data(x, "x")
        res = _safe_erfinv(data_x)
        res_tensor = Tensor(shape=res.shape, dtype=x.dtype, device=x.device, data=res)
        ctx.save_for_backward(res_tensor)
        return res_tensor

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        res_tensor, = ctx.saved_tensors
        data_y = _require_cpu_data(res_tensor, "y")
        data_g = _require_cpu_data(grad_output, "grad_output")
        # d/dx erfinv(x) = (sqrt(pi) / 2) * exp((erfinv(x))^2)
        grad_x = data_g * (np.sqrt(np.pi) / 2.0) * np.exp(data_y**2)
        return (Tensor(shape=grad_x.shape, dtype=grad_output.dtype, device=grad_output.device, data=grad_x.astype(np.float32)),)

def erfinv(input: Tensor) -> Tensor:
    """Computes the inverse error function of each element."""
    return ErfinvFunction.apply(input)


class GammalnFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        data_x = _require_cpu_data(x, "x")
        res = _safe_gammaln(data_x)
        return Tensor(shape=res.shape, dtype=x.dtype, device=x.device, data=res)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        data_x = _require_cpu_data(x, "x")
        data_g = _require_cpu_data(grad_output, "grad_output")
        grad_x = data_g * _safe_digamma(data_x)
        return (Tensor(shape=grad_x.shape, dtype=grad_output.dtype, device=grad_output.device, data=grad_x.astype(np.float32)),)

def gammaln(input: Tensor) -> Tensor:
    """Computes the logarithm of the absolute value of the gamma function."""
    return GammalnFunction.apply(input)

def loggamma(input: Tensor) -> Tensor:
    """Alias for gammaln."""
    return gammaln(input)


def digamma(input: Tensor) -> Tensor:
    """Computes the logarithmic derivative of the gamma function."""
    data_x = _require_cpu_data(input, "input")
    res = _safe_digamma(data_x)
    return Tensor(shape=res.shape, dtype=input.dtype, device=input.device, data=res)

def psi(input: Tensor) -> Tensor:
    """Alias for digamma."""
    return digamma(input)


class Expm1Function(Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        data_x = _require_cpu_data(x, "x")
        res = np.expm1(data_x)
        return Tensor(shape=res.shape, dtype=x.dtype, device=x.device, data=res.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        data_x = _require_cpu_data(x, "x")
        data_g = _require_cpu_data(grad_output, "grad_output")
        grad_x = data_g * np.exp(data_x)
        return (Tensor(shape=grad_x.shape, dtype=grad_output.dtype, device=grad_output.device, data=grad_x.astype(np.float32)),)

def expm1(input: Tensor) -> Tensor:
    """Computes exp(x) - 1 element-wise."""
    return Expm1Function.apply(input)


class Log1pFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        data_x = _require_cpu_data(x, "x")
        res = np.log1p(data_x)
        return Tensor(shape=res.shape, dtype=x.dtype, device=x.device, data=res.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        x, = ctx.saved_tensors
        data_x = _require_cpu_data(x, "x")
        data_g = _require_cpu_data(grad_output, "grad_output")
        grad_x = data_g / (1.0 + data_x)
        return (Tensor(shape=grad_x.shape, dtype=grad_output.dtype, device=grad_output.device, data=grad_x.astype(np.float32)),)

def log1p(input: Tensor) -> Tensor:
    """Computes log(1 + x) element-wise."""
    return Log1pFunction.apply(input)


def expit(input: Tensor) -> Tensor:
    """Computes the standard logistic sigmoid function 1 / (1 + exp(-x))."""
    data_x = _require_cpu_data(input, "input")
    res = 1.0 / (1.0 + np.exp(-data_x))
    return Tensor(shape=res.shape, dtype=input.dtype, device=input.device, data=res.astype(np.float32))


def logit(input: Tensor, eps: Optional[float] = None) -> Tensor:
    """Computes the logit (inverse of expit) log(x / (1 - x))."""
    data_x = _require_cpu_data(input, "input")
    if eps is not None:
        data_x = np.clip(data_x, eps, 1.0 - eps)
    res = np.log(data_x / (1.0 - data_x))
    return Tensor(shape=res.shape, dtype=input.dtype, device=input.device, data=res.astype(np.float32))


def sinc(input: Tensor) -> Tensor:
    """Computes the normalized sinc function sin(pi * x) / (pi * x)."""
    data_x = _require_cpu_data(input, "input")
    res = np.sinc(data_x)
    return Tensor(shape=res.shape, dtype=input.dtype, device=input.device, data=res.astype(np.float32))


def i0(input: Tensor) -> Tensor:
    """Computes the zeroth order modified Bessel function of the first kind."""
    data_x = _require_cpu_data(input, "input")
    res = np.i0(data_x)
    return Tensor(shape=res.shape, dtype=input.dtype, device=input.device, data=res.astype(np.float32))


class XlogyFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)
        data_x = _require_cpu_data(x, "x")
        data_y = _require_cpu_data(y, "y")
        res = np.where(data_x == 0.0, 0.0, data_x * np.log(data_y))
        return Tensor(shape=res.shape, dtype=x.dtype, device=x.device, data=res.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors
        data_x = _require_cpu_data(x, "x")
        data_y = _require_cpu_data(y, "y")
        data_g = _require_cpu_data(grad_output, "grad_output")
        
        # d/dx (x * log(y)) = log(y)
        grad_x = np.where(data_x == 0.0, 0.0, data_g * np.log(data_y))
        # d/dy (x * log(y)) = x / y
        grad_y = data_g * (data_x / data_y)
        return (
            Tensor(shape=grad_x.shape, dtype=grad_output.dtype, device=grad_output.device, data=grad_x.astype(np.float32)),
            Tensor(shape=grad_y.shape, dtype=grad_output.dtype, device=grad_output.device, data=grad_y.astype(np.float32))
        )

def xlogy(x: Tensor, y: Tensor) -> Tensor:
    """Computes x * log(y), returning 0 when x == 0."""
    return XlogyFunction.apply(x, y)