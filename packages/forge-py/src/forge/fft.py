"""
fft.py — Fast Fourier Transform module for AMEVA-Forge
Provides PyTorch-compatible torch.fft APIs.
"""
from typing import Optional, Tuple, Union, Sequence
import numpy as np
from .tensor import Tensor
from .autograd import Function
from .ops import _require_cpu_data, tensor
from .errors import AMEVAForgeShapeError, AMEVAForgeDTypeError


class RFFTFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
        data_x = _require_cpu_data(input, "input")
        ctx.input_shape = data_x.shape
        ctx.dim = dim
        ctx.norm = norm
        ctx.n = n if n is not None else data_x.shape[dim]
        
        res = np.fft.rfft(data_x, n=n, axis=dim, norm=norm)
        return Tensor(shape=res.shape, dtype="complex64" if res.dtype == np.complex64 else "complex128", device=input.device, data=res)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        data_g = _require_cpu_data(grad_output, "grad_output")
        dim = ctx.dim
        norm = ctx.norm
        n = ctx.n
        
        grad_x = np.fft.irfft(data_g, n=n, axis=dim, norm=norm)
        if grad_x.shape != ctx.input_shape:
            grad_x = grad_x[tuple(slice(0, s) for s in ctx.input_shape)]
            
        return (Tensor(shape=grad_x.shape, dtype="float32", device=grad_output.device, data=grad_x.astype(np.float32)), None, None, None)


class IRFFTFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
        data_x = _require_cpu_data(input, "input")
        ctx.input_shape = data_x.shape
        ctx.dim = dim
        ctx.norm = norm
        ctx.n = n
        
        res = np.fft.irfft(data_x, n=n, axis=dim, norm=norm)
        return Tensor(shape=res.shape, dtype="float32", device=input.device, data=res.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        data_g = _require_cpu_data(grad_output, "grad_output")
        dim = ctx.dim
        norm = ctx.norm
        
        grad_x = np.fft.rfft(data_g, n=ctx.n, axis=dim, norm=norm)
        return (Tensor(shape=grad_x.shape, dtype="complex64", device=grad_output.device, data=grad_x), None, None, None)


def rfft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    """Computes the 1D discrete Fourier transform for real input."""
    return RFFTFunction.apply(input, n, dim, norm)


def irfft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    """Computes the inverse of rfft for real output."""
    return IRFFTFunction.apply(input, n, dim, norm)


def fft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    """Computes the 1D discrete Fourier transform."""
    data_x = _require_cpu_data(input, "input")
    res = np.fft.fft(data_x, n=n, axis=dim, norm=norm)
    return Tensor(shape=res.shape, dtype=str(res.dtype), device=input.device, data=res)


def ifft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    """Computes the 1D inverse discrete Fourier transform."""
    data_x = _require_cpu_data(input, "input")
    res = np.fft.ifft(data_x, n=n, axis=dim, norm=norm)
    return Tensor(shape=res.shape, dtype=str(res.dtype), device=input.device, data=res)


def fft2(input: Tensor, s: Optional[Sequence[int]] = None, dim: Tuple[int, int] = (-2, -1), norm: Optional[str] = None) -> Tensor:
    """Computes the 2D discrete Fourier transform."""
    data_x = _require_cpu_data(input, "input")
    res = np.fft.fft2(data_x, s=s, axes=dim, norm=norm)
    return Tensor(shape=res.shape, dtype=str(res.dtype), device=input.device, data=res)


def ifft2(input: Tensor, s: Optional[Sequence[int]] = None, dim: Tuple[int, int] = (-2, -1), norm: Optional[str] = None) -> Tensor:
    """Computes the 2D inverse discrete Fourier transform."""
    data_x = _require_cpu_data(input, "input")
    res = np.fft.ifft2(data_x, s=s, axes=dim, norm=norm)
    return Tensor(shape=res.shape, dtype=str(res.dtype), device=input.device, data=res)


def rfft2(input: Tensor, s: Optional[Sequence[int]] = None, dim: Tuple[int, int] = (-2, -1), norm: Optional[str] = None) -> Tensor:
    """Computes the 2D discrete Fourier transform for real input."""
    data_x = _require_cpu_data(input, "input")
    res = np.fft.rfft2(data_x, s=s, axes=dim, norm=norm)
    return Tensor(shape=res.shape, dtype=str(res.dtype), device=input.device, data=res)


def irfft2(input: Tensor, s: Optional[Sequence[int]] = None, dim: Tuple[int, int] = (-2, -1), norm: Optional[str] = None) -> Tensor:
    """Computes the 2D inverse discrete Fourier transform for real output."""
    data_x = _require_cpu_data(input, "input")
    res = np.fft.irfft2(data_x, s=s, axes=dim, norm=norm)
    return Tensor(shape=res.shape, dtype="float32", device=input.device, data=res.astype(np.float32))


def fftfreq(n: int, d: float = 1.0) -> Tensor:
    """Computes the discrete Fourier Transform sample frequencies."""
    res = np.fft.fftfreq(n, d=d)
    return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res.astype(np.float32))


def rfftfreq(n: int, d: float = 1.0) -> Tensor:
    """Computes the discrete Fourier Transform sample frequencies for rfft."""
    res = np.fft.rfftfreq(n, d=d)
    return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res.astype(np.float32))


def fftshift(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """Shifts the zero-frequency component to the center of the spectrum."""
    data_x = _require_cpu_data(input, "input")
    res = np.fft.fftshift(data_x, axes=dim)
    return Tensor(shape=res.shape, dtype=input.dtype, device=input.device, data=res)


def ifftshift(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """Inverse of fftshift."""
    data_x = _require_cpu_data(input, "input")
    res = np.fft.ifftshift(data_x, axes=dim)
    return Tensor(shape=res.shape, dtype=input.dtype, device=input.device, data=res)