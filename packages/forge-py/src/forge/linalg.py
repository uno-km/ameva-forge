"""
linalg.py — Linear Algebra module for AMEVA-Forge
Provides PyTorch-compatible torch.linalg APIs.
"""
from typing import Optional, Tuple, Union, Sequence
import numpy as np
from .tensor import Tensor
from .autograd import Function
from .ops import _require_cpu_data, tensor
from .errors import AMEVAForgeShapeError, AMEVAForgeDTypeError


class InvFunction(Function):
    @staticmethod
    def forward(ctx, A: Tensor) -> Tensor:
        data_a = _require_cpu_data(A, "A")
        if data_a.ndim < 2 or data_a.shape[-2] != data_a.shape[-1]:
            raise AMEVAForgeShapeError(f"linalg.inv: A must be batches of square matrices, got shape {data_a.shape}")
        res = np.linalg.inv(data_a)
        res_tensor = Tensor(shape=res.shape, dtype=A.dtype, device=A.device, data=res)
        ctx.save_for_backward(res_tensor)
        return res_tensor

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        inv_a, = ctx.saved_tensors
        data_inv = _require_cpu_data(inv_a, "inv_a")
        data_g = _require_cpu_data(grad_output, "grad_output")
        # d(A^{-1}) = -A^{-T} @ grad @ A^{-T}
        grad_a = -data_inv.swapaxes(-1, -2) @ data_g @ data_inv.swapaxes(-1, -2)
        return (Tensor(shape=grad_a.shape, dtype=grad_output.dtype, device=grad_output.device, data=grad_a),)

def inv(A: Tensor) -> Tensor:
    """Computes the inverse of a square matrix or batch of matrices."""
    return InvFunction.apply(A)


class DetFunction(Function):
    @staticmethod
    def forward(ctx, A: Tensor) -> Tensor:
        data_a = _require_cpu_data(A, "A")
        if data_a.ndim < 2 or data_a.shape[-2] != data_a.shape[-1]:
            raise AMEVAForgeShapeError(f"linalg.det: A must be batches of square matrices, got shape {data_a.shape}")
        res = np.linalg.det(data_a)
        ctx.save_for_backward(A)
        return Tensor(shape=res.shape, dtype=A.dtype, device=A.device, data=res)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        A, = ctx.saved_tensors
        data_a = _require_cpu_data(A, "A")
        data_g = _require_cpu_data(grad_output, "grad_output")
        data_inv = np.linalg.inv(data_a)
        data_det = np.linalg.det(data_a)
        
        # d(det(A)) = det(A) * tr(A^{-1} * dA) -> grad_A = det(A) * (A^{-1})^T * grad_out
        if data_a.ndim == 2:
            grad_a = data_det * data_g * data_inv.T
        else:
            grad_a = data_det[..., None, None] * data_g[..., None, None] * data_inv.swapaxes(-1, -2)
        return (Tensor(shape=grad_a.shape, dtype=grad_output.dtype, device=grad_output.device, data=grad_a),)

def det(A: Tensor) -> Tensor:
    """Computes the determinant of a square matrix or batch of matrices."""
    return DetFunction.apply(A)


def pinv(A: Tensor, rcond: float = 1e-15, hermitian: bool = False) -> Tensor:
    """Computes the pseudoinverse (Moore-Penrose inverse) of a matrix."""
    data_a = _require_cpu_data(A, "A")
    res = np.linalg.pinv(data_a, rcond=rcond, hermitian=hermitian)
    return Tensor(shape=res.shape, dtype=A.dtype, device=A.device, data=res)


def cholesky(A: Tensor, upper: bool = False) -> Tensor:
    """Computes the Cholesky decomposition of a symmetric positive-definite matrix."""
    data_a = _require_cpu_data(A, "A")
    res = np.linalg.cholesky(data_a)
    if upper:
        res = res.swapaxes(-1, -2)
    return Tensor(shape=res.shape, dtype=A.dtype, device=A.device, data=res)


def qr(A: Tensor, mode: str = 'reduced') -> Tuple[Tensor, Tensor]:
    """Computes the QR decomposition of a matrix."""
    data_a = _require_cpu_data(A, "A")
    q, r = np.linalg.qr(data_a, mode=mode)
    return (
        Tensor(shape=q.shape, dtype=A.dtype, device=A.device, data=q),
        Tensor(shape=r.shape, dtype=A.dtype, device=A.device, data=r)
    )


def svd(A: Tensor, full_matrices: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes the singular value decomposition (SVD) of a matrix."""
    data_a = _require_cpu_data(A, "A")
    u, s, vh = np.linalg.svd(data_a, full_matrices=full_matrices)
    return (
        Tensor(shape=u.shape, dtype=A.dtype, device=A.device, data=u),
        Tensor(shape=s.shape, dtype=A.dtype, device=A.device, data=s),
        Tensor(shape=vh.shape, dtype=A.dtype, device=A.device, data=vh)
    )


def eigh(A: Tensor, UPLO: str = 'L') -> Tuple[Tensor, Tensor]:
    """Computes the eigenvalues and eigenvectors of a symmetric matrix."""
    data_a = _require_cpu_data(A, "A")
    w, v = np.linalg.eigh(data_a, UPLO=UPLO)
    return (
        Tensor(shape=w.shape, dtype=A.dtype, device=A.device, data=w),
        Tensor(shape=v.shape, dtype=A.dtype, device=A.device, data=v)
    )


def matrix_rank(A: Tensor, tol: Optional[float] = None, hermitian: bool = False) -> Tensor:
    """Computes the numerical rank of a matrix."""
    data_a = _require_cpu_data(A, "A")
    res = np.linalg.matrix_rank(data_a, tol=tol, hermitian=hermitian)
    return Tensor(shape=res.shape, dtype="int32", device=A.device, data=res.astype(np.int32))


def norm(A: Tensor, ord: Optional[Union[int, float, str]] = None, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> Tensor:
    """Computes a matrix or vector norm."""
    data_a = _require_cpu_data(A, "A")
    res = np.linalg.norm(data_a, ord=ord, axis=dim, keepdims=keepdim)
    return Tensor(shape=res.shape, dtype=A.dtype, device=A.device, data=np.asarray(res, dtype=np.float32))