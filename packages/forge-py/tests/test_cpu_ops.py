"""CPU 전용 단위 테스트 — Pyodide/WebGPU 환경 없이 실행 가능."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import unittest
import numpy as np

import forge as at
from forge.errors import (
    AMEVAForgeShapeError,
    AMEVAForgeDisposedError,
    AMEVAForgeDeviceError,
)


class TestCPUTensorCreation(unittest.TestCase):
    """CPU 텐서 생성 테스트."""

    def test_create_from_list(self):
        """리스트로부터 텐서를 생성한다."""
        t = at.tensor([1.0, 2.0, 3.0])
        self.assertEqual(t.shape, (3,))
        self.assertEqual(t.device, 'cpu')
        np.testing.assert_allclose(t.numpy(), [1.0, 2.0, 3.0])

    def test_create_from_numpy(self):
        """numpy 배열로부터 텐서를 생성한다."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = at.tensor(arr)
        self.assertEqual(t.shape, (2, 2))
        np.testing.assert_allclose(t.numpy(), arr)

    def test_create_with_dtype_conversion(self):
        """int64 → float32 자동 변환을 검증한다."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        t = at.tensor(arr)
        self.assertEqual(t.numpy().dtype, np.float32)

    def test_random_creation(self):
        """랜덤 텐서 생성을 검증한다."""
        t = at.random((3, 4))
        self.assertEqual(t.shape, (3, 4))
        self.assertEqual(t.device, 'cpu')

    def test_ones_like(self):
        """ones_like 생성을 검증한다."""
        t = at.tensor([[1.0, 2.0], [3.0, 4.0]])
        o = at.ones_like(t)
        np.testing.assert_allclose(o.numpy(), np.ones((2, 2), dtype=np.float32))

    def test_zeros_like(self):
        """zeros_like 생성을 검증한다."""
        t = at.tensor([[1.0, 2.0]])
        z = at.zeros_like(t)
        np.testing.assert_allclose(z.numpy(), np.zeros((1, 2), dtype=np.float32))

    def test_zeros(self):
        """zeros 팩토리를 검증한다."""
        t = at.zeros((2, 3))
        np.testing.assert_allclose(t.numpy(), np.zeros((2, 3), dtype=np.float32))

    def test_ones(self):
        """ones 팩토리를 검증한다."""
        t = at.ones((2, 3))
        np.testing.assert_allclose(t.numpy(), np.ones((2, 3), dtype=np.float32))

    def test_full(self):
        """full 팩토리를 검증한다."""
        t = at.full((2, 2), 3.14)
        np.testing.assert_allclose(t.numpy(), np.full((2, 2), 3.14, dtype=np.float32))


class TestCPUOps(unittest.TestCase):
    """CPU 연산 테스트."""

    def test_add(self):
        """덧셈 연산 정확성을 검증한다."""
        a = at.tensor([1.0, 2.0, 3.0])
        b = at.tensor([4.0, 5.0, 6.0])
        c = at.add(a, b)
        np.testing.assert_allclose(c.numpy(), [5.0, 7.0, 9.0])

    def test_mul(self):
        """곱셈 연산 정확성을 검증한다."""
        a = at.tensor([2.0, 3.0])
        b = at.tensor([4.0, 5.0])
        c = at.mul(a, b)
        np.testing.assert_allclose(c.numpy(), [8.0, 15.0])

    def test_matmul(self):
        """행렬 곱셈 정확성을 검증한다."""
        a = at.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = at.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
        np.testing.assert_allclose(c.numpy(), expected)

    def test_relu(self):
        """ReLU 활성화 함수를 검증한다."""
        a = at.tensor([-1.0, 0.0, 1.0, -2.0, 3.0])
        b = at.relu(a)
        np.testing.assert_allclose(b.numpy(), [0.0, 0.0, 1.0, 0.0, 3.0])

    def test_transpose(self):
        """전치 연산을 검증한다."""
        a = at.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = at.transpose(a)
        self.assertEqual(b.shape, (3, 2))
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
        np.testing.assert_allclose(b.numpy(), expected)

    def test_operator_overloads(self):
        """연산자 오버로딩 (+, *, @)을 검증한다."""
        a = at.tensor([1.0, 2.0])
        b = at.tensor([3.0, 4.0])
        c = a + b
        np.testing.assert_allclose(c.numpy(), [4.0, 6.0])
        d = a * b
        np.testing.assert_allclose(d.numpy(), [3.0, 8.0])


class TestCPUErrors(unittest.TestCase):
    """CPU 에러 처리 테스트."""

    def test_matmul_shape_mismatch(self):
        """matmul 내적 차원 불일치 시 에러를 검증한다."""
        a = at.tensor([[1.0, 2.0]])
        b = at.tensor([[1.0, 2.0]])
        with self.assertRaises(AMEVAForgeShapeError):
            a @ b

    def test_matmul_non_2d(self):
        """1D 텐서 matmul 시 에러를 검증한다."""
        a = at.tensor([1.0, 2.0, 3.0])
        b = at.tensor([4.0, 5.0, 6.0])
        with self.assertRaises(AMEVAForgeShapeError):
            at.matmul(a, b)

    def test_disposed_tensor_access(self):
        """해제된 텐서 접근 시 에러를 검증한다."""
        t = at.tensor([1.0, 2.0])
        at.dispose(t)
        with self.assertRaises(AMEVAForgeDisposedError):
            t.numpy()

    def test_mixed_device_error(self):
        """CPU+GPU 혼합 연산 시 에러를 검증한다."""
        cpu_t = at.tensor([1.0], device='cpu')
        from forge.tensor import Tensor
        gpu_t = Tensor(shape=(1,), dtype='float32', device='gpu')
        with self.assertRaises(AMEVAForgeDeviceError):
            at.add(cpu_t, gpu_t)


class TestCPUAutograd(unittest.TestCase):
    """CPU 자동미분 테스트."""

    def test_simple_backward(self):
        """단순 matmul 역전파를 검증한다."""
        x = at.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = at.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False)
        z = x @ y
        grad = at.ones_like(z)
        z.backward(grad)
        np.testing.assert_allclose(
            x.grad.numpy(),
            np.ones((2, 2), dtype=np.float32),
            rtol=1e-5
        )

    def test_relu_backward(self):
        """ReLU 역전파를 검증한다."""
        x = at.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
        y = at.relu(x)
        grad = at.ones_like(y)
        y.backward(grad)
        expected_grad = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(x.grad.numpy(), expected_grad)

    def test_no_grad_context(self):
        """no_grad 컨텍스트에서 grad 비활성화를 검증한다."""
        from forge.autograd import no_grad
        x = at.tensor([[1.0]], requires_grad=True)
        y = at.tensor([[2.0]], requires_grad=False)
        with no_grad():
            z = x @ y
        self.assertFalse(z.requires_grad)


if __name__ == '__main__':
    from report_generator import run_and_report
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    run_and_report(suite, 'CPU_단위_테스트', 'Category 1: CPU Unit Tests')
