"""CPU 전용 단위 테스트 — Pyodide/WebGPU 환경 없이 실행 가능."""
import unittest
import numpy as np
import sys
import os

# ameva_tensor 패키지 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import ameva_tensor as at
from ameva_tensor.errors import (
    AMEVATensorShapeError,
    AMEVATensorDisposedError,
    AMEVATensorDeviceError,
)


class TestCPUTensorCreation(unittest.TestCase):
    def test_create_from_list(self):
        t = at.tensor([1.0, 2.0, 3.0])
        self.assertEqual(t.shape, (3,))
        self.assertEqual(t.device, 'cpu')
        np.testing.assert_allclose(t.numpy(), [1.0, 2.0, 3.0])

    def test_create_from_numpy(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = at.tensor(arr)
        self.assertEqual(t.shape, (2, 2))
        np.testing.assert_allclose(t.numpy(), arr)

    def test_create_with_dtype_conversion(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        t = at.tensor(arr)
        self.assertEqual(t.numpy().dtype, np.float32)

    def test_random_creation(self):
        t = at.random((3, 4))
        self.assertEqual(t.shape, (3, 4))
        self.assertEqual(t.device, 'cpu')

    def test_ones_like(self):
        t = at.tensor([[1.0, 2.0], [3.0, 4.0]])
        o = at.ones_like(t)
        np.testing.assert_allclose(o.numpy(), np.ones((2, 2), dtype=np.float32))

    def test_zeros_like(self):
        t = at.tensor([[1.0, 2.0]])
        z = at.zeros_like(t)
        np.testing.assert_allclose(z.numpy(), np.zeros((1, 2), dtype=np.float32))

    def test_zeros(self):
        t = at.zeros((2, 3))
        np.testing.assert_allclose(t.numpy(), np.zeros((2, 3), dtype=np.float32))

    def test_ones(self):
        t = at.ones((2, 3))
        np.testing.assert_allclose(t.numpy(), np.ones((2, 3), dtype=np.float32))

    def test_full(self):
        t = at.full((2, 2), 3.14)
        np.testing.assert_allclose(t.numpy(), np.full((2, 2), 3.14, dtype=np.float32))


class TestCPUOps(unittest.TestCase):
    def test_add(self):
        a = at.tensor([1.0, 2.0, 3.0])
        b = at.tensor([4.0, 5.0, 6.0])
        c = at.add(a, b)
        np.testing.assert_allclose(c.numpy(), [5.0, 7.0, 9.0])

    def test_mul(self):
        a = at.tensor([2.0, 3.0])
        b = at.tensor([4.0, 5.0])
        c = at.mul(a, b)
        np.testing.assert_allclose(c.numpy(), [8.0, 15.0])

    def test_matmul(self):
        a = at.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = at.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
        np.testing.assert_allclose(c.numpy(), expected)

    def test_relu(self):
        a = at.tensor([-1.0, 0.0, 1.0, -2.0, 3.0])
        b = at.relu(a)
        np.testing.assert_allclose(b.numpy(), [0.0, 0.0, 1.0, 0.0, 3.0])

    def test_transpose(self):
        a = at.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = at.transpose(a)
        self.assertEqual(b.shape, (3, 2))
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
        np.testing.assert_allclose(b.numpy(), expected)

    def test_operator_overloads(self):
        a = at.tensor([1.0, 2.0])
        b = at.tensor([3.0, 4.0])
        c = a + b
        np.testing.assert_allclose(c.numpy(), [4.0, 6.0])
        d = a * b
        np.testing.assert_allclose(d.numpy(), [3.0, 8.0])


class TestCPUErrors(unittest.TestCase):
    def test_matmul_shape_mismatch(self):
        a = at.tensor([[1.0, 2.0]])
        b = at.tensor([[1.0, 2.0]])
        with self.assertRaises(AMEVATensorShapeError):
            a @ b

    def test_matmul_non_2d(self):
        a = at.tensor([1.0, 2.0, 3.0])
        b = at.tensor([4.0, 5.0, 6.0])
        with self.assertRaises(AMEVATensorShapeError):
            at.matmul(a, b)

    def test_disposed_tensor_access(self):
        t = at.tensor([1.0, 2.0])
        at.dispose(t)
        with self.assertRaises(AMEVATensorDisposedError):
            t.numpy()

    def test_mixed_device_error(self):
        # CPU + GPU 혼합은 GPU가 없어도 device 문자열 차이로 검증 가능
        cpu_t = at.tensor([1.0], device='cpu')
        # GPU 텐서를 직접 생성 (realize 없이)
        from ameva_tensor.tensor import Tensor
        gpu_t = Tensor(shape=(1,), dtype='float32', device='gpu')
        with self.assertRaises(AMEVATensorDeviceError):
            at.add(cpu_t, gpu_t)


class TestCPUAutograd(unittest.TestCase):
    def test_simple_backward(self):
        x = at.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = at.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False)
        z = x @ y  # identity matmul
        # z is 2x2, need explicit gradient
        grad = at.ones_like(z)
        z.backward(grad)
        # dZ/dX @ Y^T = ones @ I = ones
        np.testing.assert_allclose(
            x.grad.numpy(),
            np.ones((2, 2), dtype=np.float32),
            rtol=1e-5
        )

    def test_relu_backward(self):
        x = at.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
        y = at.relu(x)
        grad = at.ones_like(y)
        y.backward(grad)
        expected_grad = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(x.grad.numpy(), expected_grad)

    def test_no_grad_context(self):
        from ameva_tensor.autograd import no_grad
        x = at.tensor([[1.0]], requires_grad=True)
        y = at.tensor([[2.0]], requires_grad=False)
        with no_grad():
            z = x @ y
        self.assertFalse(z.requires_grad)


if __name__ == '__main__':
    unittest.main()
