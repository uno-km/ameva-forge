import asyncio
import sys
import unittest
import numpy as np
import ameva_tensor as at
from ameva_tensor.errors import (
    AMEVATensorDeviceError,
    AMEVATensorShapeError,
    AMEVATensorDisposedError
)

@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestAMEVATensorCore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()
        self.device = at.current_device()

    async def test_tensor_creation_and_zerocopy(self):
        # 1. Zero-Copy Upload and Readback test
        arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        t = at.tensor(arr, device=self.device)
        self.assertEqual(t.shape, (2, 2))
        
        res = await t.numpy_async()
        np.testing.assert_allclose(res, arr, err_msg="Zero copy readback data mismatch")

    async def test_add_rigorous(self):
        a_np = np.random.randn(10, 10).astype(np.float32)
        b_np = np.random.randn(10, 10).astype(np.float32)
        
        a = at.tensor(a_np, device=self.device)
        b = at.tensor(b_np, device=self.device)
        
        c = at.add(a, b)
        c_np = await c.numpy_async()
        
        expected = a_np + b_np
        np.testing.assert_allclose(c_np, expected, rtol=1e-4, err_msg="Add math mismatch")

    async def test_mul_rigorous(self):
        a_np = np.random.randn(5, 5).astype(np.float32)
        b_np = np.random.randn(5, 5).astype(np.float32)
        
        a = at.tensor(a_np, device=self.device)
        b = at.tensor(b_np, device=self.device)
        
        c = at.mul(a, b)
        c_np = await c.numpy_async()
        
        expected = a_np * b_np
        np.testing.assert_allclose(c_np, expected, rtol=1e-4, err_msg="Mul math mismatch")

    async def test_matmul_rigorous(self):
        a_np = np.random.randn(16, 32).astype(np.float32)
        b_np = np.random.randn(32, 16).astype(np.float32)
        
        a = at.tensor(a_np, device=self.device)
        b = at.tensor(b_np, device=self.device)
        
        c = a @ b
        c_np = await c.numpy_async()
        
        expected = np.matmul(a_np, b_np)
        np.testing.assert_allclose(c_np, expected, rtol=1e-3, err_msg="Matmul math mismatch")

    async def test_relu_rigorous(self):
        a_np = np.array([[-1.0, 0.0, 2.5], [3.1, -4.2, 5.0]], dtype=np.float32)
        
        a = at.tensor(a_np, device=self.device)
        b = at.relu(a)
        
        b_np = await b.numpy_async()
        expected = np.maximum(a_np, 0.0)
        
        np.testing.assert_allclose(b_np, expected, err_msg="ReLU math mismatch")

    async def test_transpose_rigorous(self):
        a_np = np.random.randn(4, 7).astype(np.float32)
        a = at.tensor(a_np, device=self.device)
        b = at.transpose(a)
        
        b_np = await b.numpy_async()
        expected = np.transpose(a_np)
        
        np.testing.assert_allclose(b_np, expected, err_msg="Transpose math mismatch")

    def test_mixed_device_error(self):
        if self.device == "gpu":
            cpu_tensor = at.tensor([1, 2], device="cpu")
            gpu_tensor = at.tensor([1, 2], device="gpu")
            with self.assertRaises(AMEVATensorDeviceError):
                cpu_tensor @ gpu_tensor

if __name__ == '__main__':
    unittest.main()
