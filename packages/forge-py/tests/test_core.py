import asyncio
import sys
import unittest
import numpy as np
import forge as at
from forge.errors import (
    AMEVAForgeDeviceError,
    AMEVAForgeShapeError,
    AMEVAForgeDisposedError
)

@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestAMEVAForgeCore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()
        self.device = at.current_device()

    async def test_tensor_creation_and_zerocopy(self):
        print("\n--- 💥 EXTREME LIMITS: Zero-Copy (2048x2048) ---")
        # 1. Zero-Copy Upload and Readback test (16MB per tensor)
        arr = np.random.randn(2048, 2048).astype(np.float32)
        t = at.tensor(arr, device=self.device)
        self.assertEqual(t.shape, (2048, 2048))
        
        res = await t.numpy_async()
        np.testing.assert_allclose(res, arr, err_msg="Zero copy readback data mismatch")

    async def test_add_rigorous(self):
        print("\n--- 💥 EXTREME LIMITS: Add (4096x4096) ---")
        import time
        a_np = np.random.randn(4096, 4096).astype(np.float32)
        b_np = np.random.randn(4096, 4096).astype(np.float32)
        
        a = at.tensor(a_np, device=self.device)
        b = at.tensor(b_np, device=self.device)
        
        t0 = time.time()
        c = at.add(a, b)
        c_np = await c.numpy_async()
        print(f"WebGPU Add (16.7M elements): {time.time() - t0:.3f}s")
        
        expected = a_np + b_np
        np.testing.assert_allclose(c_np, expected, rtol=1e-4, err_msg="Add math mismatch")

    async def test_mul_rigorous(self):
        print("\n--- 💥 EXTREME LIMITS: Mul (4096x4096) ---")
        import time
        a_np = np.random.randn(4096, 4096).astype(np.float32)
        b_np = np.random.randn(4096, 4096).astype(np.float32)
        
        a = at.tensor(a_np, device=self.device)
        b = at.tensor(b_np, device=self.device)
        
        t0 = time.time()
        c = at.mul(a, b)
        c_np = await c.numpy_async()
        print(f"WebGPU Mul (16.7M elements): {time.time() - t0:.3f}s")
        
        expected = a_np * b_np
        np.testing.assert_allclose(c_np, expected, rtol=1e-4, err_msg="Mul math mismatch")

    async def test_matmul_rigorous(self):
        print("\n--- 💥 EXTREME LIMITS: MatMul (2048x2048) ---")
        import time
        a_np = np.random.randn(2048, 2048).astype(np.float32)
        b_np = np.random.randn(2048, 2048).astype(np.float32)
        
        a = at.tensor(a_np, device=self.device)
        b = at.tensor(b_np, device=self.device)
        
        t0 = time.time()
        c = a @ b
        c_np = await c.numpy_async()
        print(f"WebGPU MatMul (17 GFLOPS): {time.time() - t0:.3f}s")
        
        expected = np.matmul(a_np, b_np)
        np.testing.assert_allclose(c_np, expected, rtol=1e-3, err_msg="Matmul math mismatch")

    async def test_relu_rigorous(self):
        print("\n--- 💥 EXTREME LIMITS: ReLU (4096x4096) ---")
        import time
        a_np = np.random.randn(4096, 4096).astype(np.float32)
        
        a = at.tensor(a_np, device=self.device)
        t0 = time.time()
        b = at.relu(a)
        
        b_np = await b.numpy_async()
        print(f"WebGPU ReLU (16.7M elements): {time.time() - t0:.3f}s")
        expected = np.maximum(a_np, 0.0)
        
        np.testing.assert_allclose(b_np, expected, err_msg="ReLU math mismatch")

    async def test_transpose_rigorous(self):
        print("\n--- 💥 EXTREME LIMITS: Transpose (4096x4096) ---")
        import time
        a_np = np.random.randn(4096, 4096).astype(np.float32)
        a = at.tensor(a_np, device=self.device)
        
        t0 = time.time()
        b = at.transpose(a)
        b_np = await b.numpy_async()
        print(f"WebGPU Transpose (16.7M elements): {time.time() - t0:.3f}s")
        
        expected = np.transpose(a_np)
        np.testing.assert_allclose(b_np, expected, err_msg="Transpose math mismatch")

    def test_mixed_device_error(self):
        if self.device == "gpu":
            cpu_tensor = at.tensor([1, 2], device="cpu")
            gpu_tensor = at.tensor([1, 2], device="gpu")
            with self.assertRaises(AMEVAForgeDeviceError):
                cpu_tensor @ gpu_tensor

if __name__ == '__main__':
    unittest.main()
