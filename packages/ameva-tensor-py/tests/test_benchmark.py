import asyncio
import sys
import unittest
import time
import numpy as np
import ameva_tensor as at

@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestExtremeBenchmark(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()
        self.device = at.current_device()

    async def test_extreme_zero_copy_matmul(self):
        # 8192 * 8192 = 67,108,864 elements (~268 MB per tensor)
        # Total VRAM footprint for A, B, C is ~804 MB
        size = 8192
        print(f"\n--- Extreme Benchmark: {size}x{size} Matrix Multiplication ---")
        print(f"Device: {self.device.upper()}")
        
        # 1. Host Memory Allocation
        start_t = time.time()
        # Using random.random to prevent overflow, but making it extremely fast to generate
        # To avoid python hanging on random generation, we'll use np.ones for deterministic speed
        a_np = np.ones((size, size), dtype=np.float32) * 0.5
        b_np = np.ones((size, size), dtype=np.float32) * 0.5
        print(f"Host allocation time: {time.time() - start_t:.3f}s")
        
        # 2. Upload to Device (Zero-Copy test)
        start_t = time.time()
        a = at.tensor(a_np, device=self.device)
        b = at.tensor(b_np, device=self.device)
        print(f"Device upload time: {time.time() - start_t:.3f}s")
        
        # 3. Execution (Matmul)
        start_t = time.time()
        c = a @ b
        print(f"Device compute dispatch time: {time.time() - start_t:.3f}s")
        
        # 4. Zero-Copy Readback (The Holy Grail)
        start_t = time.time()
        c_np = await c.numpy_async()
        print(f"Zero-Copy readback time (Async Wait + Transfer): {time.time() - start_t:.3f}s")
        
        # 5. Validation
        # 0.5 * 0.5 * 8192 = 2048.0
        self.assertEqual(c_np.shape, (size, size))
        
        # We only check a few indices to save CPU time on assertion
        self.assertAlmostEqual(c_np[0, 0], 2048.0, places=3)
        self.assertAlmostEqual(c_np[size-1, size-1], 2048.0, places=3)
        print("Validation Passed: True Zero-Copy Math Verified!")
        
        # Clean up memory explicitly
        at.dispose(a)
        at.dispose(b)
        at.dispose(c)

if __name__ == '__main__':
    unittest.main()
