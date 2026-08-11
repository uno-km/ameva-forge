import sys
import unittest
import ameva_tensor as at
import time
import numpy as np

@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestVRAMCrusher(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()
        print(f"Device: {at.current_device().upper()}")

    async def test_12288_matmul(self):
        size = 12288
        print(f"\n--- 🧨 VRAM CRUSHER: {size}x{size} Matrix Multiplication ---")
        
        # 12288x12288 float32 is ~603 MB per tensor. A, B, C = ~1.8 GB VRAM
        t0 = time.time()
        a_np = np.full((size, size), 0.1, dtype=np.float32)
        b_np = np.full((size, size), 0.1, dtype=np.float32)
        print(f"Host allocation: {time.time() - t0:.3f}s, ~1.2 GB RAM (Host)")
        
        t0 = time.time()
        a = at.tensor(a_np, device=at.current_device())
        b = at.tensor(b_np, device=at.current_device())
        print(f"WebGPU Upload: {time.time() - t0:.3f}s")
        
        t0 = time.time()
        c = a @ b
        print(f"WebGPU Compute Dispatch: {time.time() - t0:.3f}s")
        
        t0 = time.time()
        c_np = await c.numpy_async()
        print(f"Zero-Copy readback time: {time.time() - t0:.3f}s")
        
        # Validation
        self.assertAlmostEqual(c_np[0, 0], 0.01 * size, places=0)
        print("VRAM Crusher Survived! Maximum Hardware Limit Tested. 🩸")
        
        at.dispose(a)
        at.dispose(b)
        at.dispose(c)

if __name__ == '__main__':
    unittest.main()
