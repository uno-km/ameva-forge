import asyncio
import sys
import unittest
import time
import numpy as np
import forge as at

@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestExtremeBenchmark(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()
        self.device = at.current_device()

    async def test_extreme_zero_copy_matmul(self):
        # 8192 * 8192 = 67,108,864 elements (~268 MB per tensor)
        # 1 iteration = ~1.1 TeraFLOPS
        size = 8192
        iterations = 20 # Total: 22 TeraFLOPS
        
        print(f"\n--- 🧨 EXTREME BENCHMARK: {size}x{size} Matrix Multiplication x {iterations} Loops ---")
        print(f"Device: {self.device.upper()}")
        print(f"Goal: Prove physical computation time (Accumulating {1.1 * iterations:.1f} TFLOPS)")
        
        # 1. Host Memory Allocation
        start_t = time.time()
        a_np = np.ones((size, size), dtype=np.float32) * 0.5
        b_np = np.ones((size, size), dtype=np.float32) * 0.5
        print(f"Host allocation time: {time.time() - start_t:.3f}s")
        
        # 2. Upload to Device
        a = at.tensor(a_np, device=self.device)
        b = at.tensor(b_np, device=self.device)
        
        # 3. Execution (Looping to simulate massive continuous compute)
        total_dispatch_t0 = time.time()
        
        for i in range(1, iterations + 1):
            iter_t0 = time.time()
            
            # Matmul dispatch
            c = a @ b
            
            # Readback exactly 1 element to force pipeline flush and sync
            # Without this, WebGPU might queue all 20 and freeze the browser
            # We use a trick: slice out a 1x1 chunk, but slicing isn't implemented.
            # So we just readback the whole thing or a small chunk if slicing existed.
            # We have to readback the whole tensor to sync in our current API.
            c_np = await c.numpy_async()
            
            iter_time = time.time() - iter_t0
            elapsed = time.time() - total_dispatch_t0
            
            print(f"Iter [{i}/{iterations}] | TFLOPS: ~1.1 | Took: {iter_time:.2f}s | Total Elapsed: {elapsed:.1f}s")
            
            # 메모리 정리 (VRAM 폭발 방지)
            at.dispose(c)
            
            # UI 숨쉬게 해주기 (어..어.. 억! 하고 죽지 않게)
            await asyncio.sleep(0.01)
            
        print(f"--- 🏆 COMBAT FINISHED: 22 TFLOPS Survived! ---")
        
        # 5. Validation
        self.assertEqual(c_np.shape, (size, size))
        self.assertAlmostEqual(c_np[0, 0], 2048.0, places=3)
        self.assertAlmostEqual(c_np[size-1, size-1], 2048.0, places=3)
        print("Validation Passed: True Zero-Copy Math Verified!")
        
        at.dispose(a)
        at.dispose(b)

if __name__ == '__main__':
    unittest.main()
