import sys
import unittest
import forge as at
import time
import numpy as np

@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestGraphMonster(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()

    async def test_deep_chained_ops(self):
        layers = 1000
        size = 2048
        print(f"\n--- 🧟 EXTREME GRAPH MONSTER: {layers} Deep Chained Operations ({size}x{size}) ---")
        
        # We start with a base tensor and pass it through 1000 matmuls
        print(f"Starting {layers} layers of dense matrix multiplication (Accumulating 8.5 TFLOPS)...")
        import asyncio
        t0 = time.time()
        
        x = at.tensor(np.ones((size, size), dtype=np.float32) * 0.001, device="gpu")
        w = at.tensor(np.ones((size, size), dtype=np.float32), device="gpu")
        
        # Keeping track of memory so it doesn't instantly OOM before finishing the chain
        for i in range(1, layers + 1):
            out = x @ w
            
            # Force execution so the lazy graph evaluates before we destroy the parent node
            out.realize()
            
            # Dispose old x to prevent OOM (GPU memory GC)
            at.dispose(x)
            x = out
            
            if i % 50 == 0:
                print(f"Graph Monster Layer [{i}/{layers}] passed... (UI Yielded)")
                await asyncio.sleep(0.01)
            
        print(f"Dispatch & Compute completed in {time.time() - t0:.3f}s. Awaiting final readback...")
        
        t0 = time.time()
        res = await x.numpy_async()
        print(f"Zero-Copy readback time: {time.time() - t0:.3f}s")
        
        print("--- 🩸 GRAPH MONSTER DEFEATED: 1000 Layers Survived! ---")
        at.dispose(w)
        at.dispose(x)

if __name__ == '__main__':
    unittest.main()
