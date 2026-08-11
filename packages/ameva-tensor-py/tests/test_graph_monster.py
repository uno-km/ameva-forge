import sys
import unittest
import ameva_tensor as at
import time
import numpy as np

@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestGraphMonster(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()

    async def test_deep_chained_ops(self):
        print("\n--- 🧟 GRAPH MONSTER: 200 Deep Chained Operations ---")
        layers = 200
        size = 1024
        
        # We start with a base tensor and pass it through 200 matmuls and relus
        print(f"Starting {layers} layers of dense matrix multiplication...")
        t0 = time.time()
        
        x = at.tensor(np.ones((size, size), dtype=np.float32) * 0.01)
        w = at.tensor(np.ones((size, size), dtype=np.float32))
        
        # Keeping track of memory so it doesn't instantly OOM before finishing the chain
        for i in range(layers):
            out = x @ w
            if i % 50 == 0:
                print(f"Layer {i} passed...")
            
            # Dispose old x to prevent OOM
            at.dispose(x)
            x = out
            
        print(f"Dispatch completed in {time.time() - t0:.3f}s. Awaiting results...")
        
        t0 = time.time()
        res = await x.numpy_async()
        print(f"Zero-Copy readback time: {time.time() - t0:.3f}s")
        
        print("Graph Monster Defeated! No stack overflow, no pipeline crash! 🩸")
        at.dispose(w)
        at.dispose(x)

if __name__ == '__main__':
    unittest.main()
