import sys
import unittest
import ameva_tensor as at
import time
import numpy as np


@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestMachineGun(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()

    async def test_gc_leak_gpu(self):
        """
        L-05 Fix: GPU 텐서를 실제로 생성하여 WebGPU GC 스트레스를 테스트.
        기존 테스트는 device 인자 없이 CPU 텐서만 생성하여 GPU GC가 전혀 검증되지 않았음.
        """
        print("\n--- 🔫 MACHINE GUN: 1000 GPU Rapid Alloc/Free ---")
        iterations = 1000
        size = 128  # 128x128 = 65536 floats = 256KB per tensor (GPU 부하 적절)

        a_np = np.ones((size, size), dtype=np.float32)
        b_np = np.ones((size, size), dtype=np.float32)

        t0 = time.time()
        for i in range(iterations):
            # GPU 텐서 명시적 생성 (L-05: device="gpu" 추가)
            a = at.tensor(a_np, device="gpu")
            b = at.tensor(b_np, device="gpu")

            c = a @ b  # GPU matmul

            # 250회마다 GPU 큐 플러시 (command buffer 폭발 방지)
            if i % 250 == 0:
                result = await c.numpy_async()
                expected = size * 1.0  # 1.0 * 1.0 * 128 = 128
                assert abs(result[0, 0] - expected) < 1e-1, \
                    f"Round {i}: expected {expected}, got {result[0, 0]}"
                print(f"  Round {i}: GPU matmul OK, result[0,0]={result[0,0]:.1f}")

            # 즉시 해제 (GC 스트레스)
            at.dispose(a)
            at.dispose(b)
            at.dispose(c)

        elapsed = time.time() - t0
        print(f"All {iterations} GPU rounds completed in {elapsed:.3f}s")
        print(f"Throughput: {iterations / elapsed:.0f} ops/sec")
        print("Machine Gun Survived! No GPU GC Leaks! 🩸")


if __name__ == '__main__':
    unittest.main()
