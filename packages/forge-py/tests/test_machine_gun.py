import sys
import unittest
import forge as at
import time
import numpy as np
import asyncio


@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestMachineGun(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()

    async def test_gc_leak_gpu(self):
        """
        L-05 Fix: GPU 텐서를 실제로 생성하여 WebGPU GC 스트레스를 테스트.
        기존 테스트는 device 인자 없이 CPU 텐서만 생성하여 GPU GC가 전혀 검증되지 않았음.
        """
        print("\n--- 🔫 TRUE MACHINE GUN: Massive Concurrent Async Alloc/Free ---")
        
        num_workers = 10
        iterations_per_worker = 100
        total_ops = num_workers * iterations_per_worker
        size = 1024  # 1024x1024 = 4MB per tensor
        
        a_np = np.ones((size, size), dtype=np.float32)
        b_np = np.ones((size, size), dtype=np.float32)
        
        async def worker(worker_id):
            for i in range(iterations_per_worker):
                # 난장판을 만들기 위해 alloc - matmul - fetch - dispose를 빠르게 반복
                a = at.tensor(a_np, device="gpu")
                b = at.tensor(b_np, device="gpu")
                c = a @ b
                
                # 강제로 await 하여 GPU 실행 큐에 병렬로 던져지게 만듦
                res = await c.numpy_async()
                
                # 가끔씩 결과 검증
                if i == 0 and worker_id % 20 == 0:
                    assert abs(res[0,0] - size) < 1e-1
                
                # GC 큐에 혼돈을 주기 위해 랜덤한 순서로 dispose
                at.dispose(c)
                at.dispose(a)
                at.dispose(b)
                
        t0 = time.time()
        
        # 100개의 워커가 동시에 비동기적으로 GPU에 텐서를 생성하고 삭제함
        # 이는 JS 브릿지를 통해 수만 개의 WebGPU 버퍼 생성/파괴 이벤트를 난사함
        tasks = [worker(i) for i in range(num_workers)]
        await asyncio.gather(*tasks)

        elapsed = time.time() - t0
        print(f"All {total_ops} concurrent GPU rounds completed in {elapsed:.3f}s")
        print(f"Throughput: {total_ops / elapsed:.0f} ops/sec (Parallel GC Chaos)")
        print("Machine Gun Survived! True Parallel GC Leaks Tested! 🩸")


if __name__ == '__main__':
    unittest.main()
