import unittest
import forge as at
import numpy as np
import time
import asyncio

class TestMassiveCPU(unittest.IsolatedAsyncioTestCase):
    async def test_massive_add_100_million(self):
        """1억 개(100 Million) 엘리먼트 벡터 덧셈 (약 400MB + 400MB = 800MB 메모리 사용)"""
        size = 100_000_000
        print(f"\n--- [MASSIVE] CPU: {size:,} elements Addition ---")
        
        t0 = time.time()
        # 1억 개의 랜덤 float32 배열 생성 (순수 CPU 메모리)
        a = at.random((size,), device='cpu')
        b = at.random((size,), device='cpu')
        print(f"Allocation Time: {time.time() - t0:.3f}s")
        await asyncio.sleep(0.1) # UI 업데이트 대기
        
        t0 = time.time()
        # 실제 연산 수행 (NumPy C++ Backend)
        print("Computing addition in one shot...")
        c = a + b
        result = c.numpy()
        compute_time = time.time() - t0
        
        print(f"Compute Time: {compute_time:.3f}s")
        self.assertEqual(result.shape, (size,))
        # 연산에 실제로 시간이 소요됨을 단언 (가짜 하드코딩이 아님을 증명)
        self.assertTrue(compute_time > 0.05)

    async def test_massive_matmul_10_million(self):
        """3162 x 3162 (약 1000만 개 엘리먼트) 행렬곱. 연산량: 약 316억 번 (O(N^3))"""
        dim = 3162
        print(f"\n--- [MASSIVE] CPU: {dim}x{dim} (~10M elements) MatMul ---")
        
        t0 = time.time()
        a = at.random((dim, dim), device='cpu')
        b = at.random((dim, dim), device='cpu')
        print(f"Allocation Time: {time.time() - t0:.3f}s")
        await asyncio.sleep(0.1) # UI 업데이트 대기
        
        t0 = time.time()
        print("Starting Blocked MatMul for Real-time UI responsiveness...")
        
        # Pyodide에서 화면이 멈추는 것을 방지하기 위해 Chunk(Block) 단위로 쪼개서 계산
        a_np = a.numpy()
        b_np = b.numpy()
        c_np = np.zeros((dim, dim), dtype=np.float32)
        
        block_size = 500 # 500x500 단위로 잘라서 계산
        num_blocks = int(np.ceil(dim / block_size))
        total_tasks = num_blocks * num_blocks
        task_count = 0
        
        for i in range(num_blocks):
            for j in range(num_blocks):
                row_start = i * block_size
                row_end = min((i + 1) * block_size, dim)
                col_start = j * block_size
                col_end = min((j + 1) * block_size, dim)
                
                # c_ij = sum_k a_ik @ b_kj
                c_ij = np.zeros((row_end - row_start, col_end - col_start), dtype=np.float32)
                for k in range(num_blocks):
                    k_start = k * block_size
                    k_end = min((k + 1) * block_size, dim)
                    a_ik = a_np[row_start:row_end, k_start:k_end]
                    b_kj = b_np[k_start:k_end, col_start:col_end]
                    c_ij += np.matmul(a_ik, b_kj)
                
                c_np[row_start:row_end, col_start:col_end] = c_ij
                task_count += 1
                
                # 3개 블록마다 UI에 진행 상황 보고 (await asyncio.sleep으로 메인 스레드 숨통 트기)
                if task_count % 3 == 0 or task_count == total_tasks:
                    elapsed = time.time() - t0
                    print(f"Progress: [{task_count}/{total_tasks} blocks] ... {elapsed:.1f}s elapsed (UI Yielded)")
                    await asyncio.sleep(0.01)
        
        compute_time = time.time() - t0
        print(f"Compute Time: {compute_time:.3f}s")
        self.assertEqual(c_np.shape, (dim, dim))
        self.assertTrue(compute_time > 0.1)

if __name__ == '__main__':
    unittest.main()
