import time
import numpy as np
import asyncio
import ameva_tensor as at

async def run_benchmark():
    print("==================================================")
    print(" AMEVA OS WebGPU-Python Bridge Benchmark Suite")
    print("==================================================")
    
    # --- Test 1: 극한의 단순 연산 (4중 For-loop / Element-wise) ---
    print("\n[Test 1] 대규모 스칼라 연산 (Element-wise Addition/Multiplication)")
    size = 10000000  # 1천만 개 배열
    A = np.random.rand(size).astype(np.float32)
    B = np.random.rand(size).astype(np.float32)
    
    # 1. CPU (Numpy)
    start_cpu = time.time()
    _ = A * B + A
    end_cpu = time.time()
    cpu_time_1 = end_cpu - start_cpu
    print(f" -> CPU (Numpy) 소요 시간: {cpu_time_1:.5f}초")
    
    # 2. WebGPU
    start_gpu = time.time()
    mul_res = await at.mul(A, B)
    _ = await at.add(mul_res, A)
    end_gpu = time.time()
    gpu_time_1 = end_gpu - start_gpu
    print(f" -> WebGPU (adeva_tensor) 소요 시간: {gpu_time_1:.5f}초")
    print(f" ==> 성능 향상: {cpu_time_1 / max(gpu_time_1, 0.0001):.2f}배")
    
    # --- Test 2: 수학적 복합 연산 (Trigonometric / 3D Math) ---
    print("\n[Test 2] 복합 부동소수점 수학 연산 (Sine / Cosine)")
    # 3D 렌더링에 자주 쓰이는 삼각함수 연산 시뮬레이션
    size_2 = 5000000 # 5백만 개
    C = np.random.rand(size_2).astype(np.float32) * 100.0
    
    start_cpu = time.time()
    _ = np.sin(C) ** 2 + np.cos(C) ** 2
    end_cpu = time.time()
    cpu_time_2 = end_cpu - start_cpu
    print(f" -> CPU (Numpy) 소요 시간: {cpu_time_2:.5f}초")
    
    start_gpu = time.time()
    sin_res = await at.sin(C)
    cos_res = await at.cos(C)
    sin_sq = await at.mul(sin_res, sin_res)
    cos_sq = await at.mul(cos_res, cos_res)
    _ = await at.add(sin_sq, cos_sq)
    end_gpu = time.time()
    gpu_time_2 = end_gpu - start_gpu
    print(f" -> WebGPU (adeva_tensor) 소요 시간: {gpu_time_2:.5f}초")
    print(f" ==> 성능 향상: {cpu_time_2 / max(gpu_time_2, 0.0001):.2f}배")
    
    # --- Test 3: 머신러닝 라이브러리 수준 (행렬 곱 / Matmul) ---
    print("\n[Test 3] 대규모 머신러닝 행렬 가중치 연산 (Matrix Multiplication)")
    N = 2048
    M1 = np.random.rand(N, N).astype(np.float32)
    M2 = np.random.rand(N, N).astype(np.float32)
    
    start_cpu = time.time()
    _ = np.dot(M1, M2)
    end_cpu = time.time()
    cpu_time_3 = end_cpu - start_cpu
    print(f" -> CPU (Numpy dot) 소요 시간: {cpu_time_3:.5f}초")
    
    start_gpu = time.time()
    _ = await at.matmul(M1, M2)
    end_gpu = time.time()
    gpu_time_3 = end_gpu - start_gpu
    print(f" -> WebGPU (adeva_tensor) 소요 시간: {gpu_time_3:.5f}초")
    print(f" ==> 성능 향상: {cpu_time_3 / max(gpu_time_3, 0.0001):.2f}배")
    
    print("\n==================================================")
    print(" Benchmark Complete.")

# 미니 콜랩 환경에서는 비동기 실행이 필요함
asyncio.ensure_future(run_benchmark())
