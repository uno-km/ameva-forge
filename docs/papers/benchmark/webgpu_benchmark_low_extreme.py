import time
import numpy as np
import asyncio

print("===============================================================")
print(" AMEVA OS WebGPU-Python Bridge: EXTREME Benchmark Suite v2.0")
print(" [!] WARNING: This script is temporarily DISABLED (UNVERIFIED).")
print("===============================================================")
raise RuntimeError("Synthetic benchmark execution is blocked in Release 1. Run via 'npm test' in browser harness instead.")

async def run_low_extreme_benchmark():
    print("===============================================================")
    print(" AMEVA OS WebGPU-Python Bridge: LOW-EXTREME Benchmark Suite")
    print(" (OOM/크래시 직전의 극한 스트레스 테스트 - 100% 실측치 도출용)")
    print("===============================================================")
    
    # --- Test 1: 대규모 스칼라 연산 (1억 단위 배열) ---
    print("\n[Test 1] 1억 개(100 Million) 배열 요소 사칙연산 (Safe Memory Limit)")
    size_1 = 100_000_000 # 약 400MB
    print(" -> [CPU] 1억 개 배열 메모리 할당 및 연산 진행...")
    
    start_cpu = time.time()
    A = np.random.rand(size_1).astype(np.float32)
    B = np.random.rand(size_1).astype(np.float32)
    _ = A * B + A
    cpu_time_1 = time.time() - start_cpu
    print(f" -> [CPU] 소요 시간: {cpu_time_1:.5f}초 (OOM 회피 성공)")
        
    start_gpu = time.time()
    A_gpu = await at.random((size_1,))
    B_gpu = await at.random((size_1,))
    mul_res = await at.mul(A_gpu, B_gpu)
    _ = await at.add(mul_res, A_gpu)
    gpu_time_1 = time.time() - start_gpu
    print(f" -> [WebGPU] 소요 시간: {gpu_time_1:.5f}초")
    print(f" ==> 가속 배수: {cpu_time_1 / max(gpu_time_1, 0.0001):.2f}x")
    
    # --- Test 2: 부동소수점 수학 연산 (5천만 단위 삼각함수) ---
    print("\n[Test 2] 5천만 개(50 Million) 부동소수점 복합 삼각함수")
    size_2 = 50_000_000
    
    C = np.random.rand(size_2).astype(np.float32)
    start_cpu = time.time()
    _ = np.sin(C) ** 2 + np.cos(C) ** 2
    cpu_time_2 = time.time() - start_cpu
    print(f" -> [CPU] 소요 시간: {cpu_time_2:.5f}초")

    start_gpu = time.time()
    C_gpu = await at.random((size_2,))
    sin_res = await at.sin(C_gpu)
    cos_res = await at.cos(C_gpu)
    _ = await at.add(await at.mul(sin_res, sin_res), await at.mul(cos_res, cos_res))
    gpu_time_2 = time.time() - start_gpu
    print(f" -> [WebGPU] 소요 시간: {gpu_time_2:.5f}초")
    print(f" ==> 가속 배수: {cpu_time_2 / max(gpu_time_2, 0.0001):.2f}x")
    
    # --- Test 3: 행렬 곱 (1024 x 1024) ---
    print("\n[Test 3] 1024 x 1024 거대 행렬 곱 (약 1 Billion MACs)")
    N = 1024
    M1 = np.random.rand(N, N).astype(np.float32)
    M2 = np.random.rand(N, N).astype(np.float32)
    
    start_cpu = time.time()
    _ = np.dot(M1, M2)
    cpu_time_3 = time.time() - start_cpu
    print(f" -> [CPU] 소요 시간: {cpu_time_3:.5f}초")

    start_gpu = time.time()
    M1_g = await at.random((N, N))
    M2_g = await at.random((N, N))
    _ = await at.matmul(M1_g, M2_g)
    gpu_time_3 = time.time() - start_gpu
    print(f" -> [WebGPU] 소요 시간: {gpu_time_3:.5f}초")
    print(f" ==> 가속 배수: {cpu_time_3 / max(gpu_time_3, 0.0001):.2f}x")
    
    print("\n===============================================================")
    print(" LOW-EXTREME Benchmark Complete.")

if __name__ == "__main__":
    asyncio.run(run_low_extreme_benchmark())
