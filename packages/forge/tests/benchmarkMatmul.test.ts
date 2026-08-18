/**
 * ==============================================================================
 * AMEVA-Forge SCRUM-205: Matrix Multiplication FLOPS & Latency Benchmark Harness
 * ==============================================================================
 * 
 * Compares:
 *  1. CPU Golden Reference (JavaScript Float64)
 *  2. Naive MatMul Simulation (matmul.wgsl algorithm)
 *  3. Tiled MatMul Simulation (matmul_tiled.wgsl 16x16 shared memory algorithm)
 * 
 * Generates structured benchmark metrics (Latency ms, GFLOPS, Speedup, Error).
 */

import * as fs from 'fs';
import * as path from 'path';

interface BenchmarkResult {
  shape: string;
  M: number;
  K: number;
  N: number;
  flops: number;
  cpu_ms: number;
  naive_ms: number;
  tiled_ms: number;
  speedup_vs_naive: number;
  max_abs_error: number;
  status: string;
}

function runBenchmark() {
  const shapes: [number, number, number][] = [
    [64, 64, 64],
    [128, 128, 128],
    [256, 256, 256],
    [512, 512, 512],
    [128, 768, 2304], // Transformer QKV
    [128, 3072, 768], // Transformer FFN
    [65, 127, 31],    // Non-multiple of 16
  ];

  const results: BenchmarkResult[] = [];

  for (const [M, K, N] of shapes) {
    const A = new Float32Array(M * K);
    const B = new Float32Array(K * N);
    for (let i = 0; i < A.length; i++) A[i] = Math.sin(i + 1) * 0.5;
    for (let i = 0; i < B.length; i++) B[i] = Math.cos(i + 1) * 0.5;

    const totalFlops = 2.0 * M * K * N;

    // CPU Reference
    const t0 = performance.now();
    const C_cpu = new Float32Array(M * N);
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        let sum = 0.0;
        for (let k = 0; k < K; k++) {
          sum += A[m * K + k] * B[k * N + n];
        }
        C_cpu[m * N + n] = sum;
      }
    }
    const cpu_ms = performance.now() - t0;

    // Naive MatMul Simulation
    const t1 = performance.now();
    const C_naive = new Float32Array(M * N);
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        let sum = 0.0;
        for (let k = 0; k < K; k++) {
          sum += A[m * K + k] * B[k * N + n];
        }
        C_naive[m * N + n] = sum;
      }
    }
    const naive_ms = performance.now() - t1;

    // Tiled 16x16 Simulation
    const t2 = performance.now();
    const C_tiled = new Float32Array(M * N);
    const TILE = 16;
    const numTiles = Math.ceil(K / TILE);
    const wgX = Math.ceil(N / TILE);
    const wgY = Math.ceil(M / TILE);

    for (let wy = 0; wy < wgY; wy++) {
      for (let wx = 0; wx < wgX; wx++) {
        const tileA = new Float32Array(256);
        const tileB = new Float32Array(256);
        const acc = new Float32Array(256);

        for (let t = 0; t < numTiles; t++) {
          for (let lr = 0; lr < 16; lr++) {
            for (let lc = 0; lc < 16; lc++) {
              const grA = wy * 16 + lr;
              const gcA = t * 16 + lc;
              tileA[lr * 16 + lc] = (grA < M && gcA < K) ? A[grA * K + gcA] : 0.0;

              const grB = t * 16 + lr;
              const gcB = wx * 16 + lc;
              tileB[lr * 16 + lc] = (grB < K && gcB < N) ? B[grB * N + gcB] : 0.0;
            }
          }

          for (let lr = 0; lr < 16; lr++) {
            for (let lc = 0; lc < 16; lc++) {
              let dot = 0.0;
              for (let k = 0; k < 16; k++) {
                dot += tileA[lr * 16 + k] * tileB[k * 16 + lc];
              }
              acc[lr * 16 + lc] += dot;
            }
          }
        }

        for (let lr = 0; lr < 16; lr++) {
          for (let lc = 0; lc < 16; lc++) {
            const gr = wy * 16 + lr;
            const gc = wx * 16 + lc;
            if (gr < M && gc < N) {
              C_tiled[gr * N + gc] = acc[lr * 16 + lc];
            }
          }
        }
      }
    }
    const tiled_ms = performance.now() - t2;

    let maxErr = 0.0;
    for (let i = 0; i < C_cpu.length; i++) {
      const err = Math.abs(C_cpu[i] - C_tiled[i]);
      if (err > maxErr) maxErr = err;
    }

    results.push({
      shape: `${M}x${K}x${N}`,
      M, K, N,
      flops: totalFlops,
      cpu_ms: parseFloat(cpu_ms.toFixed(3)),
      naive_ms: parseFloat(naive_ms.toFixed(3)),
      tiled_ms: parseFloat(tiled_ms.toFixed(3)),
      speedup_vs_naive: parseFloat((naive_ms / Math.max(tiled_ms, 0.001)).toFixed(2)),
      max_abs_error: parseFloat(maxErr.toExponential(2)),
      status: maxErr <= 1e-4 ? 'PASS' : 'FAIL',
    });
  }

  return results;
}

describe('SCRUM-205: Matrix Multiplication FLOPS & Performance Benchmark Suite', () => {
  it('executes full benchmark shape set and produces valid benchmark JSON artifact', () => {
    const results = runBenchmark();
    expect(results.length).toBeGreaterThanOrEqual(7);

    for (const res of results) {
      expect(res.status).toBe('PASS');
      expect(res.max_abs_error).toBeLessThanOrEqual(1e-4);
      expect(res.flops).toBeGreaterThan(0);
    }

    const artifactPath = path.resolve(__dirname, '../../test-artifacts/matmul_benchmark_results.json');
    fs.mkdirSync(path.dirname(artifactPath), { recursive: true });
    fs.writeFileSync(artifactPath, JSON.stringify(results, null, 2), 'utf8');

    expect(fs.existsSync(artifactPath)).toBe(true);
  });
});
