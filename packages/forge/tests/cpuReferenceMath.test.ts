/**
 * cpuReferenceMath.test.ts ??CPU Analytical Reference Math Verification
 *
 * Verifies that reference math formulas (MatMul, ReLU, MSELoss)
 * produce mathematically precise Float32 reference values.
 */

describe('CPU Reference Analytical Math Verification', () => {
  const eps = 1e-5;

  function float32Equal(a: Float32Array, b: Float32Array, tolerance = eps): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (Math.abs(a[i] - b[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  it('verifies MatMul forward formula (C = A x B)', () => {
    // 2x3 matrix A
    const A = new Float32Array([1, 2, 3, 4, 5, 6]);
    // 3x2 matrix B
    const B = new Float32Array([7, 8, 9, 1, 2, 3]);

    // Analytical reference result (CPU):
    // Row 0: [1*7 + 2*9 + 3*2, 1*8 + 2*1 + 3*3] = [7+18+6, 8+2+9] = [31, 19]
    // Row 1: [4*7 + 5*9 + 6*2, 4*8 + 5*1 + 6*3] = [28+45+12, 32+5+18] = [85, 55]
    const expected = new Float32Array([31, 19, 85, 55]);

    const M = 2, N = 2, K = 3;
    const cpuResult = new Float32Array(M * N);
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += A[m * K + k] * B[k * N + n];
        }
        cpuResult[m * N + n] = sum;
      }
    }

    expect(float32Equal(cpuResult, expected)).toBe(true);
  });

  it('verifies ReLU forward formula', () => {
    const input = new Float32Array([-2.5, 0.0, 3.14, -0.001, 100.0]);
    const expected = new Float32Array([0.0, 0.0, 3.14, 0.0, 100.0]);

    const cpuResult = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      cpuResult[i] = Math.max(0, input[i]);
    }

    expect(float32Equal(cpuResult, expected)).toBe(true);
  });

  it('verifies MSELoss forward formula', () => {
    const pred = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const target = new Float32Array([1.5, 1.5, 3.5, 3.5]);

    // diff = [-0.5, 0.5, -0.5, 0.5]
    // sq = [0.25, 0.25, 0.25, 0.25]
    // mean = 1.0 / 4.0 = 0.25
    let sumSq = 0;
    for (let i = 0; i < pred.length; i++) {
      const diff = pred[i] - target[i];
      sumSq += diff * diff;
    }
    const mse = sumSq / pred.length;

    expect(Math.abs(mse - 0.25)).toBeLessThan(eps);
  });
});
