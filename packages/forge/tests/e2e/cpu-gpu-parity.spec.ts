import { test, expect } from '@playwright/test';

const ABS_TOL = 1e-5;
const REL_TOL = 1e-4;

function close(actual: number, expected: number): boolean {
  return Math.abs(actual - expected) <= ABS_TOL + REL_TOL * Math.abs(expected);
}

test('WebGPU tensor operations match CPU reference math', async ({ page }) => {
  await page.goto('/test.html');
  const gpuAvailable = await page.evaluate(async () => {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  });
  const releaseGate = test.info().project.name === 'release-webgpu';
  if (!gpuAvailable && releaseGate) {
    throw new Error('RELEASE_GATE_WEBGPU_UNAVAILABLE: A physical WebGPU adapter is required.');
  }
  if (!gpuAvailable) {
    if (process.env.REQUIRE_WEBGPU === '1') {
      throw new Error('REQUIRE_WEBGPU=1 but WebGPU adapter is unavailable');
    }
    test.skip(!gpuAvailable, 'WebGPU adapter is unavailable outside release-webgpu project');
  }

  const result = await page.evaluate(async () => {
    const forge = (window as any).forge;
    if (!forge || typeof forge.flushGC !== 'function') {
      throw new Error('Forge browser API or flushGC unavailable');
    }
    await forge.initWebGPU();

    const handles: string[] = [];
    try {
      // Inputs: A = [[1, 2], [3, 4]], B = [[2, 0], [1, 2]]
      const a = forge.uploadFloat32Array(new Float32Array([1, 2, 3, 4]), [2, 2]);
      const b = forge.uploadFloat32Array(new Float32Array([2, 0, 1, 2]), [2, 2]);
      handles.push(a, b);

      const graph = [
        { op: 'load', id: 1, handle: a, shape: [2, 2] },
        { op: 'load', id: 2, handle: b, shape: [2, 2] },
        // id 3: matmul(A, B) -> [[4, 4], [10, 8]]
        { op: 'matmul', id: 3, shape: [2, 2], in: [1, 2], params: [2, 2, 2] },
        // id 4: relu(matmul) -> [[4, 4], [10, 8]]
        { op: 'relu', id: 4, shape: [2, 2], in: [3], params: [] },
        // id 5: add(relu, A) -> [[5, 6], [13, 12]]
        { op: 'add', id: 5, shape: [2, 2], in: [4, 1], params: [] },
        // id 6: sub(add, B) -> [[3, 6], [12, 10]]
        { op: 'sub', id: 6, shape: [2, 2], in: [5, 2], params: [] },
        // id 7: mul(sub, A) -> [[3, 12], [36, 40]]
        { op: 'mul', id: 7, shape: [2, 2], in: [6, 1], params: [] },
        // id 8: neg(A) -> [[-1, -2], [-3, -4]]
        { op: 'neg', id: 8, shape: [2, 2], in: [1], params: [] },
      ];

      const outputs = await forge.executeGraph(JSON.stringify(graph), []);
      const outIds = [5, 6, 7, 8];
      const actuals: Record<number, number[]> = {};

      for (const id of outIds) {
        const handle = outputs[id];
        if (!handle) throw new Error(`Missing handle for node ${id}`);
        handles.push(handle);
        await forge.mapBufferAsync(handle);
        const out = new Float32Array(4);
        forge.readMappedInto(handle, out);
        actuals[id] = Array.from(out);
      }

      return {
        actualAdd: actuals[5],
        expectedAdd: [5, 6, 13, 12],
        actualSub: actuals[6],
        expectedSub: [3, 6, 12, 10],
        actualMul: actuals[7],
        expectedMul: [3, 12, 36, 40],
        actualNeg: actuals[8],
        expectedNeg: [-1, -2, -3, -4],
      };
    } finally {
      for (const handle of [...new Set(handles)].reverse()) {
        try { forge.dispose(handle); } catch { /* cleanup best effort */ }
      }
      forge.flushGC();
    }
  });

  const pairs = [
    [result.actualAdd, result.expectedAdd],
    [result.actualSub, result.expectedSub],
    [result.actualMul, result.expectedMul],
    [result.actualNeg, result.expectedNeg],
  ];

  for (const [actual, expected] of pairs) {
    expect(actual).toHaveLength(expected.length);
    actual.forEach((val: number, idx: number) => {
      expect(close(val, expected[idx])).toBe(true);
    });
  }
});
