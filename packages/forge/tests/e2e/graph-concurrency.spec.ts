import { test, expect } from '@playwright/test';

test('pre-GPU validation failure does not block the next graph', async ({ page }) => {
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
      throw new Error('flushGC API is required for the graph concurrency gate');
    }
    await forge.initWebGPU();

    const baseline = forge.getQuotaSnapshot();
    let a: string | null = null;
    let b: string | null = null;
    let successOut: string | null = null;

    try {
      a = forge.uploadFloat32Array(new Float32Array([1, 2, 3, 4]), [2, 2]);
      b = forge.uploadFloat32Array(new Float32Array([2, 0, 1, 2]), [2, 2]);

      // 첫 graph는 matmul params를 의도적으로 잘못 전달해 pre-GPU schema validation에서 실패시킨다.
      const failingGraph = [
        { op: 'load', id: 1, handle: a, shape: [2, 2] },
        { op: 'load', id: 2, handle: b, shape: [2, 2] },
        { op: 'matmul', id: 3, shape: [2, 2], in: [1, 2], params: [2, 2] },
      ];

      const validGraph = [
        { op: 'load', id: 11, handle: a, shape: [2, 2] },
        { op: 'load', id: 12, handle: b, shape: [2, 2] },
        { op: 'matmul', id: 13, shape: [2, 2], in: [11, 12], params: [2, 2, 2] },
      ];

      const [failed, succeeded] = await Promise.allSettled([
        forge.executeGraph(JSON.stringify(failingGraph), []),
        forge.executeGraph(JSON.stringify(validGraph), []),
      ]);

      if (succeeded.status === 'fulfilled') successOut = succeeded.value[13];

      let values: number[] = [];
      if (successOut) {
        await forge.mapBufferAsync(successOut);
        const out = new Float32Array(4);
        forge.readMappedInto(successOut, out);
        values = Array.from(out);
      }

      return {
        firstStatus: failed.status,
        secondStatus: succeeded.status,
        values,
        during: forge.getQuotaSnapshot(),
        baseline,
      };
    } finally {
      if (successOut) forge.dispose(successOut);
      if (b) forge.dispose(b);
      if (a) forge.dispose(a);
      forge.flushGC();
    }
  });

  expect(result.firstStatus).toBe('rejected');
  expect(result.secondStatus).toBe('fulfilled');
  expect(result.values).toEqual([4, 4, 10, 8]);
});
