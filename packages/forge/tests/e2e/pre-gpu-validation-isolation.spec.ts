import { test, expect } from '@playwright/test';

test.describe('Pre-GPU Validation Error Isolation Gate', () => {
  test('Pre-GPU dependency validation failure does not corrupt quota or block subsequent valid graph', async ({ page }) => {
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
      if (!forge) throw new Error('window.forge is undefined');
      await forge.initWebGPU();

      const baselineQuota = forge.getQuotaSnapshot();
      let firstGraphFailed = false;
      let firstErrorMessage = '';

      let hA: string | null = null;
      try {
        hA = forge.uploadFloat32Array(new Float32Array([1, 2, 3, 4]), [2, 2]);
        const failingInstructions = [
          { op: 'load', id: 1, handle: hA, shape: [2, 2], in: [], params: [] },
          { op: 'matmul', id: 2, shape: [2, 2], in: [1, 9999], params: [2, 2, 2] }
        ];
        await forge.executeGraph(JSON.stringify(failingInstructions), [], [2]);
      } catch (err: any) {
        firstGraphFailed = true;
        firstErrorMessage = err.message;
      } finally {
        if (hA) forge.dispose(hA);
      }

      // 2. Second graph: valid matmul
      const aArray = new Float32Array([1, 2, 3, 4]);
      const bArray = new Float32Array([2, 0, 1, 2]);
      const handleA = forge.uploadFloat32Array(aArray, [2, 2]);
      const handleB = forge.uploadFloat32Array(bArray, [2, 2]);

      const validInstructions = [
        { op: 'load', id: 1, handle: handleA, shape: [2, 2], in: [], params: [] },
        { op: 'load', id: 2, handle: handleB, shape: [2, 2], in: [], params: [] },
        { op: 'matmul', id: 3, shape: [2, 2], in: [1, 2], params: [2, 2, 2] }
      ];

      const outHandles = await forge.executeGraph(JSON.stringify(validInstructions), [], [3]);
      const outHandle = outHandles[3];

      await forge.mapBufferAsync(outHandle);
      const outData = new Float32Array(4);
      forge.readMappedInto(outHandle, outData);

      // Cleanup
      forge.dispose(handleA);
      forge.dispose(handleB);
      forge.dispose(outHandle);
      if (typeof forge.flushGC === 'function') await forge.flushGC();

      const finalQuota = forge.getQuotaSnapshot();

      return {
        firstGraphFailed,
        firstErrorMessage,
        validResult: Array.from(outData),
        usedBytesDiff: finalQuota.usedBytes - baselineQuota.usedBytes,
      };
    });

    expect(result.firstGraphFailed).toBe(true);
    expect(result.validResult).toEqual([4, 4, 10, 8]);
    expect(result.usedBytesDiff).toBe(0);
  });
});
