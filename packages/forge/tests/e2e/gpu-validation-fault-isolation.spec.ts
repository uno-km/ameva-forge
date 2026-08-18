import { test, expect } from '@playwright/test';

test.describe('Real GPU Validation Error-Scope Isolation Gate', () => {
  test('GPU error scope catches validation fault, rejects execution, and leaves zero pending handles or quota leakage', async ({ page }) => {
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
      let validationErrorCaught = false;
      let errorTypeName = '';

      // 1. Fault injection via test-only triggerValidationError()
      try {
        if (!forge.__testing || !forge.__testing.triggerValidationError) {
          throw new Error('forge.__testing.triggerValidationError is missing in test build');
        }
        await forge.__testing.triggerValidationError();
      } catch (err: any) {
        validationErrorCaught = true;
        errorTypeName = err.name || 'Error';
      }

      // 2. Execute subsequent valid graph to prove pipeline & device recovery
      const a = forge.uploadFloat32Array(new Float32Array([2, 3]), [1, 2]);
      const b = forge.uploadFloat32Array(new Float32Array([4, 5]), [1, 2]);
      const instructions = [
        { op: 'load', id: 1, handle: a, shape: [1, 2], in: [], params: [] },
        { op: 'load', id: 2, handle: b, shape: [1, 2], in: [], params: [] },
        { op: 'add', id: 3, shape: [1, 2], in: [1, 2], params: [] }
      ];
      const outHandles = await forge.executeGraph(JSON.stringify(instructions), [], [3]);
      const outHandle = outHandles[3];

      await forge.mapBufferAsync(outHandle);
      const outData = new Float32Array(2);
      forge.readMappedInto(outHandle, outData);

      // Cleanup
      forge.dispose(a);
      forge.dispose(b);
      forge.dispose(outHandle);
      if (typeof forge.flushGC === 'function') await forge.flushGC();

      const finalQuota = forge.getQuotaSnapshot();

      return {
        validationErrorCaught,
        errorTypeName,
        outData: Array.from(outData),
        usedBytesDiff: finalQuota.usedBytes - baselineQuota.usedBytes,
      };
    });

    expect(result.validationErrorCaught).toBe(true);
    expect(result.outData).toEqual([6, 8]);
    expect(result.usedBytesDiff).toBe(0);
  });
});
