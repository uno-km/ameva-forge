import { test, expect } from '@playwright/test';

test.describe('Controlled Quota OOM Recovery Gate', () => {
  test('controlled quota limit rejection does not corrupt existing tensors or prevent subsequent executions', async ({ page }) => {
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

      // 1. Initial valid tensor
      const baselineQuota = forge.getQuotaSnapshot();
      const initialData = new Float32Array([10, 20, 30, 40]);
      const handleInit = forge.uploadFloat32Array(initialData, [2, 2]);

      // 2. Set strict controlled quota limit via test-only API
      if (!forge.__testing || !forge.__testing.setQuotaLimit) {
        throw new Error('forge.__testing.setQuotaLimit is missing in test build');
      }
      forge.__testing.setQuotaLimit(1024); // 1KB limit

      let oomCaught = false;
      let oomErrorType = '';

      // 3. Attempt allocation of 2048 bytes (exceeding 1024 bytes limit)
      try {
        // [16, 32] float32 = 512 elements * 4 bytes = 2048 bytes
        const hugeInstructions = [
          { op: 'upload', id: 1, shape: [16, 32], in: [], params: [] }
        ];
        await forge.executeGraph(JSON.stringify(hugeInstructions), [new Float32Array(512)], [1]);
      } catch (err: any) {
        oomCaught = true;
        oomErrorType = err.name || 'Error';
      }

      // 4. Restore quota limit to normal
      forge.__testing.setQuotaLimit(256 * 1024 * 1024);

      // 5. Verify initial tensor remains completely intact
      await forge.mapBufferAsync(handleInit);
      const readBack = new Float32Array(4);
      forge.readMappedInto(handleInit, readBack);

      // 6. Subsequent small graph execution succeeds
      const instructions = [
        { op: 'load', id: 1, handle: handleInit, shape: [2, 2], in: [], params: [] },
        { op: 'relu', id: 2, shape: [2, 2], in: [1], params: [] }
      ];
      const outHandles = await forge.executeGraph(JSON.stringify(instructions), [], [2]);
      const outHandle = outHandles[2];

      await forge.mapBufferAsync(outHandle);
      const outData = new Float32Array(4);
      forge.readMappedInto(outHandle, outData);

      // Cleanup
      forge.dispose(handleInit);
      forge.dispose(outHandle);
      if (typeof forge.flushGC === 'function') await forge.flushGC();

      const finalQuota = forge.getQuotaSnapshot();

      return {
        oomCaught,
        oomErrorType,
        initialReadBack: Array.from(readBack),
        outData: Array.from(outData),
        usedBytesDiff: finalQuota.usedBytes - baselineQuota.usedBytes,
        baselineQuota,
        finalQuota,
        remainingHandles: forge.snapshotHandles(),
      };
    });

    console.log('QUOTA RESULT:', JSON.stringify(result));
    expect(result.oomCaught).toBe(true);
    expect(result.initialReadBack).toEqual([10, 20, 30, 40]);
    expect(result.outData).toEqual([10, 20, 30, 40]);
    expect(result.usedBytesDiff).toBe(0);
  });
});
