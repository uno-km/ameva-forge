import { test, expect } from '@playwright/test';

test.describe('E2E Training Cycle (F-041)', () => {
  test('initializes WebGPU and runs a complete forward/backward pass', async ({ page }) => {
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

    // Wait for the script to load
    await page.waitForFunction(() => (window as any).testReady === true);

    const result = await page.evaluate(async () => {
      try {
        const forge = (window as any).forge;
        if (!forge) throw new Error('window.forge is not defined');
        // 1. Initialize
        await forge.initWebGPU();

        // 2. Create inputs
        const aArray = new Float32Array([1, 2, 3, 4]);
        const handleA = forge.uploadFloat32Array(aArray, [2, 2]);

        const bArray = new Float32Array([2, 0, 1, 2]);
        const handleB = forge.uploadFloat32Array(bArray, [2, 2]);

        // 3. Execute graph (matmul)
        const instructions = [
          { op: 'load', id: 1, handle: handleA, shape: [2, 2], in: [], params: [] },
          { op: 'load', id: 2, handle: handleB, shape: [2, 2], in: [], params: [] },
          { op: 'matmul', id: 3, shape: [2, 2], in: [1, 2], params: [2, 2, 2] }
        ];

        const outHandles = await forge.executeGraph(JSON.stringify(instructions), [], [3]);
        const outHandle = outHandles[3];

        // 4. Readback
        await forge.mapBufferAsync(outHandle);
        const outData = new Float32Array(4);
        forge.readMappedInto(outHandle, outData);

        // Expected: [[1,2], [3,4]] * [[2,0], [1,2]] = [[2+2, 0+4], [6+4, 0+8]] = [[4, 4], [10, 8]]

        // 5. Cleanup
        forge.dispose(handleA);
        forge.dispose(handleB);
        forge.dispose(outHandle);

        return {
          success: true,
          data: Array.from(outData)
        };
      } catch (e: any) {
        return { success: false, error: e ? (e.stack || e.message) : String(e) };
      }
    });

    if (!result.success) console.error("Evaluate Error:", result.error);
    expect(result.success).toBe(true);
    expect(result.data).toEqual([4, 4, 10, 8]);
  });
});
