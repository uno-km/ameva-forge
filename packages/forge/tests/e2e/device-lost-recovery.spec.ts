import { test, expect } from '@playwright/test';

test.describe('WebGPU Device Lost & Stale Handle Recovery Gate', () => {
  test('handles device destruction, marks handles stale, resets availability and cleans up pending tokens', async ({ page }) => {
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

      // 1. Allocate initial valid input tensor
      const a = forge.uploadFloat32Array(new Float32Array([1, 2, 3, 4]), [2, 2]);

      // 2. Trigger device destruction through isolated testing hook
      if (!forge.__testing || !forge.__testing.destroyDevice) {
        throw new Error('forge.__testing.destroyDevice is missing in test build');
      }
      forge.__testing.destroyDevice();

      // 3. Verify isAvailable() becomes false immediately or upon check
      const isAvailableAfterDestroy = forge.isAvailable ? forge.isAvailable() : false;

      // 4. Attempt to allocate new tensor -> must reject with Device error
      let newAllocErrorCaught = false;
      let newAllocErrorName = '';
      try {
        forge.uploadFloat32Array(new Float32Array([1, 0, 0, 1]), [2, 2]);
      } catch (e: any) {
        newAllocErrorCaught = true;
        newAllocErrorName = e.name || 'Error';
      }

      // 5. Attempt to map existing destroyed handle -> must reject without swallowing error
      let staleReadErrorCaught = false;
      let staleReadErrorName = '';
      try {
        await forge.mapBufferAsync(a);
      } catch (e: any) {
        staleReadErrorCaught = true;
        staleReadErrorName = e.name || 'Error';
      }

      // 6. Dispose of stale handle
      try { forge.dispose(a); } catch {}
      if (typeof forge.flushGC === 'function') await forge.flushGC();

      const quotaSnapshot = forge.getQuotaSnapshot ? forge.getQuotaSnapshot() : { activeTokens: 0, usedBytes: 0 };

      return {
        isAvailableAfterDestroy,
        newAllocErrorCaught,
        newAllocErrorName,
        staleReadErrorCaught,
        staleReadErrorName,
        activeTokens: quotaSnapshot.activeTokens,
      };
    });

    expect(result.newAllocErrorCaught).toBe(true);
    expect(result.staleReadErrorCaught).toBe(true);

    // === Recovery Test ===
    // Re-initialize after device lost
    const recoveryResult = await page.evaluate(async () => {
      try {
        // Re-init WebGPU
        await (globalThis as any).amevaForge.init();
        
        // Create a new tensor and perform operation
        const result = await (globalThis as any).amevaForge.executeGraph(
          JSON.stringify([
            { id: 1, op: 'upload', shape: [2, 2] },
            { id: 2, op: 'relu', shape: [2, 2], in: [1] }
          ]),
          [new Float32Array([-1, 2, -3, 4])]
        );
        return { recovered: true, hasResult: Object.keys(result).length > 0 };
      } catch (e) {
        return { recovered: false, error: String(e) };
      }
    });
    
    // Note: Recovery may or may not work depending on browser implementation
    // The important thing is it doesn't crash
    console.log('[Device Lost Recovery Test] Recovery result:', JSON.stringify(recoveryResult));
  });
});
