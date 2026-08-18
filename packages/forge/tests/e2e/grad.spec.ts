import { test, expect } from '@playwright/test';

test.describe('Numerical Gradient Validation (F-040)', () => {
  test('validates ReLU backward using finite difference', async ({ page }) => {
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

    await page.waitForFunction(() => (window as any).testReady === true);

    const result = await page.evaluate(async () => {
      try {
        const forge = (window as any).forge;
        if (!forge) throw new Error('window.forge is not defined');
        await forge.initWebGPU();

        // 1. Data Setup
        const xArray = new Float32Array([-1.5, -0.5, 0.0, 0.5, 1.5]);
        const gradArray = new Float32Array([1, 1, 1, 1, 1]); // incoming gradient

        const hX = forge.uploadFloat32Array(xArray, [5]);
        const hGrad = forge.uploadFloat32Array(gradArray, [5]);

        // 2. Analytic Gradient via GPU
        const instructions = [
          { op: 'load', id: 1, handle: hX, shape: [5], in: [], params: [] },
          { op: 'load', id: 2, handle: hGrad, shape: [5], in: [], params: [] },
          { op: 'relu_backward', id: 3, shape: [5], in: [1, 2], params: [] }
        ];

        const outHandles = await forge.executeGraph(JSON.stringify(instructions), [], [3]);
        await forge.mapBufferAsync(outHandles[3]);
        const analyticGrad = new Float32Array(5);
        forge.readMappedInto(outHandles[3], analyticGrad);

        // 3. Numerical Gradient via CPU (Finite Difference)
        const EPSILON = 1e-4;
        const numericalGrad = new Float32Array(5);
        for (let i = 0; i < 5; i++) {
          const v = xArray[i];
          const f_plus = Math.max(0, v + EPSILON);
          const f_minus = Math.max(0, v - EPSILON);
          numericalGrad[i] = (f_plus - f_minus) / (2 * EPSILON);
        }

        // Cleanup
        forge.dispose(hX);
        forge.dispose(hGrad);
        forge.dispose(outHandles[3]);

        // 4. Compare
        const analytic = Array.from(analyticGrad);
        const numerical = Array.from(numericalGrad);

        let maxError = 0;
        for (let i = 0; i < 5; i++) {
          // Skip x=0 because ReLU is non-differentiable there and numerical grad will be 0.5
          if (xArray[i] === 0.0) continue;

          const err = Math.abs(analytic[i] - numerical[i]);
          if (err > maxError) maxError = err;
        }

        return {
          success: true,
          analytic,
          numerical,
          maxError
        };
      } catch (e: any) {
        return { success: false, error: e ? (e.stack || e.message) : String(e) };
      }
    });

    if (!result.success) console.error("Evaluate Error:", result.error);
    expect(result.success).toBe(true);
    expect(result.maxError).toBeLessThan(1e-4);
  });
});
