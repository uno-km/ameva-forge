import { test, expect } from '@playwright/test';
import * as path from 'path';

test.describe('Numerical Gradient Validation (F-040)', () => {
  test('validates ReLU backward using finite difference', async ({ page }) => {
    const htmlPath = 'file:///' + path.resolve(__dirname, '../../test.html').replace(/\\/g, '/');
    await page.goto(htmlPath);

    await page.waitForFunction(() => (window as any).testReady === true);

    const result = await page.evaluate(async () => {
      try {
        const forge = (window as any).forge;
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
        return { success: false, error: e.message };
      }
    });

    if (!result.success) console.error("Evaluate Error:", result.error);
    expect(result.success).toBe(true);
    expect(result.maxError).toBeLessThan(1e-4);
  });
});
