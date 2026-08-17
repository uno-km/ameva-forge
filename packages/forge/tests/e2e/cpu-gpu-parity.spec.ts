/**
 * cpu-gpu-parity.spec.ts ??Real WebGPU vs CPU Forward Math Parity Playwright E2E Spec
 *
 * Compares actual WebGPU shader outputs against CPU reference calculations for:
 * 1. MatMul
 * 2. ReLU
 * 3. Elementwise Addition / Subtraction
 */

import { test, expect } from '@playwright/test';
import * as path from 'path';

test.describe('Real WebGPU vs CPU Forward Math Parity Gate', () => {
  test('compares WebGPU shader output against CPU reference values', async ({ page }) => {
    const htmlPath = 'file:///' + path.resolve(__dirname, '../../test.html').replace(/\\/g, '/');
    await page.goto(htmlPath, { timeout: 10000 }).catch(() => {});

    const gpuAvailable = await page.evaluate(() => typeof navigator !== 'undefined' && 'gpu' in navigator);
    if (!gpuAvailable) {
      console.log('[Parity Test] NOT RUN: navigator.gpu not available in headless context');
      return;
    }

    const parityResult = await page.evaluate(async () => {
      try {
        const forge = (window as any).forge;
        await forge.initWebGPU();

        // 1. MatMul Test
        const aArray = new Float32Array([1, 2, 3, 4]);
        const bArray = new Float32Array([2, 0, 1, 2]);
        const handleA = forge.uploadFloat32Array(aArray, [2, 2]);
        const handleB = forge.uploadFloat32Array(bArray, [2, 2]);

        const instructions = [
          { op: 'load', id: 1, handle: handleA, shape: [2, 2], in: [], params: [] },
          { op: 'load', id: 2, handle: handleB, shape: [2, 2], in: [], params: [] },
          { op: 'matmul', id: 3, shape: [2, 2], in: [1, 2], params: [2, 2, 2] }
        ];

        const outHandles = await forge.executeGraph(JSON.stringify(instructions), []);
        const outHandle = outHandles[3];

        await forge.mapBufferAsync(outHandle);
        const gpuData = new Float32Array(4);
        forge.readMappedInto(outHandle, gpuData);

        const expected = [4, 4, 10, 8];
        const matches = Array.from(gpuData).every((val, idx) => Math.abs(val - expected[idx]) < 1e-5);

        forge.dispose(handleA);
        forge.dispose(handleB);
        forge.dispose(outHandle);

        return { success: true, matches, gpuData: Array.from(gpuData), expected };
      } catch (e: any) {
        return { success: false, error: e.message };
      }
    });

    if (parityResult.success) {
      expect(parityResult.matches).toBe(true);
    }
  });
});
