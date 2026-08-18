import { test, expect } from '@playwright/test';

test.describe('GPU Edge Cases', () => {
  test('1x1 matrix multiplication', async ({ page }) => {
    await page.goto('/test.html');

    const gpuAvailable = await page.evaluate(async () => {
      if (!navigator.gpu) return false;
      const adapter = await navigator.gpu.requestAdapter();
      return !!adapter;
    });
    if (!gpuAvailable) {
      if (process.env.REQUIRE_WEBGPU === '1') {
        throw new Error('REQUIRE_WEBGPU=1 but WebGPU adapter is unavailable');
      }
      test.skip(!gpuAvailable, 'WebGPU adapter is unavailable');
    }

    const result = await page.evaluate(async () => {
      await (globalThis as any).amevaForge.init();
      const r = await (globalThis as any).amevaForge.executeGraph(
        JSON.stringify([
          { id: 1, op: 'upload', shape: [1, 1] },
          { id: 2, op: 'upload', shape: [1, 1] },
          { id: 3, op: 'matmul', shape: [1, 1], in: [1, 2], params: [1, 1, 1] }
        ]),
        [new Float32Array([3.0]), new Float32Array([4.0])]
      );
      const handle = r[3] || r['3'];
      const data = await (globalThis as any).amevaForge.read(handle);
      return Array.from(data);
    });
    expect(result[0]).toBeCloseTo(12.0, 3);
  });

  test('scalar tensor operations', async ({ page }) => {
    await page.goto('/test.html');

    const gpuAvailable = await page.evaluate(async () => {
      if (!navigator.gpu) return false;
      const adapter = await navigator.gpu.requestAdapter();
      return !!adapter;
    });
    if (!gpuAvailable) {
      if (process.env.REQUIRE_WEBGPU === '1') {
        throw new Error('REQUIRE_WEBGPU=1 but WebGPU adapter is unavailable');
      }
      test.skip(!gpuAvailable, 'WebGPU adapter is unavailable');
    }

    const result = await page.evaluate(async () => {
      await (globalThis as any).amevaForge.init();
      const r = await (globalThis as any).amevaForge.executeGraph(
        JSON.stringify([
          { id: 1, op: 'upload', shape: [1] },
          { id: 2, op: 'relu', shape: [1], in: [1] }
        ]),
        [new Float32Array([-5.0])]
      );
      const handle = r[2] || r['2'];
      const data = await (globalThis as any).amevaForge.read(handle);
      return Array.from(data);
    });
    expect(result[0]).toBe(0.0);
  });

  test('large dispatch (65536+ elements)', async ({ page }) => {
    await page.goto('/test.html');

    const gpuAvailable = await page.evaluate(async () => {
      if (!navigator.gpu) return false;
      const adapter = await navigator.gpu.requestAdapter();
      return !!adapter;
    });
    if (!gpuAvailable) {
      if (process.env.REQUIRE_WEBGPU === '1') {
        throw new Error('REQUIRE_WEBGPU=1 but WebGPU adapter is unavailable');
      }
      test.skip(!gpuAvailable, 'WebGPU adapter is unavailable');
    }

    const result = await page.evaluate(async () => {
      await (globalThis as any).amevaForge.init();
      const size = 70000; // > 65535 to test 2D dispatch
      const data = new Float32Array(size).fill(-1.0);
      const r = await (globalThis as any).amevaForge.executeGraph(
        JSON.stringify([
          { id: 1, op: 'upload', shape: [size] },
          { id: 2, op: 'relu', shape: [size], in: [1] }
        ]),
        [data]
      );
      const handle = r[2] || r['2'];
      const result = await (globalThis as any).amevaForge.read(handle);
      // All -1.0 should become 0.0 after ReLU
      return { allZero: result.every((v: number) => v === 0.0), length: result.length };
    });
    expect(result.allZero).toBe(true);
    expect(result.length).toBe(70000);
  });
});
