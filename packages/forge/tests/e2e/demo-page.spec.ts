import { test, expect } from '@playwright/test';

test('Docs Demo Page - Live WebGPU Studio Pyodide Execution', async ({ page }) => {
  page.on('console', msg => console.log('[PAGE CONSOLE]', msg.text()));
  page.on('pageerror', err => console.error('[PAGE ERROR]', err));
  // Navigate to demo.html served from docs via HTTP
  await page.goto('http://127.0.0.1:4173/docs/demo.html');

  // Wait for Pyodide to be ready
  const consoleOutput = page.locator('#consoleOutput');
  await expect(consoleOutput).toContainText('[Ready] Studio initialized', { timeout: 30000 });

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
    test.skip(!gpuAvailable, 'WebGPU adapter is unavailable in this test environment');
    return;
  }

  // Click Run Experiment
  const btnRun = page.locator('#btnRun');
  await btnRun.click();

  // Wait for training completion
  await expect(consoleOutput).toContainText('Training Complete! Final Loss:', { timeout: 30000 });

  // Check Metric Loss is updated with decreased loss
  const metricLoss = page.locator('#metricLoss');
  const lossText = await metricLoss.textContent();
  expect(parseFloat(lossText || '1.0')).toBeLessThan(0.3);
});
