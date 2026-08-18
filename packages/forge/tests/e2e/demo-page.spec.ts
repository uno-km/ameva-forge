import { test, expect } from '@playwright/test';

test('Docs Demo Page - Live WebGPU Studio Pyodide Execution', async ({ page }) => {
  // Navigate to demo.html served from docs via HTTP
  await page.goto('http://127.0.0.1:4173/docs/demo.html');

  // Wait for Pyodide to be ready
  const consoleOutput = page.locator('#consoleOutput');
  await expect(consoleOutput).toContainText('[Ready] Studio initialized', { timeout: 30000 });

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
