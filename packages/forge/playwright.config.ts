import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30 * 1000,
  expect: {
    timeout: 5000
  },
  use: {
    browserName: 'chromium',
    headless: false,
    // Enable WebGPU in headless chrome
    launchOptions: {
      args: ['--enable-unsafe-webgpu']
    }
  },
});
