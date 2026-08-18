import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 180_000,
  use: {
    baseURL: 'http://127.0.0.1:4173',
    channel: 'chrome',
    headless: true,
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=WebGPU',
        '--use-angle=d3d11',
        '--use-gl=angle',
        '--ignore-gpu-blocklist',
        '--enable-gpu-rasterization',
      ],
    },
  },
  projects: [
    {
      name: 'dev-webgpu',
      testDir: './tests/e2e',
    },
    {
      name: 'release-webgpu',
      testDir: './tests/e2e',
    },
  ],
  webServer: {
    command: 'npx -y http-server ../.. -p 4173 -c-1',
    url: 'http://127.0.0.1:4173/docs/demo.html',
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
