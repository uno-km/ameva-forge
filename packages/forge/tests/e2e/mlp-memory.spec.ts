import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

const reportPath = path.resolve(
  __dirname,
  '../../../../reports/release1/mlp-memory-report.json'
);

function writeReport(report: Record<string, unknown>): void {
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
}

test('1,000 MLP training steps return quota to post-model baseline', async ({ page }, testInfo) => {
  const consoleErrors: string[] = [];
  page.on('console', message => {
    if (message.type() === 'error' && !message.text().includes('404') && !message.text().includes('favicon')) {
      consoleErrors.push(message.text());
    }
  });

  const startedAt = new Date().toISOString();
  const runId = process.env.AMEVA_RUN_ID || crypto.randomUUID();

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
  const releaseGate = testInfo.project.name === 'release-webgpu';
  if (!gpuAvailable && releaseGate) {
    throw new Error('RELEASE_GATE_WEBGPU_UNAVAILABLE: A physical WebGPU adapter is required.');
  }
  if (!gpuAvailable) {
    if (process.env.REQUIRE_WEBGPU === '1') {
      throw new Error('REQUIRE_WEBGPU=1 but WebGPU adapter is unavailable');
    }
    test.skip(!gpuAvailable, 'WebGPU adapter is unavailable outside release-webgpu project');
  }

  let result: Record<string, any> | null = null;
  let environment: Record<string, any> = {};

  try {
    environment = await page.evaluate(async () => {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('WebGPU adapter request failed');
      const info = 'info' in adapter ? (adapter as any).info : null;
      return {
        userAgent: navigator.userAgent,
        adapterInfo: info ? {
          vendor: info.vendor ?? null,
          architecture: info.architecture ?? null,
          device: info.device ?? null,
          description: info.description ?? null,
        } : {},
      };
    });

    result = await page.evaluate(async () => {
      const forge = (window as any).forge;
      if (!forge || typeof forge.flushGC !== 'function') {
        throw new Error('flushGC API is required for the memory acceptance gate');
      }
      if (!forge.getQuotaSnapshot) {
        throw new Error('getQuotaSnapshot API is required for memory tracking');
      }

      await (window as any).__AMEVA_PYODIDE_READY__;
      const pyodide = (window as any).pyodide;
      if (!pyodide) throw new Error('Pyodide initialization did not complete');

      const script = `
import numpy as np
import forge
import forge.nn as nn
import forge.optim as optim
from js import forge as js_forge

X_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_np = np.array([[0], [1], [1], [0]], dtype=np.float32)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def get_snapshot():
    snap = js_forge.getQuotaSnapshot()
    return {
        "used_bytes": int(snap.usedBytes),
        "max_bytes": int(snap.maxBytes),
        "active_tokens": int(snap.activeTokens),
    }

async def run_memory_gate():
    model = MLP()

    for module in [model.fc1, model.fc2]:
        module.weight = module.weight.to("gpu")
        if module.bias is not None:
            module.bias = module.bias.to("gpu")

    x_tensor = forge.tensor(X_np).to("gpu")
    y_tensor = forge.tensor(y_np).to("gpu")

    params = model.parameters()
    for p in params:
        await p.realize()
    await x_tensor.realize()
    await y_tensor.realize()

    import gc
    gc.collect()
    forge.flush_gc()
    await js_forge.flushGC()
    post_model_baseline = get_snapshot()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(params, lr=0.01, momentum=0.0)

    samples = []
    peak_bytes = post_model_baseline["used_bytes"]

    for step in range(1, 1001):
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        await optimizer.step_async()

        # Clean intermediate graph references
        output = None
        loss = None
        gc.collect()
        forge.flush_gc()
        await js_forge.flushGC()

        if step % 25 == 0:
            current = get_snapshot()
            peak_bytes = max(peak_bytes, current["used_bytes"])
            samples.append({
                "step": step,
                "used_bytes": current["used_bytes"],
                "active_tokens": current["active_tokens"],
            })

    gc.collect()
    forge.flush_gc()
    await js_forge.flushGC()
    final_snapshot = get_snapshot()

    print("REMAINING TENSORS:", [
        (h, js_forge.getTensorInfo(h).byteLength, list(js_forge.getTensorInfo(h).shape))
        for h in js_forge.snapshotHandles()
    ])

    return {
        "post_model_baseline": post_model_baseline,
        "final_snapshot": final_snapshot,
        "peak_bytes": peak_bytes,
        "samples": samples,
        "completed_steps": 1000,
        "remaining_handles": [
            [h, int(js_forge.getTensorInfo(h).byteLength), list(js_forge.getTensorInfo(h).shape)]
            for h in js_forge.snapshotHandles()
        ],
    }

result = await run_memory_gate()
result
`;

      const proxy = await pyodide.runPythonAsync(script);
      try {
        return proxy.toJs({ dict_converter: Object.fromEntries });
      } finally {
        proxy.destroy();
      }
    });

    if (!result) throw new Error('MLP memory gate script returned null');

    console.log('POST MODEL BASELINE:', JSON.stringify(result.post_model_baseline));
    console.log('FINAL SNAPSHOT:', JSON.stringify(result.final_snapshot));
    console.log('SAMPLE 0:', JSON.stringify(result.samples[0]));
    console.log('SAMPLE LAST:', JSON.stringify(result.samples[result.samples.length - 1]));
    console.log('REMAINING HANDLES:', JSON.stringify(result.remaining_handles));

    expect(result.completed_steps).toBe(1000);
    expect(result.final_snapshot.used_bytes).toBe(result.post_model_baseline.used_bytes);
    expect(result.final_snapshot.active_tokens).toBe(result.post_model_baseline.active_tokens);
    expect(consoleErrors).toEqual([]);

    writeReport({
      schema_version: 1,
      run_id: runId,
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      git_commit: process.env.AMEVA_GIT_COMMIT || 'UNKNOWN',
      working_tree: process.env.AMEVA_WORKING_TREE || 'DIRTY',
      browser_name: testInfo.project.name,
      status: 'EXECUTED',
      classification: 'PASS',
      console_errors: consoleErrors,
      ...environment,
      ...result,
    });
  } catch (error) {
    writeReport({
      schema_version: 1,
      run_id: runId,
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      git_commit: process.env.AMEVA_GIT_COMMIT || 'UNKNOWN',
      working_tree: process.env.AMEVA_WORKING_TREE || 'DIRTY',
      browser_name: testInfo.project.name,
      status: 'FAILED',
      classification: 'FAIL',
      console_errors: consoleErrors,
      ...environment,
      result,
      error: error instanceof Error ? error.stack ?? error.message : String(error),
    });
    throw error;
  }
});
