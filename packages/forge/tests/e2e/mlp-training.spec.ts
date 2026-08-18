import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

const reportPath = path.resolve(
  __dirname,
  '../../../../reports/release1/mlp-training-report.json'
);

function writeReport(report: Record<string, unknown>): void {
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
}

test.describe('Release 1 Pyodide WebGPU MLP Gate', () => {
  test('trains a 2-layer MLP with GPU-native SGD', async ({ page }, testInfo) => {
    const consoleErrors: string[] = [];
    page.on('console', message => {
      console.log(`[BROWSER ${message.type()}]`, message.text());
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
        await (window as any).__AMEVA_PYODIDE_READY__;
        const pyodide = (window as any).pyodide;
        if (!pyodide) throw new Error('Pyodide initialization did not complete');

        const script = `
import numpy as np
import forge
import forge.nn as nn
import forge.optim as optim

np.random.seed(1337)

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

async def fingerprint(tensor):
    values = (await tensor.numpy_async()).astype(np.float64).ravel()
    indices = np.arange(1, values.size + 1, dtype=np.float64)
    return {
        "sum": float(values.sum()),
        "weighted_sum": float((values * indices).sum()),
        "l2": float(np.sqrt((values * values).sum())),
    }

async def run_training():
    model = MLP()

    for module in [model.fc1, model.fc2]:
        module.weight = module.weight.to("gpu")
        if module.bias is not None:
            module.bias = module.bias.to("gpu")

    x_tensor = forge.tensor(X_np).to("gpu")
    y_tensor = forge.tensor(y_np).to("gpu")

    params = model.parameters()
    if not params:
        raise RuntimeError("Model parameters are empty")
    if any(p.device != "gpu" for p in params):
        raise RuntimeError("Every model parameter must be on GPU")

    criterion = nn.MSELoss()
    optimizer = optim.SGD(params, lr=0.1, momentum=0.0)

    for p in params:
        await p.realize()

    handles_before = [p._handle for p in params]
    fingerprints_before = [await fingerprint(p) for p in params]

    losses = []
    gradient_devices = []

    for step in range(50):
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = criterion(output, y_tensor)

        loss_value = float((await loss.numpy_async()).mean())
        if not np.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss_value}")
        losses.append(loss_value)

        loss.backward()

        current_grad_devices = [
            p.grad.device if p.grad is not None else None for p in params
        ]
        if any(device != "gpu" for device in current_grad_devices):
            raise RuntimeError(
                f"All gradients must be GPU tensors: {current_grad_devices}"
            )
        gradient_devices = current_grad_devices

        await optimizer.step_async()

    fingerprints_after = [await fingerprint(p) for p in params]
    handles_after = [p._handle for p in params]

    next_output = model(x_tensor)
    next_output_values = await next_output.numpy_async()

    return {
        "input_device": x_tensor.device,
        "target_device": y_tensor.device,
        "parameter_devices": [p.device for p in params],
        "gradient_devices": gradient_devices,
        "handles_before": handles_before,
        "handles_after": handles_after,
        "fingerprints_before": fingerprints_before,
        "fingerprints_after": fingerprints_after,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "next_output_sum": float(next_output_values.astype(np.float64).sum()),
    }

result = await run_training()
result
`;

        const proxy = await pyodide.runPythonAsync(script);
        try {
          return proxy.toJs({ dict_converter: Object.fromEntries });
        } finally {
          proxy.destroy();
        }
      });

      if (!result) throw new Error('MLP training script returned null');
      const res = result;

      console.log('FINGERPRINTS BEFORE:', JSON.stringify(res.fingerprints_before));
      console.log('FINGERPRINTS AFTER:', JSON.stringify(res.fingerprints_after));
      console.log('INITIAL LOSS:', res.initial_loss, 'FINAL LOSS:', res.final_loss);

      const fingerprintsChanged = res.fingerprints_before.some(
        (before: { sum: number; weighted_sum: number; l2: number }, index: number) => {
          const after = res.fingerprints_after[index];
          return (
            Math.abs(before.sum - after.sum) > 1e-7 ||
            Math.abs(before.weighted_sum - after.weighted_sum) > 1e-7 ||
            Math.abs(before.l2 - after.l2) > 1e-7
          );
        }
      );

      // Perform all assertions BEFORE writing the PASS report
      expect(result.input_device).toBe('gpu');
      expect(result.target_device).toBe('gpu');
      expect(result.parameter_devices.every((d: string) => d === 'gpu')).toBe(true);
      expect(result.gradient_devices.every((d: string) => d === 'gpu')).toBe(true);
      expect(result.handles_before.every((h: string | null) => Boolean(h))).toBe(true);
      expect(result.handles_after).toEqual(result.handles_before);
      expect(fingerprintsChanged).toBe(true);
      expect(Number.isFinite(result.initial_loss)).toBe(true);
      expect(Number.isFinite(result.final_loss)).toBe(true);
      expect(result.final_loss).toBeLessThan(result.initial_loss);
      expect(result.losses).toHaveLength(50);
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
        fingerprints_changed: fingerprintsChanged,
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
});
