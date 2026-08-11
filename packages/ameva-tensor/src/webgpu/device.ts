/**
 * device.ts — WebGPU 싱글톤 디바이스 래퍼
 *
 * H-04 Fix: getAdapter() export 추가 → gpuCore.ts에서 adapter.limits 조회 가능
 * L-03 Fix: device lost 시 onDeviceLostCallback을 통해 pipelineCache도 무효화
 */

import { AMEVATensorWebGPUUnavailableError, AMEVATensorDeviceError } from "../errors";

function _safeLog(msg: string) {
  try {
    if (typeof (globalThis as any).log === 'function') {
      (globalThis as any).log(msg, 'system');
    }
  } catch (e) {}
}

let device: GPUDevice | null = null;
let adapter: GPUAdapter | null = null;
let onDeviceLostCallback: (() => void) | null = null;

export async function initWebGPU(options?: GPURequestAdapterOptions): Promise<void> {
  _safeLog(`[device.ts] initWebGPU started. current device=${device ? 'SET' : 'NULL'}`);
  if (device) return;

  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new AMEVATensorWebGPUUnavailableError(
      "WebGPU is not available in this environment. " +
      "Ensure you are running in a supported browser with WebGPU enabled."
    );
  }

  adapter = await navigator.gpu.requestAdapter(options);
  if (!adapter) {
    throw new AMEVATensorWebGPUUnavailableError(
      "Failed to request a WebGPU adapter. " +
      "Your GPU may not support WebGPU, or the browser has disabled it."
    );
  }

  const requiredLimits: any = {};
  if (adapter.limits) {
    requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
    requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
  }
  
  device = await adapter.requestDevice({ requiredLimits });
  (globalThis as any).__AMEVA_DEVICE__ = device;

  _safeLog(`[device.ts] initWebGPU finished. device successfully created.`);

  device.lost.then((info) => {
    const msg = `[AMEVA] WebGPU Device Lost: ${info.message} (reason: ${info.reason})`;
    console.error(msg);
    _safeLog(msg);
    device = null;
    (globalThis as any).__AMEVA_DEVICE__ = null;
    adapter = null;
    if (onDeviceLostCallback) {
      onDeviceLostCallback();
    }
  });
}

export function getDevice(): GPUDevice {
  const globalDev = (globalThis as any).__AMEVA_DEVICE__;
  _safeLog(`[device.ts] getDevice called. local device=${device ? 'SET' : 'NULL'}, globalDev=${globalDev ? 'SET' : 'NULL'}`);
  
  if (!device) {
    const globalExists = typeof globalThis.amevaTensor !== "undefined";
    throw new AMEVATensorDeviceError(
      `WebGPU device is not initialized. (device is ${device}, __AMEVA_DEVICE__ exists: ${!!globalDev}, globalThis.amevaTensor exists: ${globalExists}). Call await init() first.`
    );
  }
  return device;
}

/** H-04: adapter.limits 접근용 */
export function getAdapter(): GPUAdapter | null {
  return adapter;
}

export function getQueue(): GPUQueue {
  return getDevice().queue;
}

export function isAvailable(): boolean {
  return device !== null;
}

export function setDeviceLostCallback(callback: () => void): void {
  onDeviceLostCallback = callback;
}
