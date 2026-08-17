/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * device.ts — WebGPU 싱글톤 디바이스 래퍼
 *
 * H-04 Fix: getAdapter() export 추가 → gpuCore.ts에서 adapter.limits 조회 가능
 * L-03 Fix: device lost 시 onDeviceLostCallback을 통해 pipelineCache도 무효화
 */

import { AMEVAForgeWebGPUUnavailableError, AMEVAForgeDeviceError } from "../errors";
import { _globalQuotaManager } from "./quota";

declare var process: any;

/**
 * WHAT: 개발 환경이나 디버그 모드에서만 시스템 메시지를 출력하는 안전한 로깅 함수입니다.
 * WHY: 불필요한 콘솔 출력을 프로덕션 환경에서 방지하고, 에러 없이 안전하게 로그를 남기기 위해 사용됩니다.
 * HOW: 현재 실행 환경이 개발 모드(NODE_ENV, AMEVA_DEBUG, __DEV__, Vite env 등)인지 확인하고 조건을 만족할 때만 `globalThis.log`를 통해 메시지를 출력합니다. 예외가 발생해도 시스템이 멈추지 않도록 try-catch로 감쌉니다.
 */
function _safeLog(msg: string) {
  try {
    // VUL-015 & L-03 Fix: Only log in development or explicit debug modes without CSP violations
    const isDev = 
      (typeof process !== 'undefined' && process.env && process.env.NODE_ENV !== 'production') ||
      (typeof (globalThis as any).AMEVA_DEBUG !== 'undefined' && (globalThis as any).AMEVA_DEBUG) ||
      (typeof (globalThis as any).__DEV__ !== 'undefined' && (globalThis as any).__DEV__);

    if (!isDev) return;

    if (typeof (globalThis as any).log === 'function') {
      (globalThis as any).log(msg, 'system');
    }
  } catch (e) {}
}

/**
 * WHAT: 초기화된 논리적 WebGPU 디바이스(GPUDevice) 인스턴스를 저장하는 내부 변수입니다.
 * WHY: 모듈 내에서 싱글톤(singleton) 패턴을 유지하여 여러 번 초기화되지 않도록 상태를 관리합니다.
 * HOW: initWebGPU 함수 내에서 생성된 디바이스가 할당되며, 디바이스 손실(device lost) 시 다시 null로 초기화됩니다.
 */
let device: GPUDevice | null = null;
/**
 * WHAT: WebGPU 기능 및 하드웨어 한계를 나타내는 물리적 GPU 어댑터(GPUAdapter) 인스턴스를 저장하는 내부 변수입니다.
 * WHY: 디바이스 생성 전 스펙(limits)을 검사하거나 생성 후 하드웨어 정보를 참조하기 위해 캐싱해 둡니다.
 * HOW: initWebGPU 실행 시 requestAdapter()로 요청받아 할당되며, 디바이스 손실 시 함께 정리됩니다.
 */
let adapter: GPUAdapter | null = null;
/**
 * WHAT: 디바이스 손실(device lost) 발생 시 실행될 콜백 함수를 저장하는 변수입니다.
 * WHY: GPU 오류나 컨텍스트 초기화 상황이 발생했을 때 상위 애플리케이션으로 이벤트를 위임하기 위해 필요합니다.
 * HOW: setDeviceLostCallback 함수를 통해 설정되며, device.lost Promise가 해결(resolve)될 때 내부적으로 호출됩니다.
 */
let onDeviceLostCallback: (() => void) | null = null;

/**
 * WHAT: 시스템 환경에서 WebGPU 디바이스 및 어댑터를 비동기적으로 초기화합니다.
 * WHY: WebGPU API를 사용하기 위해 필수적인 하드웨어 어댑터(adapter)와 논리적 디바이스(device) 인스턴스를 확보하고 전역에서 접근할 수 있도록 캐싱하기 위해 존재합니다.
 * HOW: 
 *   1. navigator.gpu 객체가 존재하는지 확인하고, requestAdapter()로 물리적 GPU 어댑터를 요청합니다.
 *   2. 어댑터가 지원하는 최대 버퍼 크기 등의 한계를 파악하여 requestDevice()로 디바이스를 생성합니다.
 *   3. 디바이스 손실(device.lost) 이벤트를 수신하여 리소스를 정리하고 등록된 콜백을 실행하도록 설정합니다.
 */
export async function initWebGPU(options?: GPURequestAdapterOptions): Promise<void> {
  _safeLog(`[device.ts] initWebGPU started. current device=${device ? 'SET' : 'NULL'}`);
  if (device) return;

  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new AMEVAForgeWebGPUUnavailableError(
      "WebGPU is not available in this environment. " +
      "Ensure you are running in a supported browser with WebGPU enabled."
    );
  }

  adapter = await navigator.gpu.requestAdapter(options);
  if (!adapter) {
    throw new AMEVAForgeWebGPUUnavailableError(
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

  if (device.limits && device.limits.maxStorageBufferBindingSize) {
    const maxBinding = device.limits.maxStorageBufferBindingSize;
    const adaptedHard = Math.max(1024 * 1024 * 1024, maxBinding * 2);
    const adaptedSoft = Math.max(768 * 1024 * 1024, maxBinding);
    try {
      _globalQuotaManager.setLimits(adaptedHard, adaptedSoft);
    } catch {}
  }

  _safeLog(`[device.ts] initWebGPU finished. device successfully created.`);

  device.lost.then((info) => {
    const msg = `[AMEVA] WebGPU Device Lost: ${info.message} (reason: ${info.reason})`;
    console.error(msg);
    _safeLog(msg);
    device = null;
    // (globalThis as any).__AMEVA_DEVICE__ = null;
    adapter = null;
    if (onDeviceLostCallback) {
      onDeviceLostCallback();
    }
  });
}

/**
 * WHAT: 전역에 캐시된 WebGPU 디바이스 인스턴스를 반환합니다.
 * WHY: 애플리케이션의 여러 모듈에서 동일한 단일 디바이스 인스턴스에 접근하여 버퍼 및 텍스처를 생성할 수 있도록 제공하기 위함입니다.
 * HOW: 내부 `device` 변수가 초기화되어 있는지 확인하고, 없을 경우 예외(AMEVAForgeDeviceError)를 발생시키며, 존재할 경우 그대로 반환합니다.
 */
export function getDevice(): GPUDevice {
  if (!device) {
    const globalExists = typeof globalThis.amevaForge !== "undefined";
    throw new AMEVAForgeDeviceError(
      `WebGPU device is not initialized. (globalThis.amevaForge exists: ${globalExists}). Call await init() first.`
    );
  }
  return device;
}

/**
 * WHAT: 전역에 캐시된 WebGPU 어댑터(Adapter) 인스턴스를 반환합니다.
 * WHY: GPU의 하드웨어 스펙(limits, features 등)을 조회하거나 디바이스 기능 제약 조건을 파악하기 위해 외부 모듈에서 어댑터에 접근할 수 있게 합니다.
 * HOW: 내부 `adapter` 변수를 그대로 반환합니다. 아직 초기화되지 않았다면 null이 반환될 수 있습니다.
 */
export function getAdapter(): GPUAdapter | null {
  return adapter;
}

/**
 * WHAT: 초기화된 WebGPU 디바이스와 연결된 커맨드 큐(GPUQueue)를 반환합니다.
 * WHY: 데이터를 버퍼로 전송(writeBuffer)하거나 렌더링/컴퓨트 커맨드(submit)를 실행할 수 있도록 접근 지점을 제공합니다.
 * HOW: `getDevice()` 함수를 호출해 디바이스를 얻은 후 `device.queue` 속성을 반환합니다.
 */
export function getQueue(): GPUQueue {
  return getDevice().queue;
}

/**
 * WHAT: WebGPU 디바이스가 현재 성공적으로 초기화되어 사용 가능한지 여부를 반환합니다.
 * WHY: 기능 호환성 검사나 런타임 조건부 로직 실행 전, WebGPU 사용 가능 여부를 안전하게 확인하기 위해 제공됩니다.
 * HOW: 내부에 저장된 `device` 변수가 null이 아닌지 불리언(Boolean) 값으로 평가하여 반환합니다.
 */
export function isAvailable(): boolean {
  return device !== null;
}

export function _resetDeviceForTesting(): void {
  device = null;
  adapter = null;
  if (onDeviceLostCallback) {
    onDeviceLostCallback();
  }
}

/**
 * WHAT: GPU 디바이스 연결이 끊어졌을 때(device lost) 호출될 콜백 함수를 등록합니다.
 * WHY: 예기치 못한 GPU 충돌이나 컨텍스트 상실 시 상위 계층(예: 파이프라인 캐시 무효화, 재초기화 로직)에 이를 알리기 위해 존재합니다.
 * HOW: 전달받은 함수(callback)를 모듈 레벨 변수인 `onDeviceLostCallback`에 할당하여 이후 디바이스 손실 이벤트 발생 시 실행될 수 있도록 합니다.
 */
export function setDeviceLostCallback(callback: () => void): void {
  onDeviceLostCallback = callback;
}
