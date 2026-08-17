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
/**
 * WHAT: 시스템 환경에서 WebGPU 디바이스 및 어댑터를 비동기적으로 초기화합니다.
 * WHY: WebGPU API를 사용하기 위해 필수적인 하드웨어 어댑터(adapter)와 논리적 디바이스(device) 인스턴스를 확보하고 전역에서 접근할 수 있도록 캐싱하기 위해 존재합니다.
 * HOW:
 *   1. navigator.gpu 객체가 존재하는지 확인하고, requestAdapter()로 물리적 GPU 어댑터를 요청합니다.
 *   2. 어댑터가 지원하는 최대 버퍼 크기 등의 한계를 파악하여 requestDevice()로 디바이스를 생성합니다.
 *   3. 디바이스 손실(device.lost) 이벤트를 수신하여 리소스를 정리하고 등록된 콜백을 실행하도록 설정합니다.
 */
export declare function initWebGPU(options?: GPURequestAdapterOptions): Promise<void>;
/**
 * WHAT: 전역에 캐시된 WebGPU 디바이스 인스턴스를 반환합니다.
 * WHY: 애플리케이션의 여러 모듈에서 동일한 단일 디바이스 인스턴스에 접근하여 버퍼 및 텍스처를 생성할 수 있도록 제공하기 위함입니다.
 * HOW: 내부 `device` 변수가 초기화되어 있는지 확인하고, 없을 경우 예외(AMEVAForgeDeviceError)를 발생시키며, 존재할 경우 그대로 반환합니다.
 */
export declare function getDevice(): GPUDevice;
/**
 * WHAT: 전역에 캐시된 WebGPU 어댑터(Adapter) 인스턴스를 반환합니다.
 * WHY: GPU의 하드웨어 스펙(limits, features 등)을 조회하거나 디바이스 기능 제약 조건을 파악하기 위해 외부 모듈에서 어댑터에 접근할 수 있게 합니다.
 * HOW: 내부 `adapter` 변수를 그대로 반환합니다. 아직 초기화되지 않았다면 null이 반환될 수 있습니다.
 */
export declare function getAdapter(): GPUAdapter | null;
/**
 * WHAT: 초기화된 WebGPU 디바이스와 연결된 커맨드 큐(GPUQueue)를 반환합니다.
 * WHY: 데이터를 버퍼로 전송(writeBuffer)하거나 렌더링/컴퓨트 커맨드(submit)를 실행할 수 있도록 접근 지점을 제공합니다.
 * HOW: `getDevice()` 함수를 호출해 디바이스를 얻은 후 `device.queue` 속성을 반환합니다.
 */
export declare function getQueue(): GPUQueue;
/**
 * WHAT: WebGPU 디바이스가 현재 성공적으로 초기화되어 사용 가능한지 여부를 반환합니다.
 * WHY: 기능 호환성 검사나 런타임 조건부 로직 실행 전, WebGPU 사용 가능 여부를 안전하게 확인하기 위해 제공됩니다.
 * HOW: 내부에 저장된 `device` 변수가 null이 아닌지 불리언(Boolean) 값으로 평가하여 반환합니다.
 */
export declare function isAvailable(): boolean;
export declare function _resetDeviceForTesting(): void;
/**
 * WHAT: GPU 디바이스 연결이 끊어졌을 때(device lost) 호출될 콜백 함수를 등록합니다.
 * WHY: 예기치 못한 GPU 충돌이나 컨텍스트 상실 시 상위 계층(예: 파이프라인 캐시 무효화, 재초기화 로직)에 이를 알리기 위해 존재합니다.
 * HOW: 전달받은 함수(callback)를 모듈 레벨 변수인 `onDeviceLostCallback`에 할당하여 이후 디바이스 손실 이벤트 발생 시 실행될 수 있도록 합니다.
 */
export declare function setDeviceLostCallback(callback: () => void): void;
