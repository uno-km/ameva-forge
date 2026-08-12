/**
 * buffers.ts — GPU 버퍼 할당, 읽기 인터페이스
 *
 * C-05 Fix: _stagingBuffers 전역 Map 제거 → mapBufferAsync가 staging buffer 직접 반환.
 * H-05 / NH-05 Fix: "Zero-Copy" 주석 수정 — GPU→CPU 전송은 1번 copy가 불가피.
 *   WebGPU 스펙상 GPU 메모리를 WASM 힙과 직접 공유할 수 없다 (CUDA pinned memory와 달리).
 *   최소 1번의 copy는 WebGPU의 구조적 한계이며 Dawn, wgpu, TensorFlow.js도 동일.
 * ARC-01 Fix: createBuffer() OOM은 device.pushErrorScope()로만 감지 가능 — 문서화.
 */
/**
 * ARC-01 Note: WebGPU에서 createBuffer()는 동기적으로 에러를 던지지 않는다.
 * 실제 OOM은 device.lost로 전파되거나 device.pushErrorScope()로 감지해야 한다.
 * 현재 try/catch는 동기 에러(예: 파라미터 검증)만 잡는다.
 */
export declare function allocateBuffer(byteLength: number, usage: GPUBufferUsageFlags): GPUBuffer;
export declare function writeFloat32Array(buffer: GPUBuffer, data: Float32Array): void;
/** 전통적인 full-copy readback */
export declare function readBufferToFloat32Array(buffer: GPUBuffer, byteLength: number): Promise<Float32Array>;
/**
 * C-05 Fix: mapBufferAsync는 staging buffer를 직접 반환한다.
 * 전역 Map 없이 호출자가 staging buffer 참조를 들고 있어 동시 readback 안전.
 *
 * GPU→CPU 전송 과정:
 * 1. VRAM의 compute buffer를 RAM의 staging buffer로 복사 (GPU copyBufferToBuffer)
 * 2. staging buffer를 WASM에서 읽을 수 있도록 맵핑 (mapAsync)
 * 이 함수는 staging buffer를 반환하며, readMappedInto()에서 최종 읽기를 수행.
 */
export declare function mapBufferAsync(buffer: GPUBuffer, byteLength: number): Promise<GPUBuffer>;
/**
 * C-05 Fix: staging buffer를 인자로 받아 데이터 읽기 수행 후 소각.
 *
 * NH-05 Note: 이 함수는 1번의 메모리 copy를 수행한다.
 * outArray.set()은 staging buffer → WASM 힙으로의 copy다.
 * WebGPU 스펙상 GPU 메모리와 WASM 힙의 직접 공유(진짜 Zero-Copy)는 불가능하다.
 * 이 1번의 copy는 구조적으로 불가피한 최소값이다.
 */
export declare function readMappedInto(stagingBuffer: GPUBuffer, outArray: Float32Array): void;
export declare function freeBuffer(buffer: GPUBuffer, byteLength: number): void;
