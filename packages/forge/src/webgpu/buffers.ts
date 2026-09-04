/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * buffers.ts — GPU 버퍼 할당, 읽기 인터페이스
 *
 * C-05 Fix: _stagingBuffers 전역 Map 제거 → mapBufferAsync가 staging buffer 직접 반환.
 * H-05 / NH-05 Fix: "Zero-Copy" 주석 수정 — GPU→CPU 전송은 1번 copy가 불가피.
 *   WebGPU 스펙상 GPU 메모리를 WASM 힙과 직접 공유할 수 없다 (CUDA pinned memory와 달리).
 *   최소 1번의 copy는 WebGPU의 구조적 한계이며 Dawn, wgpu, TensorFlow.js도 동일.
 * ARC-01 Fix: createBuffer() OOM은 device.pushErrorScope()로만 감지 가능 — 문서화.
 */

import { getDevice, getQueue } from "./device";
import { _globalQuotaManager, AllocationKind, AllocationToken } from "./quota";
import { AMEVAForgeValidationError } from "../errors";

const GPU_MAP_MODE_READ = typeof GPUMapMode !== 'undefined' ? GPUMapMode.READ : 0x0001;

/**
 * WHAT: 지정된 크기와 용도에 맞게 GPU 버퍼를 할당합니다.
 * WHY: WebGPU의 버퍼 생성을 추상화하고 전역 할당량(Quota) 관리 시스템과 통합하여 메모리 부족(OOM)을 방지하기 위해 존재합니다.
 * HOW: QuotaManager를 통해 `byteLength`만큼의 메모리를 예약한 후, `device.createBuffer`를 호출하여 버퍼를 생성합니다. 실패 시 예약된 메모리 토큰을 반환(release)하고 에러를 던집니다.
 */
export function allocateBuffer(
  byteLength: number,
  usage: GPUBufferUsageFlags,
  kind: AllocationKind = 'tensor',
  ownerGraph: string | null = null
): { buffer: GPUBuffer, token: AllocationToken } {
  const token = _globalQuotaManager.reserveToken(byteLength, kind, ownerGraph);
  try {
    const buffer = getDevice().createBuffer({ size: byteLength, usage });
    return { buffer, token };
  } catch (e) {
    _globalQuotaManager.releaseToken(token);
    throw e;
  }
}

/**
 * WHAT: 주어진 GPU 버퍼에 Float32Array 데이터를 씁니다.
 * WHY: CPU 측의 데이터를 GPU 버퍼로 복사하여 GPU 연산에 사용할 수 있도록 하기 위해 필요합니다.
 * HOW: WebGPU 큐(`device.queue.writeBuffer`)를 사용하여 주어진 데이터의 전체 크기만큼 지정된 버퍼의 오프셋 0부터 복사합니다.
 */
export function writeFloat32Array(buffer: GPUBuffer, data: Float32Array): void {
  if (data.byteLength > buffer.size) {
    throw new AMEVAForgeValidationError(
      `writeFloat32Array overflow: data size (${data.byteLength}B) exceeds buffer capacity (${buffer.size}B)`
    );
  }
  getQueue().writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
}

type PoolCleaner = () => void;
type PoolRetirer = (device: GPUDevice) => Promise<void>;

const _transientPoolCleaners: PoolCleaner[] = [];
const _transientPoolRetirers: PoolRetirer[] = [];

export function registerTransientPool(cleaner: PoolCleaner, retirer?: PoolRetirer): void {
  _transientPoolCleaners.push(cleaner);
  if (retirer) _transientPoolRetirers.push(retirer);
}

interface StagingPoolEntry {
  buffer: GPUBuffer;
  token: AllocationToken;
}

export const _stagingPool: Map<number, StagingPoolEntry[]> = new Map();
const STAGING_POOL_MAX_PER_SIZE = 4;

export function clearStagingPool(): void {
  for (const entries of _stagingPool.values()) {
    for (const { buffer, token } of entries) {
      try { freeBuffer(buffer, token); } catch {}
    }
  }
  _stagingPool.clear();
  for (const cleaner of _transientPoolCleaners) {
    try { cleaner(); } catch {}
  }
}

export async function flushGC(): Promise<void> {
  try {
    const device = getDevice();
    await device.queue.onSubmittedWorkDone();
    for (const retirer of _transientPoolRetirers) {
      try { await retirer(device); } catch {}
    }
  } catch {}
  clearStagingPool();
}

export function getStagingBucketSize(byteLength: number): number {
  if (byteLength <= 64) return 64;
  return Math.pow(2, Math.ceil(Math.log2(byteLength)));
}

export function acquireStagingBuffer(byteLength: number): { buffer: GPUBuffer, token: AllocationToken, bucketSize: number } {
  const bucketSize = getStagingBucketSize(byteLength);
  const pool = _stagingPool.get(bucketSize);
  if (pool && pool.length > 0) {
    const entry = pool.pop()!;
    return { buffer: entry.buffer, token: entry.token, bucketSize };
  }
  const usage = typeof GPUBufferUsage !== 'undefined' ? (GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST) : (0x0001 | 0x0008);
  const { buffer, token } = allocateBuffer(
    bucketSize,
    usage,
    'staging',
    'StagingPool'
  );
  return { buffer, token, bucketSize };
}

export function releaseStagingBuffer(
  buffer: GPUBuffer,
  token: AllocationToken,
  byteLength: number,
  isCorrupted: boolean = false
): void {
  if (isCorrupted) {
    try { buffer.destroy(); } catch {}
    if (token) {
      try { _globalQuotaManager.releaseToken(token); } catch {}
    }
    return;
  }

  const bucketSize = getStagingBucketSize(byteLength);
  const pool = _stagingPool.get(bucketSize) ?? [];
  if (pool.length < STAGING_POOL_MAX_PER_SIZE) {
    pool.push({ buffer, token });
    _stagingPool.set(bucketSize, pool);
  } else {
    try { freeBuffer(buffer, token); } catch {}
  }
}

/**
 * WHAT: GPU 버퍼의 데이터를 읽어서 CPU 메모리 상의 Float32Array로 반환합니다.
 * WHY: GPU에서 처리된 결과 데이터를 CPU로 가져와서 애플리케이션 수준에서 활용(예: 출력, 저장)하기 위해 존재합니다.
 * HOW: 
 *   1. 복사를 위한 중간 버퍼(Staging Buffer)를 MAP_READ와 COPY_DST 용도로 할당합니다.
 *   2. CommandEncoder를 사용해 원본 버퍼의 데이터를 Staging Buffer로 복사하고 큐에 제출합니다.
 *   3. Staging Buffer를 비동기적으로 맵핑(mapAsync)하여 CPU에서 읽을 수 있게 합니다.
 *   4. 데이터를 읽어 Float32Array로 복사한 후 버퍼를 해제(unmap, destroy)하고 토큰을 반환합니다.
 */
export async function readBufferToFloat32Array(
  buffer: GPUBuffer,
  byteLength: number
): Promise<Float32Array> {
  const device = getDevice();
  const { buffer: stagingBuffer, token, bucketSize } = acquireStagingBuffer(byteLength);
  let isCorrupted = false;

  try {
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
    device.queue.submit([commandEncoder.finish()]);
    await stagingBuffer.mapAsync(GPU_MAP_MODE_READ);

    try {
      const arrayBuffer = stagingBuffer.getMappedRange(0, byteLength);
      return new Float32Array(arrayBuffer.slice(0));
    } finally {
      stagingBuffer.unmap();
    }
  } catch (err) {
    isCorrupted = true;
    throw err;
  } finally {
    releaseStagingBuffer(stagingBuffer, token, bucketSize, isCorrupted);
  }
}

/**
 * WHAT: GPU 버퍼의 내용을 읽기 위해 Staging Buffer를 생성하고 비동기적으로 맵핑합니다.
 * WHY: 대용량 데이터 전송 시 메모리 맵핑을 직접 제어하거나 제로 카피(Zero-Copy) 메커니즘과 유사한 최적화를 구현하기 위해 필요합니다.
 * HOW: `MAP_READ | COPY_DST` 속성의 Staging 버퍼를 새로 할당하고, 원본 버퍼의 내용을 복사하기 위한 커맨드를 큐에 제출한 뒤, `mapAsync`를 호출하여 맵핑된 버퍼와 할당 토큰을 반환합니다.
 */
export async function mapBufferAsync(
  buffer: GPUBuffer,
  byteLength: number
): Promise<{ stagingBuffer: GPUBuffer, token: AllocationToken, byteLength: number }> {
  const device = getDevice();
  const { buffer: stagingBuffer, token, bucketSize } = acquireStagingBuffer(byteLength);

  try {
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
    device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPU_MAP_MODE_READ);
    return { stagingBuffer, token, byteLength };
  } catch (e) {
    releaseStagingBuffer(stagingBuffer, token, bucketSize, true);
    throw e;
  }
}

/**
 * WHAT: 맵핑이 완료된 Staging 버퍼의 데이터를 외부에서 제공된 Float32Array 배열에 직접 복사합니다.
 * WHY: 새로운 배열 객체를 생성하지 않고 기존 메모리(Pre-allocated buffer)를 재사용하여 메모리 할당 및 가비지 컬렉션(GC) 부하를 줄이기 위해 사용됩니다.
 * HOW: Staging 버퍼의 맵핑 범위를 가져와서 전달된 `outArray`에 `set` 메서드로 데이터를 덮어쓴 후, unmap 후 Staging Pool로 반환합니다.
 */
export function readMappedInto(
  stagingBuffer: GPUBuffer,
  token: AllocationToken,
  outArray: Float32Array
): void {
  const byteLength = outArray.byteLength;
  let isCorrupted = false;
  try {
    const arrayBuffer = stagingBuffer.getMappedRange(0, byteLength);
    const mapped = new Float32Array(arrayBuffer);
    if (outArray.length !== mapped.length) {
      throw new RangeError(`readMappedInto destination length mismatch: expected ${mapped.length}, got ${outArray.length}`);
    }
    outArray.set(mapped);
  } catch (err) {
    if (!(err instanceof RangeError)) {
      isCorrupted = true;
    }
    throw err;
  } finally {
    try { stagingBuffer.unmap(); } catch {}
    releaseStagingBuffer(stagingBuffer, token, byteLength, isCorrupted);
  }
}

/**
 * WHAT: 할당된 GPU 버퍼를 메모리에서 해제하고, 관련된 할당량 토큰(AllocationToken)을 반환합니다.
 * WHY: WebGPU 리소스 누수를 방지하고, 전역 쿼타 매니저(Quota Manager)에 반환하여 다른 작업에서 가용 메모리를 사용할 수 있도록 하기 위해 존재합니다.
 * HOW: `buffer.destroy()`를 호출하여 실제 GPU 리소스를 해제한 다음, `_globalQuotaManager.releaseToken(token)`을 통해 예약된 메모리 용량을 반환합니다.
 */
export function freeBuffer(buffer: GPUBuffer, token: AllocationToken): void {
  buffer.destroy();
  _globalQuotaManager.releaseToken(token);
}
