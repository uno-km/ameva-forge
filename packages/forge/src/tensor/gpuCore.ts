/**
 * gpuCore.ts — GPU 코어 API (초기화, 텐서 생명주기, 개별 op 디스패치)
 *
 * H-01 Fix: 모든 op에 _globalPipelineCache 적용 (셰이더 재컴파일 방지)
 * NH-03 Fix: quota를 maxStorageBufferBindingSize 기반으로 설정 (maxBufferSize는 단일 버퍼 크기 제한이지 VRAM 용량이 아님)
 * NH-01 Fix: 개별 op 함수들을 internal로 유지, pyodideBridge에서는 executeGraph만 노출
 * NH-07 Fix: shaderGuard.assertAllowedKernelName() 실제 호출
 * ARC-01 Fix: device.pushErrorScope로 OOM 감지 시도
 * L-01 Fix: dispatchKernel 헬퍼로 모든 op의 반복 코드 통합 (DRY)
 */

import { initWebGPU, getDevice, getAdapter, setDeviceLostCallback } from "../webgpu/device";
import {
  allocateBuffer,
  writeFloat32Array,
  readBufferToFloat32Array,
  mapBufferAsync as _mapBufferAsync,
  readMappedInto as _readMappedInto,
} from "../webgpu/buffers";
import { _globalQuotaManager } from "../webgpu/quota";
import { _globalRegistry } from "./tensorRegistry";
import { _globalPipelineCache } from "../webgpu/pipelineCache";
import { assertAllowedKernelName, registerKernelNames } from "../webgpu/shaderGuard";
import { TensorHandle, DType, TensorInfo } from "../types";
import { AMEVAForgeShapeError, AMEVAForgeDTypeError } from "../errors";
import { validateShape } from "./validateShape";
import { validateDType } from "./validateDType";

import { MATMUL_WGSL } from "./kernels/matmul.wgsl";
import { RELU_WGSL } from "./kernels/relu.wgsl";
import { ADD_WGSL } from "./kernels/add.wgsl";
import { TRANSPOSE_WGSL } from "./kernels/transpose.wgsl";
import { MUL_WGSL } from "./kernels/mul.wgsl";
import { RELU_BACKWARD_WGSL } from "./kernels/relu_backward.wgsl";
import { SUB_WGSL } from "./kernels/sub.wgsl";
import { NEG_WGSL } from "./kernels/neg.wgsl";
import { DIV_WGSL } from "./kernels/div.wgsl";
import { EXP_WGSL } from "./kernels/exp.wgsl";
import { LOG_WGSL } from "./kernels/log.wgsl";
import { SIGMOID_WGSL } from "./kernels/sigmoid.wgsl";
import { TANH_WGSL } from "./kernels/tanh.wgsl";
import { SIGMOID_BACKWARD_WGSL } from "./kernels/sigmoid_backward.wgsl";
import { TANH_BACKWARD_WGSL } from "./kernels/tanh_backward.wgsl";
import { FILL_WGSL } from "./kernels/fill.wgsl";
import { SUM_WGSL } from "./kernels/sum.wgsl";
import { MAX_WGSL } from "./kernels/max.wgsl";
import { SUM_AXIS_WGSL } from "./kernels/sum_axis.wgsl";
import { AXPY_WGSL } from "./kernels/axpy.wgsl";
import { PAD_WGSL } from "./kernels/pad.wgsl";
import { GATHER_WGSL } from "./kernels/gather.wgsl";
import { SCATTER_WGSL } from "./kernels/scatter.wgsl";
import { CAT_WGSL } from "./kernels/cat.wgsl";
import { WHERE_WGSL } from "./kernels/where.wgsl";

/**
 * 커널 레지스트리: 새 커널 추가 시 import 1줄 + 여기 1줄만 추가하면
 * warmupKernels(), graphExecutor 모두 자동으로 반영된다.
 */
export const KERNEL_REGISTRY: ReadonlyMap<string, string> = new Map([
  ['matmul', MATMUL_WGSL],
  ['relu', RELU_WGSL],
  ['add', ADD_WGSL],
  ['mul', MUL_WGSL],
  ['transpose', TRANSPOSE_WGSL],
  ['relu_backward', RELU_BACKWARD_WGSL],
  ['sub', SUB_WGSL],
  ['neg', NEG_WGSL],
  ['div', DIV_WGSL],
  ['exp', EXP_WGSL],
  ['log', LOG_WGSL],
  ['sigmoid', SIGMOID_WGSL],
  ['tanh', TANH_WGSL],
  ['sigmoid_backward', SIGMOID_BACKWARD_WGSL],
  ['tanh_backward', TANH_BACKWARD_WGSL],
  ['fill', FILL_WGSL],
  ['sum', SUM_WGSL],
  ['max', MAX_WGSL],
  ['sum_axis', SUM_AXIS_WGSL],
  ['axpy', AXPY_WGSL],
  ['pad', PAD_WGSL],
  ['gather', GATHER_WGSL],
  ['scatter', SCATTER_WGSL],
  ['cat', CAT_WGSL],
  ['where', WHERE_WGSL],
]);

// VUL-001 Fix: Register kernel names automatically to keep whitelist in sync
registerKernelNames(KERNEL_REGISTRY.keys());

// ── 핸들별 staging buffer 관리 (C-05) ──
const _pendingStagingBuffers = new Map<TensorHandle, GPUBuffer>();

export function resetRuntimeMemory(): void {
  _globalRegistry.clear();
  _globalQuotaManager.reset();
  _globalPipelineCache.clear();  // L-03 Fix: device lost 시 파이프라인 캐시도 무효화
  // L-NEW-02: 미처리된 staging buffer도 소각
  for (const [, buf] of _pendingStagingBuffers) {
    try { buf.unmap(); } catch { /* already unmapped */ }
    try { buf.destroy(); } catch { /* already destroyed */ }
  }
  _pendingStagingBuffers.clear();
}

/**
 * NH-03 Fix: 초기화 시 adapter.limits에서 GPU 정보를 조회.
 * maxBufferSize는 단일 버퍼 최대 크기이고 VRAM 총 용량이 아님.
 * maxStorageBufferBindingSize가 실제 단일 바인딩에서 사용 가능한 최대 크기.
 * 총 VRAM 쿼터는 사용자 설정 + 어댑터 힌트로 보수적으로 설정.
 */
function _safeLog(msg: string) {
  try {
    if (typeof (globalThis as any).log === 'function') {
      (globalThis as any).log(msg, 'system');
    }
  } catch (e) {}
}

export async function init(
  options?: GPURequestAdapterOptions & { vramLimitBytes?: number }
): Promise<void> {
  _safeLog(`[gpuCore.ts] init started`);
  setDeviceLostCallback(() => {
    resetRuntimeMemory();
  });

  try {
    _safeLog(`[gpuCore.ts] calling initWebGPU...`);
    await initWebGPU(options);
    _safeLog(`[gpuCore.ts] initWebGPU finished`);
  } catch (e: any) {
    _safeLog(`[gpuCore.ts] initWebGPU threw error: ${e.message}`);
    throw e;
  }


  // NH-03: 실제 GPU 제한 조회 후 쿼터 조정
  const adapter = getAdapter();
  if (adapter) {
    const limits = adapter.limits;

    if (limits.maxComputeWorkgroupSizeX < 64) {
      console.warn(`[AMEVA] Warning: Device maxComputeWorkgroupSizeX (${limits.maxComputeWorkgroupSizeX}) is less than 64. Kernels are optimized for 64.`);
    }

    // maxStorageBufferBindingSize: 단일 storage 버퍼 바인딩 최대 크기 (VRAM 상한 근사)
    // 통상 256MB~2GB. maxBufferSize는 단일 GPUBuffer 최대 크기 (≠ VRAM 총량)
    const maxBinding = limits.maxStorageBufferBindingSize ?? 256 * 1024 * 1024;

    // 사용자가 vramLimitBytes를 지정하면 그것을 우선 사용
    const userLimit = options?.vramLimitBytes;
    const hardLimit = userLimit
      ? Math.min(userLimit, 8 * 1024 * 1024 * 1024)
      : Math.min(maxBinding * 4, 8 * 1024 * 1024 * 1024); // binding 크기의 4배를 총 VRAM 추정
    const softLimit = Math.floor(hardLimit * 0.75);

    _globalQuotaManager.setLimits(Math.floor(hardLimit), Math.floor(softLimit));
    console.info(
      `[AMEVA] GPU quota set: soft=${(softLimit / 1e9).toFixed(2)}GB, ` +
      `hard=${(hardLimit / 1e9).toFixed(2)}GB ` +
      `(maxStorageBindingSize=${(maxBinding / 1e9).toFixed(2)}GB)`
    );
  }

  // H-NEW-08: 비동기 파이프라인 사전 컴파일
  await warmupKernels();
}

/**
 * 모든 커널 파이프라인을 비동기로 사전 컴파일한다.
 * KERNEL_REGISTRY를 동적으로 순회하므로, 새 커널 추가 시 여기를 수정할 필요 없다.
 */
export async function warmupKernels(): Promise<void> {
  const entries = Array.from(KERNEL_REGISTRY.entries()).map(
    ([key, wgslCode]) => ({ key, wgslCode })
  );
  await _globalPipelineCache.warmup(entries);
}

export function getTensorInfo(handle: TensorHandle): TensorInfo {
  const record = _globalRegistry.get(handle);
  return {
    handle: record.handle,
    shape: [...record.shape],
    dtype: record.dtype,
    byteLength: record.byteLength,
    disposed: record.disposed
  };
}

export function read(handle: TensorHandle): Promise<Float32Array> {
  const record = _globalRegistry.get(handle);
  return readBufferToFloat32Array(record.buffer, record.byteLength);
}

/**
 * C-05 Fix: staging buffer를 핸들 키로 _pendingStagingBuffers에 저장.
 */
export async function mapBufferAsync(handle: TensorHandle): Promise<void> {
  const record = _globalRegistry.get(handle);
  const stagingBuffer = await _mapBufferAsync(record.buffer, record.byteLength);
  _globalQuotaManager.track(stagingBuffer.size);
  _pendingStagingBuffers.set(handle, stagingBuffer);
}

/**
 * C-05 Fix: 핸들 키로 staging buffer를 조회하여 읽기.
 */
export function readMappedInto(handle: TensorHandle, outArray: any): void {
  const stagingBuffer = _pendingStagingBuffers.get(handle);
  if (!stagingBuffer) {
    throw new Error(
      `[AMEVA] No staged buffer for handle "${handle}". Call mapBufferAsync first.`
    );
  }
  _pendingStagingBuffers.delete(handle);

  let bufProxy: any = null;
  try {
    let actualData: Float32Array;
    if (outArray && typeof outArray.getBuffer === 'function') {
      bufProxy = outArray.getBuffer("f32");
      actualData = bufProxy.data;
    } else {
      actualData = outArray as Float32Array;
    }
    _readMappedInto(stagingBuffer, actualData);
  } finally {
    _globalQuotaManager.release(stagingBuffer.size);
    // H-NEW-06: bufProxy.release() 실패 시에도 리소스 정리 보장
    if (bufProxy) {
      try { bufProxy.release(); } catch { /* ignore */ }
    }
  }
}

export function dispose(handle: TensorHandle): void {
  _globalRegistry.dispose(handle);
}

// ─────────────────────────────────────────────────────────────────────────────
// L-01 Fix: dispatchKernel 헬퍼 — 모든 op의 반복 코드를 통합
// NH-07 Fix: assertAllowedKernelName() 호출
// ─────────────────────────────────────────────────────────────────────────────

interface KernelDispatchOptions {
  opKey: string;
  wgslCode: string;
  paramsData: Uint32Array;
  inputBuffers: GPUBuffer[];
  outBuffer: GPUBuffer;
  dispatchX: number;
  dispatchY?: number;
}

function dispatchKernel(opts: KernelDispatchOptions): void {
  // NH-07 Fix: shaderGuard에서 커널 이름 검증
  assertAllowedKernelName(opts.opKey);

  const device = getDevice();

  const paramsBuffer = device.createBuffer({
    size: Math.max(16, opts.paramsData.byteLength), // 최소 16바이트 (WebGPU uniform 정렬)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, opts.paramsData.buffer);

  // H-01: 파이프라인 캐시에서 조회 (없으면 컴파일 후 캐시)
  const { pipeline } = _globalPipelineCache.getPipeline(opts.opKey, opts.wgslCode);

  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: paramsBuffer } },
    ...opts.inputBuffers.map((buf, i) => ({
      binding: i + 1,
      resource: { buffer: buf }
    })),
    { binding: opts.inputBuffers.length + 1, resource: { buffer: opts.outBuffer } }
  ];

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(opts.dispatchX, opts.dispatchY ?? 1);
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);

  // params 버퍼는 GPU 제출 완료 후 즉시 소각
  void device.queue.onSubmittedWorkDone().then(() => paramsBuffer.destroy());
}

// ─────────────────────────────────────────────────────────────────────────────
// 개별 op 함수들 (내부 사용, pyodideBridge에서는 executeGraph를 통해서만 접근)
// NH-01 Note: 이 함수들은 JS 테스트와 직접 호출에서만 사용
// ─────────────────────────────────────────────────────────────────────────────

export function random(shape: number[], dtype: DType = "float32"): TensorHandle {
  validateDType(dtype);
  const elements = validateShape(shape, dtype);
  const data = new Float32Array(elements);
  for (let i = 0; i < elements; i++) data[i] = Math.random();
  const byteLength = elements * 4;
  const buffer = allocateBuffer(
    byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  writeFloat32Array(buffer, data);
  return _globalRegistry.register({ buffer, shape, dtype, byteLength });
}

export function uploadFloat32Array(data: any, shape: number[]): TensorHandle {
  let actualData: Float32Array;
  let bufProxy: any = null;
  if (data && typeof data.getBuffer === 'function') {
    bufProxy = data.getBuffer("f32");
    actualData = bufProxy.data;
  } else {
    actualData = data as Float32Array;
  }
  const elements = validateShape(shape, "float32", actualData.byteLength);
  const byteLength = elements * 4;
  const buffer = allocateBuffer(
    byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  writeFloat32Array(buffer, actualData);
  if (bufProxy) bufProxy.release();
  return _globalRegistry.register({ buffer, shape, dtype: "float32", byteLength });
}

export function matmul(handleA: TensorHandle, handleB: TensorHandle): TensorHandle {
  const a = _globalRegistry.get(handleA);
  const b = _globalRegistry.get(handleB);

  if (a.shape.length !== 2 || b.shape.length !== 2)
    throw new AMEVAForgeShapeError("Matmul requires 2D tensors");
  if (a.dtype !== "float32" || b.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Matmul requires float32 tensors");

  const M = a.shape[0], K = a.shape[1], K2 = b.shape[0], N = b.shape[1];
  if (K !== K2) throw new AMEVAForgeShapeError(`Inner dim mismatch: ${K} != ${K2}`);

  const byteLength = M * N * 4;
  const cBuffer = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  dispatchKernel({
    opKey: 'matmul',
    wgslCode: MATMUL_WGSL,
    paramsData: new Uint32Array([M, N, K, 0]),
    inputBuffers: [a.buffer, b.buffer],
    outBuffer: cBuffer,
    // M-05: X=col방향=N, Y=row방향=M
    dispatchX: Math.ceil(N / 8),
    dispatchY: Math.ceil(M / 8),
  });

  return _globalRegistry.register({ buffer: cBuffer, shape: [M, N], dtype: "float32", byteLength });
}

export function relu(handle: TensorHandle): TensorHandle {
  const x = _globalRegistry.get(handle);
  if (x.dtype !== "float32") throw new AMEVAForgeDTypeError("ReLU requires float32");
  const numElements = x.byteLength / 4;
  const outBuffer = allocateBuffer(x.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  dispatchKernel({
    opKey: 'relu',
    wgslCode: RELU_WGSL,
    paramsData: new Uint32Array([numElements, 0, 0, 0]),
    inputBuffers: [x.buffer],
    outBuffer,
    dispatchX: Math.ceil(numElements / 64),
  });

  return _globalRegistry.register({ buffer: outBuffer, shape: [...x.shape], dtype: "float32", byteLength: x.byteLength });
}

export function add(handleA: TensorHandle, handleB: TensorHandle): TensorHandle {
  const a = _globalRegistry.get(handleA);
  const b = _globalRegistry.get(handleB);
  if (a.byteLength !== b.byteLength)
    throw new AMEVAForgeShapeError("Add requires tensors of the same shape");
  if (a.dtype !== "float32" || b.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Add requires float32");

  const numElements = a.byteLength / 4;
  const outBuffer = allocateBuffer(a.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  dispatchKernel({
    opKey: 'add',
    wgslCode: ADD_WGSL,
    paramsData: new Uint32Array([numElements, 0, 0, 0]),
    inputBuffers: [a.buffer, b.buffer],
    outBuffer,
    dispatchX: Math.ceil(numElements / 64),
  });

  return _globalRegistry.register({ buffer: outBuffer, shape: [...a.shape], dtype: "float32", byteLength: a.byteLength });
}

export function mul(handleA: TensorHandle, handleB: TensorHandle): TensorHandle {
  const a = _globalRegistry.get(handleA);
  const b = _globalRegistry.get(handleB);
  if (a.byteLength !== b.byteLength)
    throw new AMEVAForgeShapeError("Mul requires tensors of the same shape");
  if (a.dtype !== "float32" || b.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Mul requires float32");

  const numElements = a.byteLength / 4;
  const outBuffer = allocateBuffer(a.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  dispatchKernel({
    opKey: 'mul',
    wgslCode: MUL_WGSL,
    paramsData: new Uint32Array([numElements, 0, 0, 0]),
    inputBuffers: [a.buffer, b.buffer],
    outBuffer,
    dispatchX: Math.ceil(numElements / 64),
  });

  return _globalRegistry.register({ buffer: outBuffer, shape: [...a.shape], dtype: "float32", byteLength: a.byteLength });
}

export function transpose(handle: TensorHandle): TensorHandle {
  const x = _globalRegistry.get(handle);
  if (x.shape.length !== 2)
    throw new AMEVAForgeShapeError("Transpose requires 2D tensors");
  if (x.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Transpose requires float32");

  const M = x.shape[0], N = x.shape[1];
  const outBuffer = allocateBuffer(x.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  dispatchKernel({
    opKey: 'transpose',
    wgslCode: TRANSPOSE_WGSL,
    paramsData: new Uint32Array([M, N, 0, 0]),
    inputBuffers: [x.buffer],
    outBuffer,
    // transpose 셰이더: row=global_id.x, col=global_id.y
    dispatchX: Math.ceil(M / 8),
    dispatchY: Math.ceil(N / 8),
  });

  return _globalRegistry.register({ buffer: outBuffer, shape: [N, M], dtype: "float32", byteLength: x.byteLength });
}

export function relu_backward(handleX: TensorHandle, handleGrad: TensorHandle): TensorHandle {
  const x = _globalRegistry.get(handleX);
  const grad = _globalRegistry.get(handleGrad);
  if (x.byteLength !== grad.byteLength)
    throw new AMEVAForgeShapeError("ReLU backward: shape mismatch");

  const numElements = x.byteLength / 4;
  const outBuffer = allocateBuffer(x.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  dispatchKernel({
    opKey: 'relu_backward',
    wgslCode: RELU_BACKWARD_WGSL,
    paramsData: new Uint32Array([numElements, 0, 0, 0]),
    inputBuffers: [x.buffer, grad.buffer],
    outBuffer,
    dispatchX: Math.ceil(numElements / 64),
  });

  return _globalRegistry.register({ buffer: outBuffer, shape: [...x.shape], dtype: "float32", byteLength: x.byteLength });
}
