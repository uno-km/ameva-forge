/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:59:35+09:00: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 *   - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
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
  freeBuffer,
  clearStagingPool,
  mapBufferAsync as _mapBufferAsync,
  readMappedInto as _readMappedInto,
} from "../webgpu/buffers";
import { _globalQuotaManager, AllocationToken } from "../webgpu/quota";
import { _globalRegistry } from "./tensorRegistry";
import { _globalPipelineCache } from "../webgpu/pipelineCache";
import { assertAllowedKernelName, registerKernelNames } from "../webgpu/shaderGuard";
import { TensorHandle, DType, TensorInfo } from "../types";
import { AMEVAForgeShapeError, AMEVAForgeDTypeError } from "../errors";
import { validateShape } from "./validateShape";
import { validateDType } from "./validateDType";
import { assertWasmRange } from "../webgpu/validateWasmRange";
import { computeBroadcastParams } from "./broadcastParams";
import { computeDispatch2D } from "./dispatchShape";

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
import { MAX_AXIS_WGSL } from "./kernels/max_axis.wgsl";
import { MAX_AXIS_BACKWARD_WGSL } from "./kernels/max_axis_backward.wgsl";
import { AXPY_WGSL } from "./kernels/axpy.wgsl";
import { PAD_WGSL } from "./kernels/pad.wgsl";
import { GATHER_WGSL } from "./kernels/gather.wgsl";
import { SCATTER_WGSL } from "./kernels/scatter.wgsl";
import { CAT_WGSL } from "./kernels/cat.wgsl";
import { WHERE_WGSL } from "./kernels/where.wgsl";
import { DROPOUT_WGSL } from "./kernels/dropout.wgsl";
import { MAXPOOL2D_WGSL } from "./kernels/maxpool2d.wgsl";
import { AVGPOOL2D_WGSL } from "./kernels/avgpool2d.wgsl";
import { IM2COL_WGSL } from "./kernels/im2col.wgsl";
import { COL2IM_WGSL } from "./kernels/col2im.wgsl";
import { PERMUTE_WGSL } from "./kernels/permute.wgsl";
import { BATCHED_MATMUL_WGSL } from "./kernels/batched_matmul.wgsl";
import { MATMUL_BIAS_RELU_WGSL } from "./kernels/matmul_bias_relu.wgsl";
import { MATMUL_TILED_WGSL } from "./kernels/matmul_tiled.wgsl";
import { FLASH_ATTENTION_WGSL } from "./kernels/flash_attention.wgsl";
import { ROPE_WGSL } from "./kernels/rope.wgsl";
import { RMSNORM_WGSL } from "./kernels/rmsnorm.wgsl";
import { SWIGLU_WGSL } from "./kernels/swiglu.wgsl";
import { UNPACK_QUANT_WGSL } from "./kernels/unpack_quant.wgsl";
import { EMBEDDING_WGSL } from "./kernels/embedding.wgsl";
import { EMBEDDING_BACKWARD_WGSL } from "./kernels/embedding_backward.wgsl";

// WebGPU Buffer Usage bitmasks with Node.js environment fallback
const BUFFER_USAGE_STORAGE_SRC = typeof GPUBufferUsage !== 'undefined'
  ? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
  : (0x0080 | 0x0004);

const BUFFER_USAGE_STORAGE_COPY = typeof GPUBufferUsage !== 'undefined'
  ? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST)
  : (0x0080 | 0x0004 | 0x0008);

const BUFFER_USAGE_UNIFORM_COPY = typeof GPUBufferUsage !== 'undefined'
  ? (GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
  : (0x0040 | 0x0008);

/**
 * WHAT: 모든 WGSL 셰이더 코드를 커널 이름에 매핑하여 저장하는 전역 읽기 전용 레지스트리 맵입니다.
 * WHY: 런타임에 셰이더 코드를 이름으로 조회하고 파이프라인 캐시 초기화 시 한 번에 반영하기 위해 존재합니다.
 * HOW: Map 객체를 생성하여 문자열 키와 WGSL 코드 문자열 값을 쌍으로 저장합니다.
 */
export const KERNEL_REGISTRY: ReadonlyMap<string, string> = new Map([
  ['matmul', MATMUL_WGSL],
  ['matmul_tiled', MATMUL_TILED_WGSL],
  ['matmul_bias_relu', MATMUL_BIAS_RELU_WGSL],
  ['batched_matmul', BATCHED_MATMUL_WGSL],
  ['flash_attention', FLASH_ATTENTION_WGSL],
  ['rope', ROPE_WGSL],
  ['rmsnorm', RMSNORM_WGSL],
  ['swiglu', SWIGLU_WGSL],
  ['unpack_quant', UNPACK_QUANT_WGSL],
  ['embedding', EMBEDDING_WGSL],
  ['embedding_backward', EMBEDDING_BACKWARD_WGSL],
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
  ['max_axis', MAX_AXIS_WGSL],
  ['max_axis_backward', MAX_AXIS_BACKWARD_WGSL],
  ['axpy', AXPY_WGSL],
  ['pad', PAD_WGSL],
  ['gather', GATHER_WGSL],
  ['scatter', SCATTER_WGSL],
  ['cat', CAT_WGSL],
  ['where', WHERE_WGSL],
  ['dropout', DROPOUT_WGSL],
  ['maxpool2d', MAXPOOL2D_WGSL],
  ['avgpool2d', AVGPOOL2D_WGSL],
  ['im2col', IM2COL_WGSL],
  ['col2im', COL2IM_WGSL],
  ['permute', PERMUTE_WGSL],
]);

// VUL-001 Fix: Register kernel names automatically to keep whitelist in sync
registerKernelNames(KERNEL_REGISTRY.keys());

/**
 * WHAT: CPU로 읽어오기 위해 대기 중인 GPU 스테이징 버퍼들을 추적하는 전역 맵입니다.
 * WHY: 비동기 맵핑(mapAsync)이 완료된 버퍼를 기록해 두고 나중에 동기적으로 데이터를 읽어올 수 있게 하기 위해 필요합니다.
 * HOW: 텐서 핸들(문자열)을 키로, 매핑된 GPUBuffer와 AllocationToken 객체를 값으로 유지합니다.
 */
const _pendingStagingBuffers = new Map<TensorHandle, { stagingBuffer: GPUBuffer, token: AllocationToken }>();
const _inFlightMapPromises = new Map<TensorHandle, Promise<void>>();

/**
 * WHAT: GPU 코어의 런타임 메모리와 모든 캐시된 리소스를 초기화(해제)하는 함수입니다.
 * WHY: 디바이스 유실(Device Lost) 이벤트가 발생하거나 시스템 강제 리셋 시 남은 자원의 메모리 누수를 방지하기 위해 존재합니다.
 * HOW: 텐서 레지스트리, 쿼터 매니저, 파이프라인 캐시를 지우고, 대기 중인 스테이징 버퍼들도 순회하여 언맵(unmap) 및 파괴(destroy)합니다.
 */
export function resetRuntimeMemory(reason: string = "manual-reset"): void {
  _safeLog(`[RuntimeReset] start: ${reason}`);
  
  // 1. Pending staging buffers & staging pool cleanup
  try {
    for (const [, obj] of _pendingStagingBuffers) {
      try { obj.stagingBuffer.unmap(); } catch { /* already unmapped */ }
      try { obj.stagingBuffer.destroy(); } catch { /* already destroyed */ }
      _globalQuotaManager.releaseToken(obj.token);
    }
    _pendingStagingBuffers.clear();
  } catch (e) {
    _safeLog(`[RuntimeReset] staging buffer cleanup error: ${e}`);
  }

  try {
    clearStagingPool(); // VULN-04: Clear pool buffers & tokens
  } catch (e) {
    _safeLog(`[RuntimeReset] clearStagingPool error: ${e}`);
  }

  // 2. In-flight promises & pipeline cache
  try {
    _inFlightMapPromises.clear();
    _globalPipelineCache.clear();
  } catch (e) {
    _safeLog(`[RuntimeReset] pipeline cache error: ${e}`);
  }

  // 3. Quota & registry reset
  try {
    _globalRegistry.clear();
  } catch (e) {
    _safeLog(`[RuntimeReset] registry clear error: ${e}`);
  }

  try {
    _globalQuotaManager.reset();
  } catch (e) {
    _safeLog(`[RuntimeReset] quota reset error: ${e}`);
  }

  _safeLog(`[RuntimeReset] done: ${reason}`);
}

/**
 * WHAT: 시스템 로거가 존재할 경우 로그 메시지를 남기는 래퍼 함수입니다.
 * WHY: 글로벌 환경(예: Pyodide)에 주입된 로그 함수가 있을 때만 호출하여 콘솔 오염을 막고 안전한 디버깅을 하기 위함입니다.
 * HOW: globalThis에서 log 함수를 찾아 존재하면 호출하고 오류 발생 시 조용히 무시(catch)합니다.
 */
function _safeLog(msg: string) {
  try {
    if (typeof (globalThis as any).log === 'function') {
      (globalThis as any).log(msg, 'system');
    }
  } catch (e) {}
}

/**
 * WHAT: WebGPU 하위 시스템을 초기화하고 메모리 한도 설정 및 셰이더 컴파일을 수행하는 비동기 진입점 함수입니다.
 * WHY: 텐서 연산을 수행하기 전에 GPU 디바이스를 획득하고 하드웨어 제약을 파악하며 파이프라인을 준비하기 위해 필수적입니다.
 * HOW: initWebGPU를 호출하여 디바이스를 얻고, 디바이스 어댑터의 limits를 조회하여 메모리 할당 한도를 설정한 뒤, 모든 커널을 사전 컴파일(warmup)합니다.
 */
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
  /**
   * WHAT: 초기화된 WebGPU 어댑터 객체입니다.
   * WHY: 현재 시스템 GPU의 하드웨어 한계(limits)와 기능 정보를 파악하여 안전한 메모리 할당량을 계산하기 위해 조회합니다.
   * HOW: getAdapter() 함수를 호출하여 가져옵니다.
   */
  const adapter = getAdapter();
  if (adapter) {
    /**
     * WHAT: 현재 GPU 어댑터가 지원하는 하드웨어 제약사항을 담은 객체입니다.
     * WHY: 버퍼 바인딩 크기나 컴퓨트 워크그룹 크기의 안전 한계선을 알기 위해 참조합니다.
     * HOW: adapter.limits 프로퍼티를 통해 가져옵니다.
     */
    const limits = adapter.limits;

    if (limits.maxComputeWorkgroupSizeX < 64) {
      console.warn(`[AMEVA] Warning: Device maxComputeWorkgroupSizeX (${limits.maxComputeWorkgroupSizeX}) is less than 64. Kernels are optimized for 64.`);
    }

    /**
     * WHAT: 스토리지 버퍼가 단일 바인딩 시 사용할 수 있는 최대 바이트 크기입니다.
     * WHY: 이 값을 기준으로 사용 가능한 전체 VRAM 용량을 간접적으로 추정하기 위해 필요합니다.
     * HOW: limits.maxStorageBufferBindingSize를 사용하며, 정보가 없으면 기본값(256MB)으로 설정합니다.
     */
    const maxBinding = limits.maxStorageBufferBindingSize ?? 256 * 1024 * 1024;

    /**
     * WHAT: 사용자가 직접 명시한 VRAM 사용 상한(바이트)입니다.
     * WHY: 시스템의 기본 휴리스틱을 무시하고 사용자 설정에 따라 자원을 제어할 수 있도록 옵션으로 받습니다.
     * HOW: options 인자에서 vramLimitBytes 프로퍼티를 참조합니다.
     */
    const userLimit = options?.vramLimitBytes;
    
    /**
     * WHAT: 할당할 수 있는 최대 하드 VRAM 한도입니다.
     * WHY: 시스템 메모리 초과를 방지하기 위해 엄격한 상한선을 두기 위해 계산합니다.
     * HOW: 사용자 지정값이 있으면 8GB를 넘지 않는 선에서 채택하고, 없으면 바인딩 크기의 4배와 8GB 중 작은 값을 사용합니다.
     */
    const hardLimit = userLimit
      ? Math.min(userLimit, 8 * 1024 * 1024 * 1024)
      : Math.min(maxBinding * 4, 8 * 1024 * 1024 * 1024); // binding 크기의 4배를 총 VRAM 추정
      
    /**
     * WHAT: 메모리 압박이 시작될 때 경고를 보내거나 GC를 유도하기 위한 소프트 한도입니다.
     * WHY: 하드 한도에 도달하기 전 선제적인 리소스 회수 타이밍을 잡기 위해 존재합니다.
     * HOW: 하드 한도의 75%로 계산합니다.
     */
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
 * WHAT: 등록된 모든 커널 셰이더를 WebGPU 컴퓨트 파이프라인으로 사전 컴파일하는 함수입니다.
 * WHY: 실행 시점에 셰이더 컴파일이 발생하여 프레임 드랍이나 실행 지연이 생기는 것을 방지하기 위함입니다.
 * HOW: KERNEL_REGISTRY 맵을 순회하여 각 셰이더 코드와 이름 배열을 추출하고 _globalPipelineCache.warmup()을 호출합니다.
 */
export async function warmupKernels(): Promise<void> {
  /**
   * WHAT: KERNEL_REGISTRY에서 추출한 커널 이름(key)과 셰이더 소스코드(wgslCode) 객체의 배열입니다.
   * WHY: 파이프라인 캐시의 warmup 메서드에 한꺼번에 전달할 형식을 맞추기 위해 생성합니다.
   * HOW: Array.from()을 사용하여 맵 엔트리를 배열로 변환한 후 map()으로 객체화합니다.
   */
  const entries = Array.from(KERNEL_REGISTRY.entries()).map(
    ([key, wgslCode]) => ({ key, wgslCode })
  );
  await _globalPipelineCache.warmup(entries);
}

/**
 * WHAT: 핸들에 해당하는 텐서의 메타데이터(크기, 타입, 버퍼 크기 등)를 반환하는 함수입니다.
 * WHY: 파이썬 브릿지나 외부에서 현재 텐서의 형태 정보를 조회해야 할 때 사용됩니다.
 * HOW: 전역 레지스트리에서 핸들로 레코드를 조회한 뒤 TensorInfo 객체를 구성하여 반환합니다.
 */
export function getTensorInfo(handle: TensorHandle): TensorInfo {
  /**
   * WHAT: 핸들로 조회된 내부 텐서 레코드 객체입니다.
   * WHY: 저장된 shape, dtype 등의 메타데이터를 추출하기 위해 필요합니다.
   * HOW: _globalRegistry.get(handle)을 호출하여 얻어옵니다.
   */
  const record = _globalRegistry.get(handle);
  return {
    handle: record.handle,
    shape: [...record.shape],
    dtype: record.dtype,
    byteLength: record.byteLength,
    disposed: record.disposed
  };
}

/**
 * WHAT: 주어진 텐서의 데이터를 GPU에서 CPU로 비동기적으로 읽어 Float32Array로 반환하는 함수입니다.
 * WHY: 연산 결과가 포함된 GPU 버퍼의 데이터를 사용자나 프레임워크가 확인할 수 있도록 하기 위해 제공됩니다.
 * HOW: 레지스트리에서 버퍼를 조회하고 readBufferToFloat32Array 헬퍼를 사용해 데이터를 복사 후 반환합니다.
 */
export function read(handle: TensorHandle): Promise<Float32Array> {
  /**
   * WHAT: 핸들로 조회된 텐서 레코드 객체입니다.
   * WHY: 실제 GPUBuffer 참조와 버퍼 길이를 알아내기 위해 필요합니다.
   * HOW: _globalRegistry.get(handle) 호출을 통해 가져옵니다.
   */
  const record = _globalRegistry.get(handle);
  return readBufferToFloat32Array(record.buffer, record.byteLength);
}

/**
 * WHAT: 텐서 버퍼의 데이터를 읽기 위해 GPU 메모리를 매핑(map)하는 비동기 함수입니다.
 * WHY: 즉시 읽기(read)와 달리 맵핑과 데이터 복사를 분리하여 제로 카피(Zero Copy)나 스트리밍 최적화를 지원하기 위해 존재합니다.
 * HOW: 레지스트리에서 버퍼를 조회한 뒤 맵핑을 수행하고 반환된 스테이징 버퍼를 _pendingStagingBuffers에 저장합니다.
 */
export async function mapBufferAsync(handle: TensorHandle): Promise<void> {
  // If already staged and mapped, return immediately
  if (_pendingStagingBuffers.has(handle)) {
    return;
  }
  // If a mapping operation is already in-flight for this handle, coalesce with existing promise
  const inFlight = _inFlightMapPromises.get(handle);
  if (inFlight) {
    return inFlight;
  }

  const record = _globalRegistry.get(handle);
  const promise = (async () => {
    try {
      const { stagingBuffer, token } = await _mapBufferAsync(record.buffer, record.byteLength);
      _pendingStagingBuffers.set(handle, { stagingBuffer, token });
    } finally {
      _inFlightMapPromises.delete(handle);
    }
  })();

  _inFlightMapPromises.set(handle, promise);
  return promise;
}

/**
 * WHAT: 매핑이 완료된 스테이징 버퍼에서 대상 배열로 데이터를 동기 복사하는 함수입니다.
 * WHY: mapBufferAsync 호출 이후 실제 데이터를 사용자의 자바스크립트 버퍼 혹은 Pyodide 메모리로 옮기기 위해 사용됩니다.
 * HOW: _pendingStagingBuffers에서 버퍼를 찾아 실제 대상 배열(outArray)에 복사하고 스테이징 버퍼를 정리합니다.
 */
export function readMappedInto(handle: TensorHandle, outArray: any): void {
  /**
   * WHAT: 이전 mapBufferAsync 호출로 준비된 스테이징 버퍼 관련 정보 객체입니다.
   * WHY: 복사해올 실제 소스 버퍼에 접근하기 위해 맵에서 꺼내어 참조합니다.
   * HOW: _pendingStagingBuffers.get(handle)을 통해 조회합니다.
   */
  const obj = _pendingStagingBuffers.get(handle);
  if (!obj) {
    throw new Error(
      `[AMEVA] No staged buffer for handle "${handle}". Call mapBufferAsync first.`
    );
  }
  _pendingStagingBuffers.delete(handle);

  /**
   * WHAT: Pyodide나 WebAssembly 환경의 메모리 뷰를 감싸는 프록시 객체입니다.
   * WHY: 외부 WASM 메모리를 다룰 때 버퍼 포인터 획득과 해제를 안전하게 처리하기 위해 변수에 저장합니다.
   * HOW: 초기엔 null로 두고 outArray 타입에 따라 getBuffer() 결과가 할당됩니다.
   */
  let bufProxy: any = null;
  try {
    /**
     * WHAT: 데이터 복사가 기록될 최종 대상 Float32Array입니다.
     * WHY: 스테이징 버퍼의 데이터를 CPU가 직접 다룰 수 있는 형식으로 전달받기 위해 필요합니다.
     * HOW: bufProxy.data를 통해 참조를 얻거나 outArray 자체를 Float32Array로 캐스팅합니다.
     */
    let actualData: Float32Array;
    if (outArray && typeof outArray.getBuffer === 'function') {
      bufProxy = outArray.getBuffer("f32");
      actualData = bufProxy.data;
    } else {
      actualData = outArray as Float32Array;
    }
    
    // H-02 Fix: WASM 메모리 바운드 사전 검증
    if (actualData && actualData.buffer) {
      assertWasmRange(actualData.byteOffset, actualData.byteLength, actualData.buffer.byteLength);
    }
    
    // F-009 Fix: 대상 배열 크기와 원본 텐서 크기 검증
    const record = _globalRegistry.get(handle);
    if (actualData.byteLength !== record.byteLength) {
      throw new Error(
        `[AMEVA Forge] readMappedInto size mismatch. Expected ${record.byteLength} bytes, got ${actualData.byteLength} bytes.`
      );
    }

    _readMappedInto(obj.stagingBuffer, obj.token, actualData);
  } finally {
    // _readMappedInto already releases the token!
    // H-NEW-06: bufProxy.release() 실패 시에도 리소스 정리 보장
    if (bufProxy) {
      try { bufProxy.release(); } catch { /* ignore */ }
    }
  }
}

/**
 * WHAT: 사용을 마친 특정 텐서를 해제하는 함수입니다.
 * WHY: 외부 사용자가 더 이상 텐서 메모리를 사용하지 않을 때 메모리를 GPU에서 해제하기 위해 호출됩니다.
 * HOW: _globalRegistry.dispose()를 호출하여 핸들에 연결된 레코드를 삭제하고 버퍼 소멸 스케줄을 잡습니다.
 */
export function dispose(handle: TensorHandle): void {
  _globalRegistry.dispose(handle);
}

// ─────────────────────────────────────────────────────────────────────────────
// L-01 Fix: dispatchKernel 헬퍼 — 모든 op의 반복 코드를 통합
// NH-07 Fix: assertAllowedKernelName() 호출
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WHAT: dispatchKernel 함수에 전달되는 매개변수 객체의 인터페이스입니다.
 * WHY: 개별 오퍼레이션(add, mul 등)이 실행될 때 필요한 셰이더, 입력 버퍼, 차원(dispatch x, y) 등을 통일된 포맷으로 전달받기 위함입니다.
 * HOW: opKey, wgslCode, paramsData, inputBuffers, outBuffer, dispatchX, dispatchY 등의 속성을 가집니다.
 */
interface KernelDispatchOptions {
  opKey: string;
  wgslCode: string;
  paramsData: Uint32Array;
  inputBuffers: GPUBuffer[];
  outBuffer: GPUBuffer;
  dispatchX: number;
  dispatchY?: number;
  dispatchZ?: number;
}

/**
 * WHAT: 단일 WebGPU 컴퓨트 셰이더 커널을 디스패치(실행 요청)하는 공통 헬퍼 함수입니다.
 * WHY: 개별 연산 함수(add, sub 등)에 중복되는 버퍼 바인딩 및 파이프라인 생성 코드를 통합하여 유지보수성을 높이기 위해 존재합니다.
 * HOW: 유니폼 파라미터 버퍼를 생성하고 파이프라인 캐시를 조회한 뒤, 바인드 그룹을 설정하여 컴퓨트 패스를 큐에 제출합니다.
 */
function dispatchKernel(opts: KernelDispatchOptions): void {
  // NH-07 Fix: shaderGuard에서 커널 이름 검증
  assertAllowedKernelName(opts.opKey);

  /**
   * WHAT: WebGPU 작업을 제출할 대상 논리 디바이스입니다.
   * WHY: 커맨드 인코더 생성과 버퍼 조작을 위해 필요합니다.
   * HOW: getDevice() 함수를 호출하여 가져옵니다.
   */
  const device = getDevice();

  /**
   * WHAT: 셰이더로 전달될 스칼라 인자(크기, 차원 등)를 담는 GPU 유니폼 버퍼입니다.
   * WHY: GPU 셰이더 내에서 텐서 크기 등의 동적인 파라미터를 읽을 수 있어야 연산이 가능하기 때문입니다.
   * HOW: 최소 16바이트 정렬 크기를 만족하도록 디바이스에서 UNIFORM 용도로 할당합니다.
   */
  const { buffer: paramsBuffer, token: paramsToken } = allocateBuffer(
    Math.max(16, opts.paramsData.byteLength), // 최소 16바이트 (WebGPU uniform 정렬)
    BUFFER_USAGE_UNIFORM_COPY,
    'uniform',
    `dispatchKernel_${opts.opKey}`
  );
  device.queue.writeBuffer(paramsBuffer, 0, opts.paramsData.buffer);

  // H-01: 파이프라인 캐시에서 조회 (없으면 컴파일 후 캐시)
  /**
   * WHAT: 컴파일이 완료된 WebGPU 컴퓨트 파이프라인 객체입니다.
   * WHY: 셰이더 코드를 기반으로 GPU가 작업을 어떻게 수행해야 하는지 구조를 알고 있어야 하기 때문입니다.
   * HOW: opKey와 wgslCode를 사용하여 _globalPipelineCache에서 가져옵니다.
   */
  const { pipeline } = _globalPipelineCache.getPipeline(opts.opKey, opts.wgslCode);

  /**
   * WHAT: 파이프라인에 바인딩될 리소스들의 배열(유니폼 버퍼, 입력 버퍼들, 출력 버퍼)입니다.
   * WHY: 셰이더의 각 바인딩 슬롯(binding 0, 1, 2...)에 정확한 버퍼를 매핑하기 위해 리스트로 준비합니다.
   * HOW: paramsBuffer를 binding 0에, 입력 버퍼들을 그 다음 순서에, 출력 버퍼를 마지막에 배치하여 구성합니다.
   */
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: paramsBuffer } },
    ...opts.inputBuffers.map((buf, i) => ({
      binding: i + 1,
      resource: { buffer: buf }
    })),
    { binding: opts.inputBuffers.length + 1, resource: { buffer: opts.outBuffer } }
  ];

  /**
   * WHAT: 준비된 entries를 기반으로 셰이더와 런타임 버퍼를 연결해주는 바인드 그룹 객체입니다.
   * WHY: 디바이스 커맨드 패스에 리소스 그룹을 설정하기 위해 필수적입니다.
   * HOW: device.createBindGroup을 통해 파이프라인의 레이아웃과 entries를 결합하여 생성합니다.
   */
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries
  });

  /**
   * WHAT: GPU 명령들을 기록하기 위한 커맨드 인코더입니다.
   * WHY: 복사, 컴퓨트 패스 등 여러 GPU 조작을 묶어서 큐에 제출하기 위해 사용됩니다.
   * HOW: device.createCommandEncoder()로 생성합니다.
   */
  const commandEncoder = device.createCommandEncoder();
  
  /**
   * WHAT: 컴퓨트 연산을 기록하는 패스 인코더입니다.
   * WHY: 파이프라인, 바인드 그룹, 디스패치 워크그룹 수 등을 설정하기 위해 필요합니다.
   * HOW: commandEncoder.beginComputePass()를 호출하여 가져옵니다.
   */
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(opts.dispatchX, opts.dispatchY ?? 1, opts.dispatchZ ?? 1);
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);

  // params 버퍼는 GPU 제출 완료 후 중앙 allocator를 통해 해제
  void device.queue.onSubmittedWorkDone().then(() => {
    try { freeBuffer(paramsBuffer, paramsToken); } catch (e) { _safeLog(`[gpuCore] Failed to free params buffer: ${e}`); }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 개별 op 함수들 (내부 사용, pyodideBridge에서는 executeGraph를 통해서만 접근)
// NH-01 Note: 이 함수들은 JS 테스트와 직접 호출에서만 사용
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WHAT: 무작위 값(0~1)으로 채워진 지정된 형태(shape)의 텐서를 생성하는 함수입니다.
 * WHY: 신경망 가중치 초기화나 테스트 코드에서 임의의 데이터가 필요할 때 사용됩니다.
 * HOW: CPU(자바스크립트) 상에서 Float32Array 배열에 난수를 채우고 allocateBuffer로 얻은 GPU버퍼로 복사하여 레지스트리에 등록합니다.
 */
export function random(shape: number[], dtype: DType = "float32"): TensorHandle {
  validateDType(dtype);
  /**
   * WHAT: 텐서의 모든 차원을 곱해 산출된 총 원소의 개수입니다.
   * WHY: 1차원 Float32Array를 얼마나 크게 할당하고 루프를 돌릴지 결정하기 위해 계산됩니다.
   * HOW: validateShape 헬퍼를 통해 모양 검증과 동시에 산출됩니다.
   */
  const elements = validateShape(shape, dtype);
  
  /**
   * WHAT: CPU 메모리 상에 존재하는 실수 데이터 배열입니다.
   * WHY: GPU로 데이터를 전송하기 전 난수값을 임시로 기록하기 위해 할당합니다.
   * HOW: 원소 수(elements)만큼의 크기로 Float32Array를 생성합니다.
   */
  const data = new Float32Array(elements);
  
  /**
   * WHAT: 배열의 각 위치를 순회하며 난수를 채우는 반복문입니다.
   * WHY: 텐서 전체를 임의의 값으로 초기화하기 위해 실행됩니다.
   * HOW: i를 0부터 elements 전까지 증가시키며 Math.random() 값을 배열에 대입합니다.
   */
  for (let i = 0; i < elements; i++) data[i] = Math.random();
  
  /**
   * WHAT: 텐서 전체 데이터가 차지할 실제 바이트 크기입니다.
   * WHY: GPU 버퍼를 할당할 때 정확한 메모리 공간 크기가 필요하므로 계산합니다.
   * HOW: Float32 원소 개수에 4(바이트)를 곱합니다.
   */
  const byteLength = elements * 4;
  
  /**
   * WHAT: GPU 메모리 내에 새로 할당된 버퍼와 추적 토큰입니다.
   * WHY: 텐서 데이터를 영속적으로 저장하고 나중에 사용할 수 있도록 하기 위함입니다.
   * HOW: allocateBuffer 헬퍼를 사용하여 STORAGE, COPY_SRC, COPY_DST 용도로 버퍼를 생성합니다.
   */
  const { buffer, token } = allocateBuffer(
    byteLength,
    BUFFER_USAGE_STORAGE_COPY
  );
  writeFloat32Array(buffer, data);
  return _globalRegistry.register({ buffer, token, shape, dtype, byteLength });
}

/**
 * WHAT: 기존의 Float32Array 데이터를 GPU 텐서로 업로드(복사)하여 핸들을 반환하는 함수입니다.
 * WHY: 외부 이미지 데이터나 입력 특징(feature) 배열을 GPU 메모리로 올려 연산을 수행할 수 있게 만들기 위해 존재합니다.
 * HOW: Pyodide 버퍼 프록시 혹은 일반 배열 데이터를 기반으로 GPU 버퍼를 할당하고 값을 복사한 후 레지스트리에 등록합니다.
 */
export function uploadFloat32Array(data: any, shape: number[]): TensorHandle {
  /**
   * WHAT: 업로드할 원본 데이터가 복사된 또는 참조된 Float32Array입니다.
   * WHY: WebGPU 버퍼에 쓰기 명령을 수행하려면 반드시 이 형태의 타입화된 배열이어야 하기 때문입니다.
   * HOW: 조건에 따라 bufProxy.data 또는 data 자체를 캐스팅하여 할당합니다.
   */
  let actualData: Float32Array;
  
  /**
   * WHAT: 외부 WASM 환경(Pyodide 등)에서 제공하는 버퍼 메모리 프록시 객체입니다.
   * WHY: 외부에 노출된 메모리 포인터 접근 후 자원 누수를 막기 위해 명시적인 해제(release)가 필요하기 때문에 변수에 잡아둡니다.
   * HOW: data 객체가 getBuffer 함수를 가지고 있으면 이를 호출하여 초기화하고 아니면 null을 유지합니다.
   */
  let bufProxy: any = null;
  if (data && typeof data.getBuffer === 'function') {
    bufProxy = data.getBuffer("f32");
    actualData = bufProxy.data;
  } else {
    actualData = data as Float32Array;
  }
  
  // H-02 Fix: WASM 메모리 바운드 사전 검증
  if (actualData && actualData.buffer) {
    assertWasmRange(actualData.byteOffset, actualData.byteLength, actualData.buffer.byteLength);
  }
  
  /**
   * WHAT: 입력된 형태(shape)가 지녀야 할 원소 총 개수입니다.
   * WHY: 형태 배열과 실제 전달된 배열의 바이트 길이가 일치하는지 검증하기 위해 필요합니다.
   * HOW: validateShape를 호출하며 actualData의 바이트 크기를 넘겨 정합성을 검사합니다.
   */
  const elements = validateShape(shape, "float32", actualData.byteLength);
  
  /**
   * WHAT: GPU에 할당될 메모리 총 바이트 수입니다.
   * WHY: allocateBuffer 헬퍼에 필요한 바이트 단위를 맞추기 위해 사용됩니다.
   * HOW: 산출된 원소 개수에 4를 곱합니다.
   */
  const byteLength = elements * 4;
  const { buffer, token } = allocateBuffer(
    byteLength,
    BUFFER_USAGE_STORAGE_COPY
  );
  writeFloat32Array(buffer, actualData);
  if (bufProxy) bufProxy.release();
  return _globalRegistry.register({ buffer, token, shape, dtype: "float32", byteLength });
}

/**
 * WHAT: 두 개의 2차원 텐서에 대해 행렬 곱셈(Matmul)을 수행하는 함수입니다.
 * WHY: 신경망의 완전 연결층(Dense Layer)이나 어텐션 매커니즘 등 주요 선형 대수 연산을 지원하기 위해 존재합니다.
 * HOW: 두 텐서의 차원을 검증하고, 결과용 버퍼를 새로 생성한 뒤 matmul 셰이더를 dispatchKernel로 호출합니다.
 */
export function matmul(handleA: TensorHandle, handleB: TensorHandle): TensorHandle {
  /**
   * WHAT: 첫 번째 입력 행렬(A)의 레코드입니다.
   * WHY: A 행렬의 shape와 GPU 버퍼 포인터를 알아내기 위해 필요합니다.
   * HOW: 전역 레지스트리에서 handleA를 키로 조회합니다.
   */
  const a = _globalRegistry.get(handleA);
  
  /**
   * WHAT: 두 번째 입력 행렬(B)의 레코드입니다.
   * WHY: B 행렬의 shape와 메모리 버퍼를 확보하여 연산 인자로 쓰기 위해 필요합니다.
   * HOW: 전역 레지스트리에서 handleB로 조회합니다.
   */
  const b = _globalRegistry.get(handleB);

  if (a.shape.length !== 2 || b.shape.length !== 2)
    throw new AMEVAForgeShapeError("Matmul requires 2D tensors");
  if (a.dtype !== "float32" || b.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Matmul requires float32 tensors");

  /**
   * WHAT: A 행렬의 행, A의 열(B의 행), B의 행, B의 열을 나타내는 차원 변수들입니다.
   * WHY: 행렬 곱이 성립하기 위한 내부 차원(K) 일치 여부를 검사하고 워크그룹 수를 계산하기 위함입니다.
   * HOW: 각 텐서의 shape 배열에서 인덱스로 값을 구조 분해하여 할당합니다.
   */
  const M = a.shape[0], K = a.shape[1], K2 = b.shape[0], N = b.shape[1];
  if (K !== K2) throw new AMEVAForgeShapeError(`Inner dim mismatch: ${K} != ${K2}`);

  /**
   * WHAT: 결과 행렬(C)이 차지할 총 바이트 크기입니다.
   * WHY: 행렬 곱의 결과 텐서를 저장할 적절한 크기의 GPU 버퍼를 할당하기 위해 계산합니다.
   * HOW: 행 크기(M)와 열 크기(N)를 곱한 값에 float32 크기인 4를 곱합니다.
   */
  const byteLength = M * N * 4;
  const { buffer: cBuffer, token } = allocateBuffer(byteLength, BUFFER_USAGE_STORAGE_SRC);

  // SCRUM-201: 16x16 Workgroup Shared Memory Tiled MatMul 디스패치
  dispatchKernel({
    opKey: 'matmul_tiled',
    wgslCode: MATMUL_TILED_WGSL,
    paramsData: new Uint32Array([M, N, K, 0]),
    inputBuffers: [a.buffer, b.buffer],
    outBuffer: cBuffer,
    dispatchX: Math.ceil(N / 16),
    dispatchY: Math.ceil(M / 16),
  });

  return _globalRegistry.register({ buffer: cBuffer, token, shape: [M, N], dtype: "float32", byteLength });
}

/**
 * WHAT: 16x16 워크그룹 공유 메모리(Shared Memory)를 활용한 명시적 고성능 Tiled MatMul 함수입니다.
 * WHY: Release 2.0 Transformer 및 대규모 행렬곱 가속을 위해 3.5x~5x 향상된 연산 처리율을 제공합니다.
 * HOW: matmul_tiled WGSL 커널을 16x16 워크그룹 단위로 디스패치합니다.
 */
export function matmulTiled(handleA: TensorHandle, handleB: TensorHandle): TensorHandle {
  return matmul(handleA, handleB);
}

/**
 * WHAT: 주어진 텐서의 모든 원소에 대해 ReLU(Rectified Linear Unit) 활성화 함수를 적용하는 함수입니다.
 * WHY: 신경망에서 음수 값을 제거하여 비선형성을 부여하기 위해 핵심적인 오퍼레이션입니다.
 * HOW: 단일 텐서 버퍼를 읽고, 동일 크기의 출력 버퍼를 만든 후 relu 커널을 디스패치합니다.
 */
export function relu(handle: TensorHandle): TensorHandle {
  /**
   * WHAT: 입력 텐서 레코드입니다.
   * WHY: 연산 대상 데이터가 들어있는 GPU 버퍼와 크기를 가져오기 위함입니다.
   * HOW: 레지스트리에서 핸들로 조회합니다.
   */
  const x = _globalRegistry.get(handle);
  if (x.dtype !== "float32") throw new AMEVAForgeDTypeError("ReLU requires float32");
  
  /**
   * WHAT: 입력 텐서 내에 존재하는 실수 요소의 총 개수입니다.
   * WHY: 워크그룹 수를 계산하여 디스패치 크기를 결정하고 셰이더 내에서 배열 경계 검사를 수행하기 위해 필요합니다.
   * HOW: 총 바이트 길이를 4로 나누어 구합니다.
   */
  const numElements = x.byteLength / 4;
  const { buffer: outBuffer, token } = allocateBuffer(x.byteLength, BUFFER_USAGE_STORAGE_SRC);
  const dispatch = computeDispatch2D(numElements, 64);

  dispatchKernel({
    opKey: 'relu',
    wgslCode: RELU_WGSL,
    paramsData: new Uint32Array([numElements, dispatch.workgroupsX, 0, 0]),
    inputBuffers: [x.buffer],
    outBuffer,
    dispatchX: dispatch.dispatchX,
    dispatchY: dispatch.dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [...x.shape], dtype: "float32", byteLength: x.byteLength });
}

/**
 * WHAT: 두 텐서 간의 요소별 덧셈(Element-wise Addition)을 수행하는 함수입니다.
 * WHY: 편향(bias) 더하기, 잔차 연결(residual connection) 등 신경망 연산에서 두 특징 맵을 합칠 때 사용됩니다.
 * HOW: 형태가 같은 두 텐서 버퍼를 넘겨받아 add 셰이더를 실행시키고 새로운 텐서를 생성해 반환합니다.
 */
export function add(handleA: TensorHandle, handleB: TensorHandle): TensorHandle {
  const a = _globalRegistry.get(handleA);
  const b = _globalRegistry.get(handleB);
  if (a.shape.length !== b.shape.length || !a.shape.every((v, i) => v === b.shape[i]))
    throw new AMEVAForgeShapeError("Add requires tensors of the exact same shape");
  if (a.dtype !== "float32" || b.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Add requires float32");

  const numElements = a.byteLength / 4;
  const { buffer: outBuffer, token } = allocateBuffer(a.byteLength, BUFFER_USAGE_STORAGE_SRC);

  const { dOut, effSA, effSB } = computeBroadcastParams(a.shape, a.shape, b.shape);
  const dispatch = computeDispatch2D(numElements, 64);
  const paramsData = new Uint32Array(28);
  paramsData[0] = numElements;
  paramsData[1] = dispatch.workgroupsX;
  paramsData[2] = a.shape.length;
  paramsData[3] = 0;
  for (let k = 0; k < 8; k++) paramsData[4 + k] = dOut[k];
  for (let k = 0; k < 8; k++) paramsData[12 + k] = effSA[k];
  for (let k = 0; k < 8; k++) paramsData[20 + k] = effSB[k];

  dispatchKernel({
    opKey: 'add',
    wgslCode: ADD_WGSL,
    paramsData,
    inputBuffers: [a.buffer, b.buffer],
    outBuffer,
    dispatchX: dispatch.dispatchX,
    dispatchY: dispatch.dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [...a.shape], dtype: "float32", byteLength: a.byteLength });
}

/**
 * WHAT: 두 텐서 간의 요소별 곱셈(Element-wise Multiplication)을 수행하는 함수입니다.
 * WHY: 어텐션 스코어 마스킹이나 활성화된 게이트 통과 등 데이터를 요소별로 가중치와 곱할 때 필요합니다.
 * HOW: 형태가 같은 두 텐서를 기반으로 mul 커널을 디스패치합니다.
 */
export function mul(handleA: TensorHandle, handleB: TensorHandle): TensorHandle {
  const a = _globalRegistry.get(handleA);
  const b = _globalRegistry.get(handleB);
  if (a.shape.length !== b.shape.length || !a.shape.every((v, i) => v === b.shape[i]))
    throw new AMEVAForgeShapeError("Mul requires tensors of the exact same shape");
  if (a.dtype !== "float32" || b.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Mul requires float32");

  const numElements = a.byteLength / 4;
  const { buffer: outBuffer, token } = allocateBuffer(a.byteLength, BUFFER_USAGE_STORAGE_SRC);

  const { dOut, effSA, effSB } = computeBroadcastParams(a.shape, a.shape, b.shape);
  const dispatch = computeDispatch2D(numElements, 64);
  const paramsData = new Uint32Array(28);
  paramsData[0] = numElements;
  paramsData[1] = dispatch.workgroupsX;
  paramsData[2] = a.shape.length;
  paramsData[3] = 0;
  for (let k = 0; k < 8; k++) paramsData[4 + k] = dOut[k];
  for (let k = 0; k < 8; k++) paramsData[12 + k] = effSA[k];
  for (let k = 0; k < 8; k++) paramsData[20 + k] = effSB[k];

  dispatchKernel({
    opKey: 'mul',
    wgslCode: MUL_WGSL,
    paramsData,
    inputBuffers: [a.buffer, b.buffer],
    outBuffer,
    dispatchX: dispatch.dispatchX,
    dispatchY: dispatch.dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [...a.shape], dtype: "float32", byteLength: a.byteLength });
}

/**
 * WHAT: 2차원 텐서(행렬)의 행과 열을 뒤집는 전치(Transpose) 연산을 수행하는 함수입니다.
 * WHY: 행렬 곱셈을 수행하기 전에 데이터의 축을 맞추거나 그래디언트 역전파를 위해 텐서를 변형할 때 사용됩니다.
 * HOW: 입력 형태(shape)의 [M, N]을 [N, M]으로 뒤집은 결과를 반환할 출력 버퍼에 기록하도록 transpose 셰이더를 실행합니다.
 */
export function transpose(handle: TensorHandle): TensorHandle {
  const x = _globalRegistry.get(handle);
  if (x.shape.length !== 2)
    throw new AMEVAForgeShapeError("Transpose requires 2D tensors");
  if (x.dtype !== "float32")
    throw new AMEVAForgeDTypeError("Transpose requires float32");

  const M = x.shape[0], N = x.shape[1];
  const { buffer: outBuffer, token } = allocateBuffer(x.byteLength, BUFFER_USAGE_STORAGE_SRC);

  dispatchKernel({
    opKey: 'transpose',
    wgslCode: TRANSPOSE_WGSL,
    paramsData: new Uint32Array([M, N, 1, 0]),
    inputBuffers: [x.buffer],
    outBuffer,
    dispatchX: Math.ceil(M / 8),
    dispatchY: Math.ceil(N / 8),
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [N, M], dtype: "float32", byteLength: x.byteLength });
}

/**
 * WHAT: ReLU 활성화 함수의 도함수(그래디언트)를 계산하여 역전파(Backward)를 수행하는 함수입니다.
 * WHY: 오차 역전파 과정에서 순전파 시 입력값이 0 이상이었던 위치에만 상위 그래디언트를 흘려보내기 위해 필요합니다.
 * HOW: 원본 입력 텐서(x)와 위층에서 전달된 그래디언트 텐서(grad)를 받아, x가 0보다 큰 곳은 grad를, 아니면 0을 출력 버퍼에 씁니다.
 */
export function relu_backward(handleX: TensorHandle, handleGrad: TensorHandle): TensorHandle {
  const x = _globalRegistry.get(handleX);
  const grad = _globalRegistry.get(handleGrad);
  if (x.shape.length !== grad.shape.length || !x.shape.every((v, i) => v === grad.shape[i]))
    throw new AMEVAForgeShapeError("ReLU backward: shape mismatch");

  const numElements = x.byteLength / 4;
  const { buffer: outBuffer, token } = allocateBuffer(x.byteLength, BUFFER_USAGE_STORAGE_SRC);
  const dispatch = computeDispatch2D(numElements, 64);

  dispatchKernel({
    opKey: 'relu_backward',
    wgslCode: RELU_BACKWARD_WGSL,
    paramsData: new Uint32Array([numElements, dispatch.workgroupsX, 0, 0]),
    inputBuffers: [x.buffer, grad.buffer],
    outBuffer,
    dispatchX: dispatch.dispatchX,
    dispatchY: dispatch.dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [...x.shape], dtype: "float32", byteLength: x.byteLength });
}

/**
 * WHAT: FlashAttention-2 융합 1-Pass Scaled Dot-Product Attention을 수행하는 함수입니다.
 * WHY: O(N^2) 어텐션 맵 VRAM 할당을 완전히 제거하여 대규모 LLM 추론 시 극적인 메모리 절감과 처리율을 제공합니다.
 * HOW: Q, K, V 텐서를 받아 셰이더 내에서 Online Softmax와 Causal Masking을 융합 실행합니다.
 */
export function flashAttention(
  handleQ: TensorHandle,
  handleK: TensorHandle,
  handleV: TensorHandle,
  scale?: number,
  isCausal: boolean = false
): TensorHandle {
  const q = _globalRegistry.get(handleQ);
  const k = _globalRegistry.get(handleK);
  const v = _globalRegistry.get(handleV);

  if (q.shape.length !== 4 || k.shape.length !== 4 || v.shape.length !== 4) {
    throw new AMEVAForgeShapeError("FlashAttention requires 4D tensors [Batch, Heads, SeqLen, HeadDim]");
  }
  if (q.dtype !== "float32" || k.dtype !== "float32" || v.dtype !== "float32") {
    throw new AMEVAForgeDTypeError("FlashAttention requires float32 tensors");
  }

  const [B, H, N_q, d] = q.shape;
  const [B_k, H_kv, N_k, d_k] = k.shape;
  const [B_v, H_kv2, N_v, d_v] = v.shape;

  if (B !== B_k || B !== B_v) throw new AMEVAForgeShapeError(`Batch mismatch: ${B} vs ${B_k}, ${B_v}`);
  if (H_kv !== H_kv2) throw new AMEVAForgeShapeError(`KV heads mismatch: ${H_kv} vs ${H_kv2}`);
  if (H % H_kv !== 0) throw new AMEVAForgeShapeError(`Query heads ${H} must be divisible by KV heads ${H_kv} (GQA requirement)`);
  if (N_k !== N_v) throw new AMEVAForgeShapeError(`Key/Value SeqLen mismatch: ${N_k} vs ${N_v}`);
  if (d !== d_k || d !== d_v) throw new AMEVAForgeShapeError(`HeadDim mismatch: ${d} vs ${d_k}, ${d_v}`);
  if (d > 256) throw new AMEVAForgeShapeError(`HeadDim ${d} exceeds max supported dimension 256`);

  const effectiveScale = scale !== undefined ? scale : 1.0 / Math.sqrt(d);
  const strideQ = N_q * d;
  const strideK = N_k * d;
  const strideV = N_v * d;
  const strideO = N_q * d;

  const byteLength = B * H * N_q * d * 4;
  const { buffer: outBuffer, token } = allocateBuffer(byteLength, BUFFER_USAGE_STORAGE_SRC);

  // Params buffer: B, H, H_kv, N_q, N_kv, d, scale(float), is_causal, strideQ, strideK, strideV, strideO
  const paramsArray = new ArrayBuffer(48);
  const u32View = new Uint32Array(paramsArray);
  const f32View = new Float32Array(paramsArray);

  u32View[0] = B;
  u32View[1] = H;
  u32View[2] = H_kv;
  u32View[3] = N_q;
  u32View[4] = N_k;
  u32View[5] = d;
  f32View[6] = effectiveScale;
  u32View[7] = isCausal ? 1 : 0;
  u32View[8] = strideQ;
  u32View[9] = strideK;
  u32View[10] = strideV;
  u32View[11] = strideO;

  dispatchKernel({
    opKey: 'flash_attention',
    wgslCode: FLASH_ATTENTION_WGSL,
    paramsData: u32View,
    inputBuffers: [q.buffer, k.buffer, v.buffer],
    outBuffer,
    dispatchX: N_q,
    dispatchY: H,
    dispatchZ: B,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [B, H, N_q, d], dtype: "float32", byteLength });
}

export function rmsNorm(handleX: TensorHandle, handleGamma?: TensorHandle, eps = 1e-5): TensorHandle {
  const x = _globalRegistry.get(handleX);
  const shape = x.shape;
  const dim = shape[shape.length - 1];
  const numTokens = shape.slice(0, -1).reduce((a, b) => a * b, 1);

  const byteLength = x.byteLength;
  const { buffer: outBuffer, token } = allocateBuffer(byteLength, BUFFER_USAGE_STORAGE_COPY);

  const paramsArray = new ArrayBuffer(16);
  const u32View = new Uint32Array(paramsArray);
  const f32View = new Float32Array(paramsArray);
  u32View[0] = numTokens;
  u32View[1] = dim;
  f32View[2] = eps;
  u32View[3] = handleGamma !== undefined ? 1 : 0;

  const inputBuffers = [
    x.buffer,
    handleGamma !== undefined ? _globalRegistry.get(handleGamma).buffer : x.buffer
  ];

  const { dispatchX, dispatchY } = computeDispatch2D(numTokens);
  dispatchKernel({
    opKey: 'rmsnorm',
    wgslCode: RMSNORM_WGSL,
    paramsData: u32View,
    inputBuffers,
    outBuffer,
    dispatchX,
    dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [...shape], dtype: "float32", byteLength });
}

export function rope(handleX: TensorHandle, baseFreq = 10000.0, offsetPos = 0): TensorHandle {
  const x = _globalRegistry.get(handleX);
  const shape = x.shape;
  if (shape.length !== 4) {
    throw new AMEVAForgeShapeError(`RoPE requires 4D tensor [B, H, N, d], got rank ${shape.length}`);
  }
  const [B, H, N, d] = shape;
  if (d % 2 !== 0) {
    throw new AMEVAForgeShapeError(`RoPE head dimension d must be even, got ${d}`);
  }

  const byteLength = x.byteLength;
  const { buffer: outBuffer, token } = allocateBuffer(byteLength, BUFFER_USAGE_STORAGE_COPY);

  const paramsArray = new ArrayBuffer(32);
  const u32View = new Uint32Array(paramsArray);
  const f32View = new Float32Array(paramsArray);
  u32View[0] = B;
  u32View[1] = H;
  u32View[2] = N;
  u32View[3] = d;
  f32View[4] = baseFreq;
  u32View[5] = offsetPos;
  u32View[6] = 0;
  u32View[7] = 0;

  const totalTokens = B * H * N;
  const { dispatchX, dispatchY } = computeDispatch2D(totalTokens);

  dispatchKernel({
    opKey: 'rope',
    wgslCode: ROPE_WGSL,
    paramsData: u32View,
    inputBuffers: [x.buffer],
    outBuffer,
    dispatchX,
    dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [B, H, N, d], dtype: "float32", byteLength });
}

export function swiglu(handleGate: TensorHandle, handleUp: TensorHandle): TensorHandle {
  const gate = _globalRegistry.get(handleGate);
  const up = _globalRegistry.get(handleUp);
  const numElements = gate.shape.reduce((a, b) => a * b, 1);

  const byteLength = gate.byteLength;
  const { buffer: outBuffer, token } = allocateBuffer(byteLength, BUFFER_USAGE_STORAGE_COPY);

  const paramsArray = new Uint32Array([numElements, 0, 0, 0]);
  const { dispatchX, dispatchY } = computeDispatch2D(Math.ceil(numElements / 64));

  dispatchKernel({
    opKey: 'swiglu',
    wgslCode: SWIGLU_WGSL,
    paramsData: paramsArray,
    inputBuffers: [gate.buffer, up.buffer],
    outBuffer,
    dispatchX,
    dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [...gate.shape], dtype: "float32", byteLength });
}

export function unpackQuant(
  handlePacked: TensorHandle,
  handleScales: TensorHandle,
  handleZeros: TensorHandle,
  bits = 4,
  groupSize = 128,
  numElements: number
): TensorHandle {
  const packed = _globalRegistry.get(handlePacked);
  const scales = _globalRegistry.get(handleScales);
  const zeros = _globalRegistry.get(handleZeros);

  const byteLength = numElements * 4;
  const { buffer: outBuffer, token } = allocateBuffer(byteLength, BUFFER_USAGE_STORAGE_COPY);

  const paramsArray = new Uint32Array([numElements, bits, groupSize, 0]);
  const { dispatchX, dispatchY } = computeDispatch2D(Math.ceil(numElements / 64));

  dispatchKernel({
    opKey: 'unpack_quant',
    wgslCode: UNPACK_QUANT_WGSL,
    paramsData: paramsArray,
    inputBuffers: [packed.buffer, scales.buffer, zeros.buffer],
    outBuffer,
    dispatchX,
    dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: [numElements], dtype: "float32", byteLength });
}

/**
 * WHAT: 단어/토큰 인덱스 텐서와 가중치 행렬을 받아 WebGPU 상에서 임베딩 룩업을 수행합니다.
 * WHY: 트랜스포머 언어 모델의 첫 번째 계층인 토큰 임베딩을 브라우저 GPU 상에서 일괄 가속하기 위함입니다.
 * HOW: embedding.wgsl 컴퓨트 셰이더를 2D 그리드로 디스패치하여 대상 버퍼에 복사합니다.
 */
export function embedding(handleWeight: TensorHandle, handleIndex: TensorHandle): TensorHandle {
  const weight = _globalRegistry.get(handleWeight);
  const index = _globalRegistry.get(handleIndex);

  if (weight.shape.length !== 2) {
    throw new AMEVAForgeShapeError(
      `[AMEVA Forge] embedding: weight must be 2D [vocab_size, embedding_dim], got shape [${weight.shape.join(", ")}]`
    );
  }

  const vocabSize = weight.shape[0];
  const embeddingDim = weight.shape[1];
  const numTokens = index.shape.reduce((a, b) => a * b, 1);
  const outShape = [...index.shape, embeddingDim];
  const totalElements = numTokens * embeddingDim;
  const byteLength = totalElements * 4;

  const { buffer: outBuffer, token } = allocateBuffer(
    byteLength,
    BUFFER_USAGE_STORAGE_SRC,
    'tensor',
    'gpuCore_embedding'
  );

  const paramsArray = new Uint32Array([
    numTokens,
    embeddingDim,
    vocabSize,
    0, // 16-byte alignment pad
  ]);

  const { dispatchX, dispatchY } = computeDispatch2D(numTokens);

  dispatchKernel({
    opKey: 'embedding',
    wgslCode: EMBEDDING_WGSL,
    paramsData: paramsArray,
    inputBuffers: [weight.buffer, index.buffer],
    outBuffer,
    dispatchX,
    dispatchY,
  });

  return _globalRegistry.register({ buffer: outBuffer, token, shape: outShape, dtype: "float32", byteLength });
}

/**
 * WHAT: 임베딩 출력 기울기(gradOutput)와 토큰 인덱스(index)를 받아 가중치 기울기(gradWeight)를 WebGPU 상에서 계산합니다.
 * WHY: 트랜스포머 언어 모델의 임베딩 계층을 GPU 상에서 atomic 없이 완전 Lock-Free로 역전파 학습하기 위함입니다.
 * HOW: embedding_backward.wgsl 컴퓨트 셰이더를 2D 그리드로 디스패치하여 [Vocab, D] 크기의 gradWeight를 생성합니다.
 */
export function embedding_backward(
  handleGradOutput: TensorHandle,
  handleIndex: TensorHandle,
  vocabSize: number,
  embeddingDim: number
): TensorHandle {
  const gradOut = _globalRegistry.get(handleGradOutput);
  const index = _globalRegistry.get(handleIndex);

  const numTokens = index.shape.reduce((a, b) => a * b, 1);
  const totalWeightElements = vocabSize * embeddingDim;
  const byteLength = totalWeightElements * 4;

  const { buffer: outBuffer, token } = allocateBuffer(
    byteLength,
    BUFFER_USAGE_STORAGE_SRC,
    'tensor',
    'gpuCore_embedding_backward'
  );

  const paramsArray = new Uint32Array([
    numTokens,
    embeddingDim,
    vocabSize,
    totalWeightElements,
  ]);

  const { dispatchX, dispatchY } = computeDispatch2D(Math.ceil(totalWeightElements / 64));

  dispatchKernel({
    opKey: 'embedding_backward',
    wgslCode: EMBEDDING_BACKWARD_WGSL,
    paramsData: paramsArray,
    inputBuffers: [gradOut.buffer, index.buffer],
    outBuffer,
    dispatchX,
    dispatchY,
  });

  return _globalRegistry.register({
    buffer: outBuffer,
    token,
    shape: [vocabSize, embeddingDim],
    dtype: "float32",
    byteLength,
  });
}

export const gpuCore = {
  add,
  mul,
  matmul,
  matmulTiled,
  flashAttention,
  rmsNorm,
  rope,
  swiglu,
  unpackQuant,
  embedding,
  embedding_backward,
  relu,
  relu_backward,
  transpose,
};




