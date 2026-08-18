/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:59:35+09:00: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 *   - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * graphExecutor.ts — JSON 그래프 파서 & GPU 스케줄러
 *
 * C-04 Fix: JSON 입력에 대한 강력한 검증 추가
 * M-05 Fix: matmul dispatch X/Y swap 수정
 * H-01 Fix: _globalPipelineCache를 모든 op에 적용
 * NC-06 Fix: inst.in null-guard 추가 (! 비null 단언 제거)
 * NH-07 Fix: shaderGuard.assertAllowedKernelName() 실제 호출
 * NM-05 Fix: device.pushErrorScope()로 op별 에러 감지
 */

import { getDevice } from "../webgpu/device";
import { _globalRegistry, TensorRegistry } from "./tensorRegistry";
import { TensorHandle, DType } from "../types";
import { allocateBuffer, writeFloat32Array, freeBuffer } from "../webgpu/buffers";
import { _globalQuotaManager, AllocationToken } from "../webgpu/quota";
import { AMEVAForgeShapeError, AMEVAForgeSecurityError, AMEVAForgeUnsupportedOpError, AMEVAForgeValidationError, AMEVAForgeOutOfMemoryError, AMEVAForgeInternalGPUError } from "../errors";
import { assertAllowedKernelName } from "../webgpu/shaderGuard";
import { assertWasmRange } from "../webgpu/validateWasmRange";
import { _globalPipelineCache } from "../webgpu/pipelineCache";
import { computeBroadcastParams } from "./broadcastParams";
import { computeDispatch2D } from "./dispatchShape";
import { _globalUniformPool, UniformEntry } from "../webgpu/uniformPool";


// kernels
import { MATMUL_WGSL } from "./kernels/matmul.wgsl";
import { MATMUL_BIAS_RELU_WGSL } from "./kernels/matmul_bias_relu.wgsl";
import { BATCHED_MATMUL_WGSL } from "./kernels/batched_matmul.wgsl";
import { RELU_WGSL } from "./kernels/relu.wgsl";
import { ADD_WGSL } from "./kernels/add.wgsl";
import { MUL_WGSL } from "./kernels/mul.wgsl";
import { TRANSPOSE_WGSL } from "./kernels/transpose.wgsl";
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
import { MAXPOOL2D_WGSL } from "./kernels/maxpool2d.wgsl";
import { AVGPOOL2D_WGSL } from "./kernels/avgpool2d.wgsl";
import { IM2COL_WGSL } from "./kernels/im2col.wgsl";
import { COL2IM_WGSL } from "./kernels/col2im.wgsl";
import { PAD_WGSL } from "./kernels/pad.wgsl";
import { GATHER_WGSL } from "./kernels/gather.wgsl";
import { SCATTER_WGSL } from "./kernels/scatter.wgsl";
import { CAT_WGSL } from "./kernels/cat.wgsl";
import { WHERE_WGSL } from "./kernels/where.wgsl";
import { DROPOUT_WGSL } from "./kernels/dropout.wgsl";
import { PERMUTE_WGSL } from "./kernels/permute.wgsl";
import { MATMUL_TILED_WGSL } from "./kernels/matmul_tiled.wgsl";
import { FLASH_ATTENTION_WGSL } from "./kernels/flash_attention.wgsl";
import { ROPE_WGSL } from "./kernels/rope.wgsl";
import { RMSNORM_WGSL } from "./kernels/rmsnorm.wgsl";
import { SWIGLU_WGSL } from "./kernels/swiglu.wgsl";
import { UNPACK_QUANT_WGSL } from "./kernels/unpack_quant.wgsl";

/** 
 * WHAT: 그래프 실행기가 처리할 수 있는 모든 허용된 오퍼레이션(op)의 집합입니다.
 * WHY: 악의적인 JSON 그래프가 알 수 없거나 금지된 셰이더를 실행하여 GPU를 공격하는 것을 방지하기 위한 화이트리스트입니다.
 * HOW: Set 자료구조에 허용되는 오퍼레이션 문자열을 초기화하여 빠른 조회(O(1))를 제공합니다.
 */
const ALLOWED_OPS = new Set([
  'upload', 'load', 'matmul', 'matmul_tiled', 'batched_matmul', 'relu', 'add', 'mul', 'transpose', 'relu_backward',
  'sub', 'neg', 'div', 'exp', 'log', 'sigmoid', 'tanh', 'sigmoid_backward', 'tanh_backward',
  'fill', 'sum', 'max', 'sum_axis', 'max_axis', 'max_axis_backward', 'axpy', 'cat', 'where', 'pad', 'gather', 'scatter', 'maxpool2d', 'avgpool2d',
  'im2col', 'col2im', 'dropout', 'permute', 'matmul_bias_relu', 'reshape',
  'flash_attention', 'rope', 'rmsnorm', 'swiglu', 'unpack_quant'
]);

export type ForgeRuntimeConfig = {
  workloadBudgetElements?: number;
  maxOpsPerSubmit?: number;
  maxShapeDim?: number;
  maxElements?: number;
  maxInstructions?: number;
  allowNonFinite?: boolean;
};

const DEFAULT_RUNTIME_CONFIG: Required<ForgeRuntimeConfig> = {
  workloadBudgetElements: 100_000_000,
  maxOpsPerSubmit: 256,
  maxShapeDim: 8,
  maxElements: 256 * 1024 * 1024,
  maxInstructions: 10_000,
  allowNonFinite: false,
};

let _runtimeConfig = { ...DEFAULT_RUNTIME_CONFIG };

export function configureRuntime(config: ForgeRuntimeConfig): void {
  _runtimeConfig = {
    ..._runtimeConfig,
    ...config,
  };
}

export function getRuntimeConfig(): Required<ForgeRuntimeConfig> {
  return { ..._runtimeConfig };
}

const BUFFER_USAGE_STORAGE_COPY = typeof GPUBufferUsage !== 'undefined'
  ? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST)
  : (0x0080 | 0x0004 | 0x0008);

const BUFFER_USAGE_UNIFORM_COPY = typeof GPUBufferUsage !== 'undefined'
  ? (GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
  : (0x0040 | 0x0008);

const BUFFER_USAGE_STORAGE_SRC = typeof GPUBufferUsage !== 'undefined'
  ? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
  : (0x0080 | 0x0004);

/**
 * WHAT: 단일 텐서 연산을 지시하는 그래프 명령어의 데이터 타입 인터페이스입니다.
 * WHY: JSON 형태의 무타입 입력 데이터를 검증하고, 이후의 컴파일 과정에서 정적 타입 체크를 하기 위해 존재합니다.
 * HOW: 연산 종류(op), 식별자(id), 차원(shape), 입력 배열(in), 파라미터(params) 등의 속성을 정의합니다.
 */
interface GraphInstruction {
  op: string;
  id: number;
  shape: number[];
  in?: number[];
  handle?: string;
  params?: number[];
}

export interface PendingTensorRecord {
  handle: TensorHandle;
  buffer: GPUBuffer;
  token: AllocationToken;
  shape: number[];
  dtype: DType;
  byteLength: number;
}

function _safeLog(msg: string): void {
  try {
    if (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'development') {
      console.warn(msg);
    }
  } catch { /* intentionally empty: _safeLog is the outermost logging fallback, catching here prevents infinite recursion */ }
}

interface DeferredBufferRecord {
  buffer: GPUBuffer;
  token: AllocationToken;
  retries: number;
}

const _deferredGCQueue: DeferredBufferRecord[] = [];

/**
 * WHAT: 롤백 과정에서 즉시 destroy에 실패한 GPU 버퍼들의 지연 해제를 재시도합니다.
 * WHY: 일시적 GPU busy 상태 등으로 파괴 실패 시 유령 VRAM 누수를 방지합니다.
 */
export function processDeferredGC(): void {
  for (let i = _deferredGCQueue.length - 1; i >= 0; i--) {
    const item = _deferredGCQueue[i];
    try {
      item.buffer.destroy();
      _globalQuotaManager.releaseToken(item.token);
      _deferredGCQueue.splice(i, 1);
    } catch (e) {
      item.retries++;
      if (item.retries >= 3) {
        try {
          _globalQuotaManager.markOrphaned(item.token, String(e));
        } catch (err) { _safeLog(`[DeferredGC] markOrphaned failed: ${err}`); }
        _deferredGCQueue.splice(i, 1);
        _safeLog(`[DeferredGC] Failed to destroy buffer after 3 attempts, token marked orphaned: ${item.token.id}`);
      }
    }
  }
  
  if (_deferredGCQueue.length > 100) {
    _safeLog(`[DeferredGC] WARNING: ${_deferredGCQueue.length} items still pending after flush`);
  }
}

export class GraphTransaction {
  private readonly pending = new Map<TensorHandle, PendingTensorRecord>();

  add(record: PendingTensorRecord): void {
    if (this.pending.has(record.handle)) {
      throw new AMEVAForgeValidationError(`Duplicate pending handle: ${record.handle}`);
    }
    this.pending.set(record.handle, record);
  }

  get(handle: TensorHandle): PendingTensorRecord | undefined {
    return this.pending.get(handle);
  }

  get handles(): TensorHandle[] {
    return Array.from(this.pending.keys());
  }

  commit(registry: TensorRegistry): void {
    for (const record of this.pending.values()) {
      registry.registerRecord(record);
    }
    this.pending.clear();
  }

  rollback(): void {
    for (const record of this.pending.values()) {
      try {
        record.buffer.destroy();
        _globalQuotaManager.releaseToken(record.token);
      } catch (e) {
        _safeLog(`[GraphTransaction.rollback] Buffer destroy failed, queued for deferred GC: ${e}`);
        _deferredGCQueue.push({
          buffer: record.buffer,
          token: record.token,
          retries: 0
        });
      }
    }
    this.pending.clear();
    processDeferredGC();
  }
}


/**
 * WHAT: JSON에서 파싱된 단일 명령어 객체의 무결성을 엄격하게 검증하는 함수입니다.
 * WHY: 타입 오류나 범위 초과 등을 가진 악성 데이터가 하위 WebGPU 계층으로 흘러가 충돌을 일으키지 않도록 방어하기 위함입니다.
 * HOW: 속성의 존재 유무와 타입, 배열 길이, 연산의 결과 오버플로우 등을 꼼꼼하게 검사합니다.
 */
function validateInstruction(inst: unknown, idx: number): GraphInstruction {
  if (typeof inst !== 'object' || inst === null) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: must be an object`);
  }

  /**
   * WHAT: 타입 캐스팅을 위해 임시로 생성된 레코드 변수입니다.
   * WHY: unknown 타입을 Record<string, unknown>으로 변환하여 속성에 동적으로 접근하기 위해 필요합니다.
   * HOW: inst를 타입 단언(as)으로 캐스팅합니다.
   */
  const i = inst as Record<string, unknown>;

  if (typeof i.op !== 'string') {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: op must be a string`);
  }
  if (!ALLOWED_OPS.has(i.op)) {
    throw new AMEVAForgeUnsupportedOpError(`Instruction[${idx}]: unknown op "${i.op}"`);
  }

  if (!Number.isSafeInteger(i.id) || (i.id as number) < 1) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: id must be a positive safe integer`);
  }

  if (!Array.isArray(i.shape)) {
    throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape must be an array`);
  }
  // NM-06: rank 0 허용 (스칼라)
  if (i.shape.length > _runtimeConfig.maxShapeDim) {
    throw new AMEVAForgeShapeError(
      `Instruction[${idx}]: shape rank must be 0–${_runtimeConfig.maxShapeDim}, got ${i.shape.length}`
    );
  }

  /**
   * WHAT: 해당 명령어 텐서의 누적 원소 수를 계산하는 변수입니다.
   * WHY: 차원의 곱이 안전한 정수 범위를 넘거나 최대 한계(_runtimeConfig.maxElements)를 초과하는지 확인하기 위해 계산합니다.
   * HOW: 루프를 통해 차원(dim)을 곱하여 누적합니다. 초기값은 스칼라 연산을 위해 1로 시작합니다.
   */
  let elements = 1;
  
  /**
   * WHAT: shape 배열의 각 차원에 대해 안전성을 검사하는 루프입니다.
   * WHY: 음수 차원, 부동소수점 차원, 정수 오버플로우로 인한 악의적 크기 공격을 차단하기 위해 순회합니다.
   * HOW: for...of 구문으로 각 차원(dim)을 검사하고 elements 변수에 곱합니다.
   */
  for (const dim of i.shape) {
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new AMEVAForgeShapeError(
        `Instruction[${idx}]: shape dim must be a positive safe integer, got ${dim}`
      );
    }
    if (dim > Number.MAX_SAFE_INTEGER / elements) {
      throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape product integer overflow`);
    }
    elements *= dim;
  }
  if (elements > _runtimeConfig.maxElements) {
    throw new AMEVAForgeShapeError(
      `Instruction[${idx}]: tensor too large (${elements} elements > ${_runtimeConfig.maxElements})`
    );
  }

  // NC-06: in 필드가 있으면 배열인지 확인
  if (i.in !== undefined && !Array.isArray(i.in)) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: 'in' field must be an array`);
  }
  if (i.params !== undefined && !Array.isArray(i.params)) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: 'params' field must be an array`);
  }
  
  // F-017 Fix: 각 커널별 엄격한 스키마 검증 (in 개수 및 params 길이 강제)
  const OP_SCHEMA: Record<string, { minIn: number, exactIn?: boolean, minParams: number, exactParams?: boolean }> = {
    'upload': { minIn: 0, exactIn: true, minParams: 0, exactParams: true },
    'load': { minIn: 0, exactIn: true, minParams: 0, exactParams: true },
    'fill': { minIn: 0, exactIn: true, minParams: 2, exactParams: true },
    'sum': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'max': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'relu': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'exp': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'log': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'sigmoid': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'tanh': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'neg': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
    'relu_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: false },
    'sigmoid_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'tanh_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'pad': { minIn: 1, exactIn: true, minParams: 9, exactParams: false }, // pad는 최대 8차원 144바이트 = 36 uint32s.
    'sum_axis': { minIn: 1, exactIn: true, minParams: 2, exactParams: false },
    'max_axis': { minIn: 1, exactIn: true, minParams: 2, exactParams: false },
    'max_axis_backward': { minIn: 2, exactIn: true, minParams: 2, exactParams: false },
    'dropout': { minIn: 1, exactIn: true, minParams: 2, exactParams: true },
    'maxpool2d': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'avgpool2d': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'im2col': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'col2im': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'transpose': { minIn: 1, exactIn: true, minParams: 2, exactParams: false },
    'permute': { minIn: 1, exactIn: true, minParams: 1, exactParams: false }, // rank 길이 가변
    'reshape': { minIn: 1, exactIn: true, minParams: 0, exactParams: false },
    'add': { minIn: 2, exactIn: true, minParams: 0, exactParams: false },
    'sub': { minIn: 2, exactIn: true, minParams: 0, exactParams: false },
    'mul': { minIn: 2, exactIn: true, minParams: 0, exactParams: false },
    'div': { minIn: 2, exactIn: true, minParams: 0, exactParams: false },
    'axpy': { minIn: 2, exactIn: true, minParams: 2, exactParams: false },
    'gather': { minIn: 2, exactIn: true, minParams: 7, exactParams: false },
    'scatter': { minIn: 2, exactIn: false, minParams: 7, exactParams: false },
    'matmul': { minIn: 2, exactIn: true, minParams: 3, exactParams: true },
    'matmul_bias_relu': { minIn: 3, exactIn: true, minParams: 3, exactParams: true },
    'batched_matmul': { minIn: 2, exactIn: true, minParams: 4, exactParams: false },
    'where': { minIn: 3, exactIn: true, minParams: 0, exactParams: false },
    'cat': { minIn: 2, exactIn: false, minParams: 1, exactParams: false } // 가변 개수 입력, params는 axis 등
  };

  const opStr = i.op as string;
  const schema = OP_SCHEMA[opStr];
  if (schema) {
    const inLen = i.in ? (i.in as unknown[]).length : 0;
    const pLen = i.params ? (i.params as unknown[]).length : 0;
    
    if (schema.exactIn && inLen !== schema.minIn) {
      throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected exact ${schema.minIn} inputs, got ${inLen}`);
    } else if (inLen < schema.minIn) {
      throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected min ${schema.minIn} inputs, got ${inLen}`);
    }
    
    if (schema.exactParams && pLen !== schema.minParams) {
      throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected exact ${schema.minParams} params, got ${pLen}`);
    } else if (pLen < schema.minParams) {
      throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected min ${schema.minParams} params, got ${pLen}`);
    }
  }

  // params 타입 검증 (전부 안전한 number 이어야 함)
  if (i.params) {
    for (const p of i.params as unknown[]) {
      if (typeof p !== 'number' || !Number.isFinite(p)) {
        throw new AMEVAForgeSecurityError(`Instruction[${idx}]: param must be a finite number`);
      }
    }
  }

  return i as unknown as GraphInstruction;
}

/**
 * executeGraph — Python 레이지 그래프를 단일 FFI 호출로 GPU에 실행한다.
 * WHAT: Python 등 외부 환경에서 직렬화된 연산 그래프(JSON)를 받아 일괄적으로 GPU에서 실행하는 함수입니다.
 * WHY: 매 연산마다 JS와 WebAssembly/GPU 사이를 왕복(context switch)하면 극심한 오버헤드가 발생하므로, 한 번의 호출로 많은 명령을 처리(Transaction)하기 위해 설계되었습니다.
 * HOW: JSON을 파싱하고, 명령을 검증하며, 적절한 청크로 분할하여 WebGPU 커맨드 버퍼에 기록하고 제출(submit)합니다. 실패 시 트랜잭션을 롤백합니다.
 */
let _executionQueueChain: Promise<any> = Promise.resolve();

export async function executeGraph(
  instructionsJson: string,
  inputs: (Float32Array | any)[],
  outputIds?: number[]
): Promise<Record<string, TensorHandle>> {
  const previous = _executionQueueChain;
  let releaseLock: () => void;
  _executionQueueChain = new Promise<void>((resolve) => {
    releaseLock = resolve;
  });

  try {
    await previous;
  } catch {
    // Suppress previous transaction error so queue continues processing
  }

  try {
    return await _executeGraphCore(instructionsJson, inputs);
  } finally {
    releaseLock!();
  }
}

async function _executeGraphCore(
  instructionsJson: string,
  jsInputs: unknown
): Promise<Record<number, TensorHandle>> {
  // Flush any pending deferred GC items before new execution
  processDeferredGC();

  // ── 1. Parse ──
  let rawInstructions: unknown[];
  try {
    rawInstructions = JSON.parse(instructionsJson, (key, value) => {
      // M-01 Fix: JSON Prototype Pollution 방어
      if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
        throw new AMEVAForgeSecurityError(`Forbidden property name in JSON: ${key}`);
      }
      return value;
    });
  } catch (e) {
    if (e instanceof AMEVAForgeSecurityError) throw e;
    throw new AMEVAForgeSecurityError("executeGraph: invalid JSON in instructionsJson");
  }

  if (!Array.isArray(rawInstructions)) {
    throw new AMEVAForgeSecurityError("executeGraph: instructionsJson must be a JSON array");
  }
  if (rawInstructions.length > _runtimeConfig.maxInstructions) {
    throw new AMEVAForgeSecurityError(
      `executeGraph: too many instructions (${rawInstructions.length} > ${_runtimeConfig.maxInstructions})`
    );
  }

  // ── 2. Validate ──
  const instructions: GraphInstruction[] = rawInstructions.map(validateInstruction);

  // VULN-06: Ensure AXPY is only executed in the optimizer commit phase and not followed by downstream ops
  let seenAxpy = false;
  for (const inst of instructions) {
    if (inst.op === 'axpy') {
      seenAxpy = true;
    } else if (seenAxpy) {
      throw new AMEVAForgeSecurityError(
        `Invalid graph execution: In-place 'axpy' is an optimizer commit phase operation and cannot be followed by downstream op '${inst.op}' in the same transaction.`
      );
    }
  }

  let inputs: unknown[];
  if (jsInputs && typeof (jsInputs as any).toJs === 'function') {
    inputs = (jsInputs as any).toJs();
  } else if (Array.isArray(jsInputs)) {
    inputs = jsInputs;
  } else {
    inputs = [];
  }

  // ── 3. Plan ──
  // (In the future: calculate peak memory, check dependency DAG)
  
  // ── 4. Execute ──
  const device = getDevice();
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('internal');

  let commandEncoder = device.createCommandEncoder();
  let opsInCurrentBatch = 0;
  let encoderHasCommands = false;
  let workloadElements = 0;
  
  const idToHandle: Record<number, TensorHandle> = {};
  const idToBuffer: Record<number, GPUBuffer> = {};
  const idToByteLength: Record<number, number> = {};
  const idToShape: Record<number, number[]> = {};
  const transaction = new GraphTransaction();
  let inputIdx = 0;
  
  const paramsAllocations: Array<{ buffer: GPUBuffer, token: AllocationToken, isUniformPool?: boolean, uniformEntry?: UniformEntry }> = [];

  try {
    /**
     * WHAT: 검증된 각 그래프 명령어를 순차적으로 순회하며 GPU 작업으로 변환하는 메인 루프입니다.
     * WHY: 계획된 그래프 연산들을 실제 WebGPU 파이프라인 디스패치로 번역하기 위해 반드시 실행해야 합니다.
     * HOW: for...of 구문을 사용하여 instructions 배열의 각 객체(inst)를 처리합니다.
     */
    for (const inst of instructions) {
      /**
       * WHAT: 현재 명령어가 결과로 생성할 텐서의 바이트 크기입니다.
       * WHY: 결과를 담을 출력 버퍼(OutBuffer)의 크기를 GPU에 요청할 때 필요합니다.
       * HOW: 배열 차원(shape)을 모두 곱한 뒤, float32 크기(4)를 곱하여 계산합니다.
       */
      const byteLength = inst.shape.reduce((a, b) => a * b, 1) * 4;

      if (inst.op === 'load') {
        /**
         * WHAT: load 명령에 전달된 기존 텐서의 핸들 문자열입니다.
         * WHY: 이미 VRAM에 존재하는 텐서를 그래프의 내부 ID에 매핑하여 입력으로 사용하기 위해 필요합니다.
         * HOW: inst.handle 속성을 읽어오고 유효성을 검증합니다.
         */
        const handle = inst.handle;
        if (typeof handle !== 'string') {
          throw new AMEVAForgeSecurityError(`load instruction missing handle`);
        }
        
        if (!_globalRegistry.has(handle)) {
          console.error(`[GraphExecutor DIAGNOSTIC] load op failed for handle="${handle}". Registered handles count=${_globalRegistry.snapshotHandles().length}, list=${JSON.stringify(_globalRegistry.snapshotHandles())}`);
        }
        const record = _globalRegistry.get(handle);
        // F-018 Fix: JSON 형상과 레지스트리 실제 형상 일치 여부 검사
        if (inst.shape.length !== record.shape.length || !inst.shape.every((v, i) => v === record.shape[i])) {
          throw new AMEVAForgeShapeError(`load instruction shape mismatch for handle ${handle}. Expected [${record.shape}], got [${inst.shape}]`);
        }
        
        idToHandle[inst.id] = handle;
        idToBuffer[inst.id] = record.buffer;
        idToByteLength[inst.id] = record.byteLength;
        idToShape[inst.id] = record.shape;
        continue;
      }

      if (inst.op === 'upload') {
        const rawData = inputs[inputIdx++];
        let actualData: Float32Array;
        let bufProxy: any = null;

        if (rawData && typeof (rawData as any).getBuffer === 'function') {
          bufProxy = (rawData as any).getBuffer("f32");
          actualData = bufProxy.data;
        } else if (rawData instanceof Float32Array) {
          actualData = rawData;
        } else if (rawData && typeof (rawData as any).toJs === 'function') {
          const converted = (rawData as any).toJs();
          actualData = converted instanceof Float32Array ? converted : new Float32Array(converted);
        } else {
          throw new AMEVAForgeSecurityError(`upload input[${inputIdx - 1}] is not a Float32Array`);
        }

        // H-02 Fix: WASM 메모리 바운드 및 Detached 버퍼 사전 검증
        if (actualData && actualData.buffer) {
          const buf = actualData.buffer as any;
          if (buf.detached === true || actualData.byteLength === 0) {
            if (bufProxy) bufProxy.release();
            throw new AMEVAForgeSecurityError(`upload input[${inputIdx - 1}] buffer is detached (WASM heap growth)`);
          }
          assertWasmRange(actualData.byteOffset, actualData.byteLength, actualData.buffer.byteLength);
        }

        // VULN-10: NaN / Inf fail-fast check (Strictly governed by trusted ForgeRuntimeConfig)
        const allowNonFinite = _runtimeConfig.allowNonFinite === true;
        for (let i = 0; i < actualData.length; i++) {
          if (!Number.isFinite(actualData[i])) {
            if (allowNonFinite) {
              _safeLog(`[GraphExecutor] Non-finite value in upload input[${inputIdx - 1}] allowed by runtime config`);
            } else {
              if (bufProxy) bufProxy.release();
              throw new AMEVAForgeValidationError(
                `Invalid tensor data: upload input[${inputIdx - 1}] contains NaN or Infinity at index ${i}. ` +
                `Configure runtime allowNonFinite=true to bypass if intended.`
              );
            }
          }
        }

        const { buffer, token } = allocateBuffer(
          byteLength,
          BUFFER_USAGE_STORAGE_COPY,
          'tensor',
          `Graph_${instructions[0]?.id}`
        );
        try {
          writeFloat32Array(buffer, actualData);
        } finally {
          if (bufProxy) bufProxy.release();
        }

        const uuid = typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).substring(2, 15);
        const handle = `tensor_${uuid}`;
        transaction.add({
          handle,
          buffer,
          token,
          shape: inst.shape,
          dtype: "float32",
          byteLength
        });
        idToHandle[inst.id] = handle;
        idToBuffer[inst.id] = buffer;
        idToByteLength[inst.id] = byteLength;
        idToShape[inst.id] = inst.shape;
        continue;
      }

      if (inst.op === 'reshape') {
        if (!inst.in || inst.in.length < 1) {
          throw new AMEVAForgeSecurityError(`reshape instruction missing 'in' tensor`);
        }
        const inBuf = idToBuffer[inst.in[0]];
        const inByteLength = idToByteLength[inst.in[0]];
        if (!inBuf) {
          throw new AMEVAForgeSecurityError(`reshape input tensor not found for id ${inst.in[0]}`);
        }
        if (inByteLength !== byteLength) {
          throw new AMEVAForgeShapeError(
            `reshape size mismatch: input has ${inByteLength / 4} elements, output has ${byteLength / 4} elements`
          );
        }

        const { buffer: outBuffer, token } = allocateBuffer(
          byteLength,
          BUFFER_USAGE_STORAGE_COPY,
          'tensor',
          `Graph_${instructions[0]?.id}`
        );
        commandEncoder.copyBufferToBuffer(inBuf, 0, outBuffer, 0, byteLength);
        encoderHasCommands = true;

        const uuid = typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).substring(2, 15);
        const handle = `tensor_${uuid}`;

        transaction.add({
          handle,
          buffer: outBuffer,
          token,
          shape: inst.shape,
          dtype: "float32",
          byteLength
        });

        idToHandle[inst.id] = handle;
        idToBuffer[inst.id] = outBuffer;
        idToByteLength[inst.id] = byteLength;
        idToShape[inst.id] = inst.shape;
        continue;
      }

      assertAllowedKernelName(inst.op);

      let outBuffer: GPUBuffer;
      if (inst.op === 'axpy') {
        if (!inst.in || inst.in.length < 2) {
          throw new AMEVAForgeSecurityError(`Instruction axpy is missing 'in' fields.`);
        }
        outBuffer = idToBuffer[inst.in[1]];
        idToHandle[inst.id] = idToHandle[inst.in[1]];
        idToBuffer[inst.id] = outBuffer;
        idToByteLength[inst.id] = byteLength;
        idToShape[inst.id] = inst.shape;
      } else {
        const { buffer, token } = allocateBuffer(
          byteLength,
          BUFFER_USAGE_STORAGE_COPY,
          'tensor',
          `Graph_${instructions[0]?.id}`
        );
        outBuffer = buffer;
        const uuid = typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).substring(2, 15);
        const handle = `tensor_${uuid}`;
        transaction.add({
          handle,
          buffer: outBuffer,
          token,
          shape: inst.shape,
          dtype: "float32",
          byteLength
        });
        idToHandle[inst.id] = handle;
        idToBuffer[inst.id] = outBuffer;
        idToByteLength[inst.id] = byteLength;
        idToShape[inst.id] = inst.shape;
      }

      /**
       * WHAT: 현재 오퍼레이션의 유니폼 파라미터를 담기 위해 필요한 바이트 크기입니다.
       * WHY: 오퍼레이션(패딩, 풀링 등)마다 셰이더가 요구하는 인자의 종류와 개수가 다르므로 가변적인 버퍼 크기를 잡기 위해 결정합니다.
       * HOW: inst.op 문자열을 판별하여 필요한 바이트 수(최소 32바이트)를 할당합니다.
       */
      let paramsSize = 32;
      if (inst.op === 'pad') paramsSize = 144;
      else if (inst.op === 'gather' || inst.op === 'scatter') paramsSize = 112;
      else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d') paramsSize = 64;
      else if (inst.op === 'im2col' || inst.op === 'col2im') paramsSize = 48;
      else if (inst.op === 'permute') paramsSize = 112;
      else if (['add', 'sub', 'mul', 'div'].includes(inst.op)) paramsSize = 112;

      const { buffer: paramsBuffer, token: paramsToken } = allocateBuffer(
        paramsSize,
        BUFFER_USAGE_UNIFORM_COPY,
        'uniform',
        `Graph_${instructions[0]?.id}_params`
      );
      paramsAllocations.push({ buffer: paramsBuffer, token: paramsToken });

      let wgslCode = "";
      let dispatchX = 1, dispatchY = 1, dispatchZ = 1;
      let isMatmul = false;
      let B = 1, M = 1, N = 1, K = 1;

      if (inst.op === 'matmul' || inst.op === 'matmul_bias_relu') {
        if (!inst.params || inst.params.length < 3) {
          throw new AMEVAForgeSecurityError(`${inst.op} instruction missing params`);
        }
        [M, N, K] = inst.params;
        const isFused = inst.op === 'matmul_bias_relu';
        const tileSize = isFused ? 16 : 8;
        wgslCode = isFused ? MATMUL_BIAS_RELU_WGSL : MATMUL_WGSL;
        isMatmul = true;
        // TS-H01 Fix: matmul X축도 65535 클램핑 — 초과분은 Z 차원으로 분산
        const rawDispatchX = Math.ceil(N / tileSize);
        if (rawDispatchX <= 65535) {
          dispatchX = rawDispatchX;
        } else {
          dispatchX = 65535;
          dispatchZ = Math.ceil(rawDispatchX / 65535);
        }
        const maxWorkgroupsM = Math.ceil(M / tileSize);
        if (maxWorkgroupsM > 65535) {
          throw new AMEVAForgeSecurityError(
            `Matmul M dimension (${M}) exceeds single-pass dispatch limit (${65535 * tileSize} rows). Partition tensor or reduce batch size.`
          );
        }
        dispatchY = maxWorkgroupsM;
      } else if (inst.op === 'batched_matmul') {
        if (!inst.params || inst.params.length < 4) {
          throw new AMEVAForgeSecurityError(`batched_matmul instruction missing params`);
        }
        const [B_param, N_param, P_param, M_param] = inst.params;
        B = B_param;
        wgslCode = BATCHED_MATMUL_WGSL;
        
        const rawDispatchX = Math.ceil(P_param / 16);
        if (rawDispatchX <= 65535) {
          dispatchX = rawDispatchX;
        } else {
          throw new AMEVAForgeSecurityError(`batched_matmul dispatchX exceeded limit: ${rawDispatchX}`);
        }
        
        const rawDispatchY = Math.ceil(N_param / 16);
        if (rawDispatchY <= 65535) {
          dispatchY = rawDispatchY;
        } else {
          throw new AMEVAForgeSecurityError(`batched_matmul dispatchY exceeded limit: ${rawDispatchY}`);
        }

        if (B <= 65535) {
          dispatchZ = B;
        } else {
          throw new AMEVAForgeSecurityError(`batched_matmul dispatchZ (Batch) exceeded limit: ${B}`);
        }
        
        const strideA = inst.params.length >= 7 ? inst.params[4] : N_param * M_param;
        const strideB = inst.params.length >= 7 ? inst.params[5] : M_param * P_param;
        const strideC = inst.params.length >= 7 ? inst.params[6] : N_param * P_param;
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([B_param, N_param, P_param, M_param, strideA, strideB, strideC, 0]));
      } else if (inst.op === 'transpose') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`transpose instruction missing params`);
        }
        const rM = inst.params[0];
        const rN = inst.params[1];
        const rB = inst.params.length >= 3 ? inst.params[2] : 1;
        wgslCode = TRANSPOSE_WGSL;
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([rM, rN, rB, 0]));
        dispatchX = Math.ceil(rM / 8);
        dispatchY = Math.ceil(rN / 8);
        dispatchZ = rB;
      } else if (inst.op === 'sum_axis' || inst.op === 'max_axis') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`${inst.op} instruction missing params`);
        }
        let outer_size = 1;
        let reduction_size = 1;
        let inner_stride = 1;
        if (inst.params.length >= 3) {
          [outer_size, reduction_size, inner_stride] = inst.params;
        } else {
          [reduction_size, inner_stride] = inst.params;
          outer_size = 1;
        }
        const output_numel = outer_size * inner_stride;
        wgslCode = inst.op === 'sum_axis' ? SUM_AXIS_WGSL : MAX_AXIS_WGSL;
        const totalWGs = Math.ceil(output_numel / 64);
        if (totalWGs <= 65535) {
          dispatchX = totalWGs;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWGs)));
          dispatchY = Math.min(65535, Math.ceil(totalWGs / dispatchX));
        }
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([outer_size, reduction_size, inner_stride, output_numel, dispatchX, 0, 0, 0]));
      } else if (inst.op === 'max_axis_backward') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`max_axis_backward instruction missing params`);
        }
        let outer_size = 1;
        let reduction_size = 1;
        let inner_stride = 1;
        if (inst.params.length >= 3) {
          [outer_size, reduction_size, inner_stride] = inst.params;
        } else {
          [reduction_size, inner_stride] = inst.params;
          outer_size = 1;
        }
        const input_numel = outer_size * reduction_size * inner_stride;
        wgslCode = MAX_AXIS_BACKWARD_WGSL;
        const totalWGs = Math.ceil(input_numel / 64);
        if (totalWGs <= 65535) {
          dispatchX = totalWGs;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWGs)));
          dispatchY = Math.min(65535, Math.ceil(totalWGs / dispatchX));
        }
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([outer_size, reduction_size, inner_stride, input_numel, dispatchX, 0, 0, 0]));
      } else if (inst.op === 'fill') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`fill instruction missing params`);
        }
        const numElements = inst.params[0];
        const fillValue = inst.params[1];
        wgslCode = FILL_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const f32arr = new Float32Array([0, fillValue, 0, 0]);
        const u32arr = new Uint32Array(f32arr.buffer);
        u32arr[0] = numElements;
        u32arr[2] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, u32arr);
      } else if (inst.op === 'axpy') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`axpy instruction missing params`);
        }
        const numElements = inst.params[0];
        const lr = inst.params[1];
        wgslCode = AXPY_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const f32arr = new Float32Array([0, lr, 0, 0]);
        const u32arr = new Uint32Array(f32arr.buffer);
        u32arr[0] = numElements;
        u32arr[2] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, u32arr);
      } else if (inst.op === 'pad') {
        const numElements = byteLength / 4;
        wgslCode = PAD_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const p = new Uint32Array(36);
        /**
         * WHAT: 패딩 옵션들을 유니폼 버퍼 배열에 복사하는 루프입니다.
         * WHY: 셰이더에서 사용될 스칼라 인자(정수 및 실수)를 메모리에 연속적으로 배치하기 위해 사용됩니다.
         * HOW: for 루프를 통해 inst.params 배열의 인자들을 p 배열로 옮기며, 실수형인 패딩 값은 Float32Array 뷰를 통해 씁니다.
         */
        for (let i = 0; i < inst.params!.length; i++) {
          if (i === 2) new Float32Array(p.buffer)[2] = inst.params![2];
          else p[i] = inst.params![i];
        }
        p[3] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'gather') {
        const numElements = byteLength / 4;
        wgslCode = GATHER_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const p = new Uint32Array(28);
        /**
         * WHAT: 파라미터를 복사하는 짧은 루프입니다.
         * WHY: gather 커널에 필요한 형태와 인덱싱 오프셋 정보들을 전송하기 위해 복사합니다.
         * HOW: 파라미터를 하나씩 Uint32Array에 대입합니다.
         */
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        p[3] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'scatter') {
        const numElements = inst.params![0];
        wgslCode = SCATTER_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const p = new Uint32Array(28);
        /**
         * WHAT: scatter 셰이더 인자를 복사하는 루프입니다.
         * WHY: 분산 배치할 인덱스 스텝 정보를 넘기기 위함입니다.
         * HOW: 파라미터를 하나씩 Uint32Array에 대입합니다.
         */
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        p[3] = dispatchX; // workgroups_x
        if (inst.params!.length < 28) {
          const shapeX = (inst.in && inst.in.length >= 3 && idToShape[inst.in[2]]) ? idToShape[inst.in[2]] : inst.shape;
          for (let i = 0; i < shapeX.length; i++) {
            p[20 + i] = shapeX[i];
          }
        }
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'dropout') {
        const numElements = byteLength / 4;
        const rawSeed = Number(inst.params![0]);
        const seed_u32 = (Number.isFinite(rawSeed) && rawSeed !== 0)
          ? (rawSeed >>> 0)
          : ((typeof crypto !== 'undefined' && crypto.getRandomValues)
              ? crypto.getRandomValues(new Uint32Array(1))[0]
              : (Math.floor(Math.random() * 0xFFFFFFFF) >>> 0));
        const p_val = inst.params![1];
        wgslCode = DROPOUT_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const buf = new ArrayBuffer(16);
        const u32view = new Uint32Array(buf);
        const f32view = new Float32Array(buf);
        u32view[0] = numElements;
        u32view[1] = seed_u32;
        f32view[2] = p_val;
        u32view[3] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, buf);
      } else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d') {
        const numElements = byteLength / 4;
        wgslCode = inst.op === 'maxpool2d' ? MAXPOOL2D_WGSL : AVGPOOL2D_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const p = new Uint32Array(16);
        /**
         * WHAT: 풀링 파라미터를 복사하는 루프입니다.
         * WHY: 윈도우 크기, 스트라이드, 패딩 등 컨볼루션 구조를 셰이더에 넘기기 위함입니다.
         * HOW: 요소별로 배열에 대입합니다.
         */
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        p[12] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'im2col' || inst.op === 'col2im') {
        const numElements = byteLength / 4;
        wgslCode = inst.op === 'im2col' ? IM2COL_WGSL : COL2IM_WGSL;

        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
          dispatchX = totalWorkgroups;
          dispatchY = 1;
        } else {
          dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
          dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        const p = new Uint32Array(12);
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        p[10] = dispatchX; // workgroups_x
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'permute') {
        const numElements = byteLength / 4;
        wgslCode = PERMUTE_WGSL;
        const dims = inst.params!;
        const rank = dims.length;
        
        const inHandle = idToHandle[inst.in![0]];
        const inShape = idToShape[inst.in![0]] ?? (_globalRegistry.has(inHandle) ? _globalRegistry.get(inHandle)!.shape : inst.shape);
        
        const inStrides = new Array(rank).fill(0);
        let s = 1;
        /**
         * WHAT: 입력 텐서의 각 차원별 메모리 보폭(stride)을 계산하는 역순 루프입니다.
         * WHY: 다차원 인덱스를 1차원 플랫 메모리 오프셋으로 변환할 때 곱해줄 가중치를 구하기 위해 필요합니다.
         * HOW: 가장 마지막 차원(우측)부터 시작하여 1부터 차례로 곱해나가며 배열을 채웁니다.
         */
        for (let i = rank - 1; i >= 0; i--) {
            inStrides[i] = s;
            s *= inShape[i];
        }
        
        const outStrides = new Array(rank).fill(0);
        let s2 = 1;
        /**
         * WHAT: 출력 텐서의 각 차원별 스트라이드를 계산하는 역순 루프입니다.
         * WHY: 출력을 기록할 1차원 주소를 생성할 때 사용될 가중치를 미리 연산해두기 위함입니다.
         * HOW: 마찬가지로 맨 우측 차원부터 누적하여 곱합니다.
         */
        for (let i = rank - 1; i >= 0; i--) {
            outStrides[i] = s2;
            s2 *= inst.shape[i];
        }
        
        const dispatch = computeDispatch2D(numElements, 64);
        dispatchX = dispatch.dispatchX;
        dispatchY = dispatch.dispatchY;

        const p = new Uint32Array(28);
        p[0] = rank;
        p[1] = numElements;
        p[2] = dispatch.workgroupsX;
        p[3] = 0;
        
        /**
         * WHAT: 계산된 각 차원들의 스트라이드와 형태 정보를 WebGPU vec4 정렬 규칙에 맞게 유니폼 버퍼 패딩 구조에 삽입하는 루프입니다.
         * WHY: GPU 셰이더 내에서 배열이나 벡터 형태로 데이터를 오차 없이 접근하기 위해 메모리 오프셋을 맞추어 기록합니다.
         * HOW: i를 0부터 rank 전까지 증가시키며 4개 단위 벡터 위치를 계산하여 씁니다.
         */
        for (let i = 0; i < rank; i++) {
           const vecOffset = i < 4 ? 4 + i : 8 + (i - 4);
           p[vecOffset] = inStrides[dims[i]];
           
           const outShapeOffset = i < 4 ? 12 + i : 16 + (i - 4);
           p[outShapeOffset] = inst.shape[i];
           
           const outStrideOffset = i < 4 ? 20 + i : 24 + (i - 4);
           p[outStrideOffset] = outStrides[i];
        }
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'flash_attention') {
        wgslCode = FLASH_ATTENTION_WGSL;
        const [B, H, N, d] = inst.shape;
        const H_kv = inst.params?.[0] ?? H;
        const scale = inst.params?.[1] ?? (1.0 / Math.sqrt(d));
        const isCausal = (inst.params?.[2] ?? 0) === 1 ? 1 : 0;
        
        const strideQ = N * d;
        const strideK = N * d;
        const strideV = N * d;
        const strideO = N * d;

        const buf = new ArrayBuffer(48);
        const u32view = new Uint32Array(buf);
        const f32view = new Float32Array(buf);
        u32view[0] = B;
        u32view[1] = H;
        u32view[2] = H_kv;
        u32view[3] = N;
        u32view[4] = d;
        f32view[5] = scale;
        u32view[6] = isCausal;
        u32view[7] = strideQ;
        u32view[8] = strideK;
        u32view[9] = strideV;
        u32view[10] = strideO;
        u32view[11] = 0;

        dispatchX = N;
        dispatchY = H;
        dispatchZ = B;
        device.queue.writeBuffer(paramsBuffer, 0, u32view);
      } else if (inst.op === 'rope') {
        wgslCode = ROPE_WGSL;
        const [B, H, N, d] = inst.shape;
        const baseFreq = inst.params?.[0] ?? 10000.0;
        const offsetPos = inst.params?.[1] ?? 0;

        const buf = new ArrayBuffer(32);
        const u32view = new Uint32Array(buf);
        const f32view = new Float32Array(buf);
        u32view[0] = B;
        u32view[1] = H;
        u32view[2] = N;
        u32view[3] = d;
        f32view[4] = baseFreq;
        u32view[5] = offsetPos;
        u32view[6] = 0;
        u32view[7] = 0;

        dispatchX = N;
        dispatchY = H;
        dispatchZ = B;
        device.queue.writeBuffer(paramsBuffer, 0, u32view);
      } else if (inst.op === 'rmsnorm') {
        wgslCode = RMSNORM_WGSL;
        const dim = inst.shape[inst.shape.length - 1];
        const numTokens = inst.shape.slice(0, -1).reduce((a, b) => a * b, 1);
        const eps = inst.params?.[0] ?? 1e-5;
        const hasGamma = (inst.in && inst.in.length >= 2) ? 1 : 0;

        const buf = new ArrayBuffer(16);
        const u32view = new Uint32Array(buf);
        const f32view = new Float32Array(buf);
        u32view[0] = numTokens;
        u32view[1] = dim;
        f32view[2] = eps;
        u32view[3] = hasGamma;

        const { dispatchX: dx, dispatchY: dy } = computeDispatch2D(numTokens);
        dispatchX = dx;
        dispatchY = dy;
        dispatchZ = 1;
        device.queue.writeBuffer(paramsBuffer, 0, u32view);
      } else if (inst.op === 'swiglu') {
        wgslCode = SWIGLU_WGSL;
        const numElements = inst.shape.reduce((a, b) => a * b, 1);
        const p = new Uint32Array([numElements, 0, 0, 0]);
        const { dispatchX: dx, dispatchY: dy } = computeDispatch2D(Math.ceil(numElements / 64));
        dispatchX = dx;
        dispatchY = dy;
        dispatchZ = 1;
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'unpack_quant') {
        wgslCode = UNPACK_QUANT_WGSL;
        const numElements = inst.shape.reduce((a, b) => a * b, 1);
        const bits = inst.params?.[0] ?? 4;
        const groupSize = inst.params?.[1] ?? 128;
        const p = new Uint32Array([numElements, bits, groupSize, 0]);
        const { dispatchX: dx, dispatchY: dy } = computeDispatch2D(Math.ceil(numElements / 64));
        dispatchX = dx;
        dispatchY = dy;
        dispatchZ = 1;
        device.queue.writeBuffer(paramsBuffer, 0, p);
      } else if (inst.op === 'sum' || inst.op === 'max') {
        // Handled entirely dynamically below, but we need to bypass normal flow
        wgslCode = inst.op === 'sum' ? SUM_WGSL : MAX_WGSL;
      } else {
        const numElements = byteLength / 4;
        wgslCode = inst.op === 'relu'          ? RELU_WGSL :
                   inst.op === 'add'           ? ADD_WGSL :
                   inst.op === 'mul'           ? MUL_WGSL :
                   inst.op === 'sub'           ? SUB_WGSL :
                   inst.op === 'neg'           ? NEG_WGSL :
                   inst.op === 'div'           ? DIV_WGSL :
                   inst.op === 'relu_backward' ? RELU_BACKWARD_WGSL :
                   inst.op === 'exp'           ? EXP_WGSL :
                   inst.op === 'log'           ? LOG_WGSL :
                   inst.op === 'sigmoid'       ? SIGMOID_WGSL :
                   inst.op === 'tanh'          ? TANH_WGSL :
                   inst.op === 'sigmoid_backward' ? SIGMOID_BACKWARD_WGSL :
                   inst.op === 'tanh_backward' ? TANH_BACKWARD_WGSL : 
                   inst.op === 'cat'           ? CAT_WGSL :
                   inst.op === 'where'         ? WHERE_WGSL : 
                   inst.op === 'dropout'       ? DROPOUT_WGSL : '';

        if (!wgslCode) {
          throw new AMEVAForgeSecurityError(`Unknown op "${inst.op}"`);
        }
        
        const totalWorkgroups = Math.ceil(numElements / 64);
        // TS-C01 Fix: 65535 초과 시 2D 그리드로 분산
        if (totalWorkgroups <= 65535) {
            dispatchX = totalWorkgroups;
            dispatchY = 1;
        } else {
            // 2D 분산: sqrt로 균등 분할
            dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
            dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }

        if (['add', 'sub', 'mul', 'div'].includes(inst.op)) {
          let shapeA = [1];
          let shapeB = [1];
          if (inst.in && inst.in.length >= 2) {
            const in0Handle = idToHandle[inst.in[0]];
            const in1Handle = idToHandle[inst.in[1]];
            shapeA = idToShape[inst.in[0]] ?? (_globalRegistry.has(in0Handle) ? _globalRegistry.get(in0Handle).shape : [1]);
            shapeB = idToShape[inst.in[1]] ?? (_globalRegistry.has(in1Handle) ? _globalRegistry.get(in1Handle).shape : [1]);
          }
          const { dOut, effSA, effSB } = computeBroadcastParams(inst.shape, shapeA, shapeB);
          const p = new Uint32Array(28);
          p[0] = numElements;
          p[1] = dispatchX;
          p[2] = inst.shape.length;
          p[3] = 0;
          for (let k = 0; k < 8; k++) p[4 + k] = dOut[k];
          for (let k = 0; k < 8; k++) p[12 + k] = effSA[k];
          for (let k = 0; k < 8; k++) p[20 + k] = effSB[k];
          device.queue.writeBuffer(paramsBuffer, 0, p);
        } else {
          let numA = 0;
          let numB = 0;
          if (inst.in && inst.in.length >= 2) {
            numA = (idToByteLength[inst.in[0]] ?? byteLength) / 4;
            numB = (idToByteLength[inst.in[1]] ?? byteLength) / 4;
          }
          device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, dispatchX, numA, numB, 0, 0, 0, 0]));
        }

        if (inst.op === 'cat') {
          if (!inst.params || inst.params.length < 3) {
            throw new AMEVAForgeSecurityError(`cat instruction missing params`);
          }
          const [a_dim, b_dim, stride] = inst.params;
          // Overwrite the params for cat
          device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, dispatchX, a_dim, b_dim, stride, 0, 0, 0]));
        }
      }

      const { pipeline } = _globalPipelineCache.getPipeline(inst.op, wgslCode);

      if (inst.op === 'sum' || inst.op === 'max') {
          if (!inst.in || inst.in.length === 0) {
              throw new AMEVAForgeSecurityError(`Instruction op="${inst.op}" is missing 'in' field.`);
          }
          const REDUCTION_WG_SIZE = 256;
          const reductionInputHandle = idToHandle[inst.in[0]];
          if (!reductionInputHandle) throw new AMEVAForgeSecurityError(`Unresolved reduction input id ${inst.in[0]}`);
          
          let currentByteLength = idToByteLength[inst.in[0]];
          if (currentByteLength === undefined) {
            const rec = _globalRegistry.has(reductionInputHandle)
              ? _globalRegistry.get(reductionInputHandle)
              : transaction.get(reductionInputHandle);
            currentByteLength = rec ? rec.byteLength : 4;
          }
          let currentSize = currentByteLength / 4;
          let currentInputBuf = idToBuffer[inst.in[0]];
          const intermediateAllocations: Array<{ buffer: GPUBuffer, token: AllocationToken }> = [];
          
          while (currentSize > 1) {
              const numWGs = Math.ceil(currentSize / REDUCTION_WG_SIZE);
              let rDispatchX = 1;
              let rDispatchY = 1;
              if (numWGs <= 65535) {
                rDispatchX = numWGs;
                rDispatchY = 1;
              } else {
                rDispatchX = Math.min(65535, Math.ceil(Math.sqrt(numWGs)));
                rDispatchY = Math.min(65535, Math.ceil(numWGs / rDispatchX));
              }

              const { buffer: passBuf, token: passBufToken } = allocateBuffer(
                  Math.max(4, numWGs * 4),
                  BUFFER_USAGE_STORAGE_SRC,
                  'temporary',
                  `Graph_${instructions[0]?.id}_reduction`
              );
              intermediateAllocations.push({ buffer: passBuf, token: passBufToken });
              
              const { buffer: passParamsBuf, token: passParamsToken } = allocateBuffer(
                  16,
                  BUFFER_USAGE_UNIFORM_COPY,
                  'uniform',
                  `Graph_${instructions[0]?.id}_reduction_params`
              );
              intermediateAllocations.push({ buffer: passParamsBuf, token: passParamsToken });
              device.queue.writeBuffer(passParamsBuf, 0, new Uint32Array([currentSize, rDispatchX, 0, 0]));
              
              const wgsl = inst.op === 'sum' ? SUM_WGSL : MAX_WGSL;
              const { pipeline: reducePipeline } = _globalPipelineCache.getPipeline(inst.op + '_pass', wgsl);
              
              const passEncoder = commandEncoder.beginComputePass();
              passEncoder.setPipeline(reducePipeline);
              passEncoder.setBindGroup(0, device.createBindGroup({
                  layout: reducePipeline.getBindGroupLayout(0),
                  entries: [
                      { binding: 0, resource: { buffer: passParamsBuf } },
                      { binding: 1, resource: { buffer: currentInputBuf } },
                      { binding: 2, resource: { buffer: passBuf } },
                  ],
              }));
              passEncoder.dispatchWorkgroups(rDispatchX, rDispatchY, 1);
              passEncoder.end();
              encoderHasCommands = true;
              currentInputBuf = passBuf;
              currentSize = numWGs;
          }
          
          commandEncoder.copyBufferToBuffer(currentInputBuf, 0, outBuffer, 0, 4);
          encoderHasCommands = true;
          
          for (const alloc of intermediateAllocations) {
              paramsAllocations.push(alloc);
          }
          continue;
      }

      if (inst.op !== 'fill' && (!inst.in || inst.in.length === 0)) {
        throw new AMEVAForgeSecurityError(`Instruction op="${inst.op}" is missing 'in' field.`);
      }

      let bindGroupEntries: GPUBindGroupEntry[] = [];
      if (inst.op === 'fill') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'axpy') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'rmsnorm') {
        const gammaBuf = (inst.in && inst.in.length >= 2) ? idToBuffer[inst.in[1]] : idToBuffer[inst.in![0]];
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: gammaBuf } },
          { binding: 3, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'unpack_quant' || inst.op === 'flash_attention') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } },
          { binding: 3, resource: { buffer: idToBuffer[inst.in![2]] } },
          { binding: 4, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'gather' || inst.op === 'scatter') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } },
          { binding: 3, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'matmul_bias_relu') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } },
          { binding: 3, resource: { buffer: idToBuffer[inst.in![2]] } },
          { binding: 4, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'where') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } },
          { binding: 3, resource: { buffer: idToBuffer[inst.in![2]] } },
          { binding: 4, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'dropout') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: outBuffer } },
        ];
      } else {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
        ];

        if (inst.in!.length > 1) {
          bindGroupEntries.push({ binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } });
          bindGroupEntries.push({ binding: 3, resource: { buffer: outBuffer } });
        } else {
          bindGroupEntries.push({ binding: 2, resource: { buffer: outBuffer } });
        }
      }

      if (isMatmul) {
        const MACS_PER_CHUNK = 2_000_000_000;
        const macsPerRow = N * K;
        let chunkY = Math.max(1, Math.floor(MACS_PER_CHUNK / macsPerRow));
        chunkY = Math.min(chunkY, 65535 * 8);
        chunkY = Math.min(M, chunkY);

        const has_bias = inst.op === 'matmul_bias_relu' ? (inst.params?.[3] ?? 1) : 0;
        const has_relu = inst.op === 'matmul_bias_relu' ? (inst.params?.[4] ?? 1) : 0;

        for (let offsetY = 0; offsetY < M; offsetY += chunkY) {
          const currentChunkY = Math.min(chunkY, M - offsetY);
          
          const chunkParamEntry = _globalUniformPool.acquire(32);
          const chunkParamsBuffer = chunkParamEntry.buffer;
          paramsAllocations.push({ buffer: chunkParamsBuffer, token: chunkParamEntry.token, isUniformPool: true, uniformEntry: chunkParamEntry });
          device.queue.writeBuffer(chunkParamsBuffer, 0, new Uint32Array([M, N, K, offsetY, has_bias, has_relu, 0, 0]));
          
          const chunkBindGroupEntries = bindGroupEntries.map(e => {
            if (e.binding === 0) return { binding: 0, resource: { buffer: chunkParamsBuffer } };
            return e;
          });
          const chunkBindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: chunkBindGroupEntries
          });

          const tileSizeY = inst.op === 'matmul_bias_relu' ? 16 : 8;
          const passEncoder = commandEncoder.beginComputePass();
          passEncoder.setPipeline(pipeline);
          passEncoder.setBindGroup(0, chunkBindGroup);
          passEncoder.dispatchWorkgroups(dispatchX, Math.ceil(currentChunkY / tileSizeY), dispatchZ);
          passEncoder.end();

          opsInCurrentBatch++;
          workloadElements += (dispatchX * currentChunkY * tileSizeY * tileSizeY); 
          
          if (offsetY + currentChunkY < M || workloadElements >= _runtimeConfig.workloadBudgetElements || opsInCurrentBatch >= _runtimeConfig.maxOpsPerSubmit) {
            device.queue.submit([commandEncoder.finish()]);
            commandEncoder = device.createCommandEncoder();
            opsInCurrentBatch = 0;
            workloadElements = 0;
          }
        }
      } else {
        if (inst.op === 'scatter') {
          // If in[2] exists (base tensor x), copy x to outBuffer so unscattered elements retain x values
          if (inst.in && inst.in.length >= 3 && idToBuffer[inst.in[2]]) {
            commandEncoder.copyBufferToBuffer(idToBuffer[inst.in[2]], 0, outBuffer, 0, byteLength);
            encoderHasCommands = true;
          }
        }

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: bindGroupEntries
        });

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        passEncoder.end();

        opsInCurrentBatch++;
        workloadElements += byteLength / 4;
        if (workloadElements >= _runtimeConfig.workloadBudgetElements || opsInCurrentBatch >= _runtimeConfig.maxOpsPerSubmit) {
          device.queue.submit([commandEncoder.finish()]);
          commandEncoder = device.createCommandEncoder();
          opsInCurrentBatch = 0;
          workloadElements = 0;
        }
      }
    }

  } catch (err: any) {
    // ── 5. Rollback on Sync Error ──
    _safeLog(`[AMEVA Forge] Transaction Sync Failed. Rolling back... ${err}`);
    
    transaction.rollback();
    
    for (const alloc of paramsAllocations) {
      if (alloc.isUniformPool && alloc.uniformEntry) {
        _globalUniformPool.releaseSync(alloc.uniformEntry);
      } else {
        try { freeBuffer(alloc.buffer, alloc.token); } catch (e) {}
      }
    }
    try {
      await device.popErrorScope();
      await device.popErrorScope();
      await device.popErrorScope();
    } catch {}
    throw err;
  }

  if (encoderHasCommands || opsInCurrentBatch > 0) {
    device.queue.submit([commandEncoder.finish()]);
    encoderHasCommands = false;
  }

  // ── 5. Commit / Rollback (Async) — await error scopes before returning ──
  const internalError = await device.popErrorScope();
  const oomError = await device.popErrorScope();
  const validationError = await device.popErrorScope();

  // Check for GPU errors BEFORE returning handles
  const gpuError = internalError || oomError || validationError;
  if (gpuError) {
    _safeLog(`[AMEVA Forge] GPU error detected. Rolling back transaction... ${gpuError}`);
    transaction.rollback();
    for (const alloc of paramsAllocations) {
      if (alloc.isUniformPool && alloc.uniformEntry) {
        _globalUniformPool.releaseAfterSubmit(alloc.uniformEntry);
      } else {
        try { freeBuffer(alloc.buffer, alloc.token); } catch (e) {}
      }
    }
    void _globalUniformPool.retireSubmitted(device);
    // Determine error type
    if (internalError) {
      throw new AMEVAForgeInternalGPUError(`Internal GPU Error: ${internalError.message}`);
    } else if (oomError) {
      throw new AMEVAForgeOutOfMemoryError(`GPU Out of Memory: ${oomError.message}`);
    } else {
      throw new AMEVAForgeValidationError(`GPU Validation Error: ${validationError!.message}`);
    }
  }

  // ── 6. Commit transaction to global registry only on verified success ──
  transaction.commit(_globalRegistry);

  // ── 7. Cleanup temporary/uniform allocations after GPU completion ──
  if (paramsAllocations.length > 0) {
    const nonPoolAllocs: Array<{ buffer: GPUBuffer, token: AllocationToken }> = [];
    for (const alloc of paramsAllocations) {
      if (alloc.isUniformPool && alloc.uniformEntry) {
        _globalUniformPool.releaseAfterSubmit(alloc.uniformEntry);
      } else {
        nonPoolAllocs.push(alloc);
      }
    }
    if (_globalUniformPool.inFlightBytes() > 512 * 1024) {
      await _globalUniformPool.retireSubmitted(device);
    } else {
      void _globalUniformPool.retireSubmitted(device);
    }
    if (nonPoolAllocs.length > 0) {
      device.queue.onSubmittedWorkDone().then(() => {
        for (const alloc of nonPoolAllocs) {
          try { freeBuffer(alloc.buffer, alloc.token); } catch (e) {}
        }
      }).catch(() => {
        for (const alloc of nonPoolAllocs) {
          try { freeBuffer(alloc.buffer, alloc.token); } catch (e) {}
        }
      });
    }
  }

  return idToHandle;
}
