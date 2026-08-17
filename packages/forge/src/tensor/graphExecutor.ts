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


// kernels
import { MATMUL_WGSL } from "./kernels/matmul.wgsl";
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

/** 
 * WHAT: 그래프 실행기가 처리할 수 있는 모든 허용된 오퍼레이션(op)의 집합입니다.
 * WHY: 악의적인 JSON 그래프가 알 수 없거나 금지된 셰이더를 실행하여 GPU를 공격하는 것을 방지하기 위한 화이트리스트입니다.
 * HOW: Set 자료구조에 허용되는 오퍼레이션 문자열을 초기화하여 빠른 조회(O(1))를 제공합니다.
 */
const ALLOWED_OPS = new Set([
  'upload', 'load', 'matmul', 'batched_matmul', 'relu', 'add', 'mul', 'transpose', 'relu_backward',
  'sub', 'neg', 'div', 'exp', 'log', 'sigmoid', 'tanh', 'sigmoid_backward', 'tanh_backward',
  'fill', 'sum', 'max', 'sum_axis', 'axpy', 'cat', 'where', 'pad', 'gather', 'scatter', 'maxpool2d', 'avgpool2d',
  'im2col', 'col2im', 'dropout', 'permute', 'matmul_bias_relu'
]);

/** 
 * WHAT: 단일 텐서가 가질 수 있는 최대 랭크(차원 수)입니다.
 * WHY: 다차원 반복이나 과도하게 큰 셰이더 파라미터가 유발하는 오버플로우와 성능 저하를 방지하기 위해 제한합니다.
 * HOW: 상수 8로 설정되어, 0(스칼라)부터 8차원까지만 검증을 통과하도록 합니다.
 */
const MAX_SHAPE_DIM = 8; // NM-06: rank 0~8 허용

/** 
 * WHAT: 단일 텐서가 가질 수 있는 최대 원소의 개수입니다 (float32 기준 1GB).
 * WHY: 악의적인 대용량 텐서 생성 명령으로 인해 브라우저나 디바이스의 VRAM이 고갈(OOM)되는 것을 막기 위함입니다.
 * HOW: 256 * 1024 * 1024 (약 2억 6천만 개)로 정의되어 상한선으로 동작합니다.
 */
const MAX_ELEMENTS = 256 * 1024 * 1024; // 1GB (float32)

/** 
 * WHAT: 하나의 그래프 실행 요청에 포함될 수 있는 최대 명령어(instruction)의 수입니다.
 * WHY: 너무 거대한 그래프 루프를 실행하다가 메인 스레드가 블로킹되거나 TDR이 발생하는 것을 막습니다.
 * HOW: 상수 10,000으로 설정되어 JSON 배열 길이를 제한합니다.
 */
const MAX_INSTRUCTIONS = 10_000;

/**
 * TDR 방지를 위한 워크로드 기반 적응형 분할.
 * WHAT: 단일 커맨드 제출(Submit) 당 누적 허용되는 총 GPU 작업량(원소 수) 예산입니다.
 * WHY: 윈도우 환경 등에서 GPU 작업이 2초 이상 걸리면 발생하는 TDR(Timeout Detection and Recovery)을 회피하기 위해 작업을 쪼갭니다.
 * HOW: 약 1억 개(100M)의 요소를 기준으로 청크(chunk)를 나누도록 상수를 설정합니다.
 */
const WORKLOAD_BUDGET_ELEMENTS = 100_000_000; // 100M elements per submit

/** 
 * WHAT: 단일 커맨드 제출 당 포함될 수 있는 최대 디스패치(오퍼레이션) 수입니다.
 * WHY: 워크로드가 작더라도 слишком 많은 작은 연산을 한 번에 보내면 발생할 수 있는 오버헤드와 브라우저 블로킹을 방지합니다.
 * HOW: 256개 명령어마다 무조건 큐에 submit 하도록 강제합니다.
 */
const MAX_OPS_PER_SUBMIT = 256; // 안전장치: element 수 관계없이 256 ops마다 강제 분할

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
  } catch {}
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
          _globalQuotaManager.releaseToken(item.token);
        } catch {}
        _deferredGCQueue.splice(i, 1);
        _safeLog(`[DeferredGC] Failed to destroy buffer after 3 attempts, token released: ${e}`);
      }
    }
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
  if (i.shape.length > MAX_SHAPE_DIM) {
    throw new AMEVAForgeShapeError(
      `Instruction[${idx}]: shape rank must be 0–${MAX_SHAPE_DIM}, got ${i.shape.length}`
    );
  }

  /**
   * WHAT: 해당 명령어 텐서의 누적 원소 수를 계산하는 변수입니다.
   * WHY: 차원의 곱이 안전한 정수 범위를 넘거나 최대 한계(MAX_ELEMENTS)를 초과하는지 확인하기 위해 계산합니다.
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
  if (elements > MAX_ELEMENTS) {
    throw new AMEVAForgeShapeError(
      `Instruction[${idx}]: tensor too large (${elements} elements > ${MAX_ELEMENTS})`
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
    'relu_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'sigmoid_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'tanh_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'pad': { minIn: 1, exactIn: true, minParams: 9, exactParams: true }, // pad는 최대 4차원 36바이트 = 9 uint32s.
    'sum_axis': { minIn: 1, exactIn: true, minParams: 2, exactParams: true },
    'dropout': { minIn: 1, exactIn: true, minParams: 2, exactParams: true },
    'maxpool2d': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'avgpool2d': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'im2col': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
    'transpose': { minIn: 1, exactIn: true, minParams: 2, exactParams: false },
    'permute': { minIn: 1, exactIn: true, minParams: 1, exactParams: false }, // rank 길이 가변
    'add': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'sub': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'mul': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'div': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
    'axpy': { minIn: 2, exactIn: true, minParams: 2, exactParams: true },
    'gather': { minIn: 2, exactIn: true, minParams: 7, exactParams: true },
    'scatter': { minIn: 2, exactIn: true, minParams: 7, exactParams: true },
    'matmul': { minIn: 2, exactIn: true, minParams: 3, exactParams: true },
    'batched_matmul': { minIn: 2, exactIn: true, minParams: 4, exactParams: true },
    'where': { minIn: 3, exactIn: true, minParams: 0, exactParams: true },
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
  // ── 1. Parse ──
  /**
   * WHAT: JSON 문자열에서 파싱된 원시(unvalidated) 자바스크립트 객체 배열입니다.
   * WHY: 외부 문자열 데이터를 자바스크립트 객체 트리로 메모리에 올리기 위해 저장합니다.
   * HOW: JSON.parse()를 시도하며, 예외 발생 시 AMEVAForgeSecurityError를 던집니다.
   */
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
  if (rawInstructions.length > MAX_INSTRUCTIONS) {
    throw new AMEVAForgeSecurityError(
      `executeGraph: too many instructions (${rawInstructions.length} > ${MAX_INSTRUCTIONS})`
    );
  }

  // ── 2. Validate ──
  /**
   * WHAT: 원시 배열을 검증하여 타입 안정성이 보장된 명령어들의 배열입니다.
   * WHY: 이후의 실행 단계(Execution)에서 데이터를 신뢰하고 빠른 연산을 수행할 수 있도록 합니다.
   * HOW: Array.map을 이용해 각 항목을 validateInstruction 함수로 통과시킵니다.
   */
  const instructions: GraphInstruction[] = rawInstructions.map(validateInstruction);

  /**
   * WHAT: 그래프 실행 중 'upload' 오퍼레이션에서 사용할 외부 입력 데이터들의 배열입니다.
   * WHY: GPU 밖에서 들어오는 가중치(weights)나 입력 데이터(inputs)를 순차적으로 소비하기 위해 변환해 둡니다.
   * HOW: Pyodide의 toJs()가 있으면 변환하고, 없으면 배열로 간주하며, 둘 다 아니면 빈 배열로 초기화합니다.
   */
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
  /**
   * WHAT: WebGPU 작업을 제출할 대상 디바이스 인터페이스입니다.
   * WHY: 커맨드 인코더 생성과 버퍼 조작 및 에러 스코프를 설정하기 위해 필요합니다.
   * HOW: getDevice()를 호출하여 얻습니다.
   */
  const device = getDevice();
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('internal');

  /**
   * WHAT: 현재 트랜잭션 배치의 GPU 명령을 기록하는 커맨드 인코더 객체입니다.
   * WHY: 개별 오퍼레이션의 상태 변경과 디스패치를 모아 한 번에 GPU 큐로 전송하기 위해 유지합니다.
   * HOW: device.createCommandEncoder()로 생성하며, 청크가 나뉠 때마다 재생성됩니다.
   */
  let commandEncoder = device.createCommandEncoder();
  
  /**
   * WHAT: 현재 커맨드 인코더에 쌓인 오퍼레이션(디스패치)의 개수입니다.
   * WHY: MAX_OPS_PER_SUBMIT 상한선에 도달했는지 판단하여 강제 플러시(flush)를 트리거하기 위해 카운팅합니다.
   * HOW: 디스패치를 하나 추가할 때마다 1씩 증가시킵니다.
   */
  let opsInCurrentBatch = 0;
  let encoderHasCommands = false;
  
  /**
   * WHAT: 현재 커맨드 인코더에 제출된 총 연산 원소 수(워크로드)입니다.
   * WHY: WORKLOAD_BUDGET_ELEMENTS 상한선에 도달했는지 확인하여 TDR 현상을 피하도록 배치를 끊기 위해 계산합니다.
   * HOW: 디스패치할 때마다 해당 텐서의 요소를 더합니다.
   */
  let workloadElements = 0;
  
  /**
   * WHAT: 명령어 ID(명령어별 고유 식별자)를 생성된 텐서 핸들에 매핑하는 객체입니다.
   * WHY: 최종적으로 외부 환경(Python 등)에 어떤 ID가 어떤 텐서를 반환했는지 결과를 돌려주기 위해 유지합니다.
   * HOW: 키는 명령어 id, 값은 TensorHandle(문자열)로 할당합니다.
   */
  const idToHandle: Record<number, TensorHandle> = {};
  const idToBuffer: Record<number, GPUBuffer> = {};
  const idToByteLength: Record<number, number> = {};
  const transaction = new GraphTransaction();
  let inputIdx = 0;
  
  /**
   * WHAT: 셰이더의 파라미터 전달을 위해 임시로 생성된 유니폼 버퍼(Uniform Buffer)들의 배열입니다.
   * WHY: GPU 큐 작업이 비동기적으로 완료된 후, 이 임시 버퍼들을 모아서 파괴(destroy)하여 메모리 누수를 방지하기 위해 저장합니다.
   * HOW: 디스패치 과정에서 createBuffer된 파라미터 버퍼들을 push()로 수집합니다.
   */
  const paramsAllocations: Array<{ buffer: GPUBuffer, token: AllocationToken }> = [];

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
        continue;
      }

      if (inst.op === 'upload') {
        /**
         * WHAT: CPU 혹은 Pyodide 메모리에서 건네받은 입력 텐서의 원시 데이터입니다.
         * WHY: 이 데이터를 GPU 버퍼로 복사하여 연산에 투입하기 위해 필요합니다.
         * HOW: inputs 배열에서 inputIdx가 가리키는 값을 꺼내옵니다.
         */
        const rawData = inputs[inputIdx++];
        
        /**
         * WHAT: 원시 데이터에서 실제 복사 가능한 형태로 추출된 Float32Array 데이터입니다.
         * WHY: WebGPU의 writeBuffer API는 타입화된 자바스크립트 배열 뷰를 요구하기 때문입니다.
         * HOW: 데이터의 타입(Pyodide 프록시, Float32Array, 일반 배열)에 따라 변환 및 캐스팅을 수행합니다.
         */
        let actualData: Float32Array;
        
        /**
         * WHAT: 외부 WASM 메모리 뷰 프록시입니다.
         * WHY: 데이터를 다 읽은 후 메모리 락을 해제(release)하기 위해 보존합니다.
         * HOW: 데이터가 getBuffer 메서드를 제공할 때만 생성됩니다.
         */
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

        // H-02 Fix: WASM 메모리 바운드 사전 검증
        if (actualData && actualData.buffer) {
          assertWasmRange(actualData.byteOffset, actualData.byteLength, actualData.buffer.byteLength);
        }

        // VUL-018: NaN / Inf 방어
        /**
         * WHAT: 입력 데이터에 무한대나 NaN 값이 포함되어 있는지 검사하고 치환하는 방어 루프입니다.
         * WHY: NaN 또는 무한대 값이 GPU 셰이더로 흘러가면 연산을 망가뜨리고 TDR 크래시를 유발할 수 있으므로 보호막 역할을 합니다.
         * HOW: 배열의 모든 요소를 순회하며 Number.isFinite()로 검사하고, 유효하지 않으면 0으로 마스킹합니다.
         */
        for (let i = 0; i < actualData.length; i++) {
          if (!Number.isFinite(actualData[i])) {
            actualData[i] = 0; // TDR 방지를 위해 0으로 클램프하거나, 경고 로깅 가능 (여기서는 0으로 마스킹)
            _safeLog(`[GraphExecutor] NaN or Inf detected in upload input[${inputIdx - 1}], masked to 0`);
          }
        }

        /**
         * WHAT: 업로드된 데이터를 담기 위해 GPU에 새로 생성된 스토리지 버퍼와 토큰입니다.
         * WHY: 데이터를 VRAM으로 옮겨 이후 연산 노드들이 접근할 수 있도록 만듭니다.
         * HOW: allocateBuffer 헬퍼를 호출하여 STORAGE 및 COPY 용도의 버퍼를 생성합니다.
         */
        const { buffer, token } = allocateBuffer(
          byteLength,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
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
      } else {
        const { buffer, token } = allocateBuffer(
          byteLength,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
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
      }

      /**
       * WHAT: 현재 오퍼레이션의 유니폼 파라미터를 담기 위해 필요한 바이트 크기입니다.
       * WHY: 오퍼레이션(패딩, 풀링 등)마다 셰이더가 요구하는 인자의 종류와 개수가 다르므로 가변적인 버퍼 크기를 잡기 위해 결정합니다.
       * HOW: inst.op 문자열을 판별하여 필요한 바이트 수(최소 32바이트)를 할당합니다.
       */
      let paramsSize = 32;
      if (inst.op === 'pad') paramsSize = 144;
      else if (inst.op === 'gather' || inst.op === 'scatter') paramsSize = 112;
      else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d') paramsSize = 48;
      else if (inst.op === 'im2col' || inst.op === 'col2im') paramsSize = 40;
      else if (inst.op === 'permute') paramsSize = 112;

      /**
       * WHAT: GPU 연산 커널에 동적 스칼라 인자를 전달하기 위한 유니폼 버퍼입니다.
       * WHY: 각 연산의 크기나 특수 인자(스토라이드, 패딩 값 등)를 셰이더 내에서 읽을 수 있게 제공해야 합니다.
       * HOW: 계산된 paramsSize로 device.createBuffer를 호출하고 UNIFORM 속성을 지정합니다. 생성 후에는 paramsBuffersToDestroy에 등록해 사후 삭제를 예약합니다.
       */
      const { buffer: paramsBuffer, token: paramsToken } = allocateBuffer(
        paramsSize,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        'uniform',
        `Graph_${instructions[0]?.id}_params`
      );
      paramsAllocations.push({ buffer: paramsBuffer, token: paramsToken });

      /**
       * WHAT: 현재 실행할 WGSL 셰이더 소스 코드를 담는 문자열 변수입니다.
       * WHY: 오퍼레이션 키워드(inst.op)에 맞는 셰이더를 매핑하여 캐시 조회 및 파이프라인 생성에 넘기기 위함입니다.
       * HOW: op의 종류에 따라 상수 문자열을 매핑합니다.
       */
      let wgslCode = "";
      
      /**
       * WHAT: 컴퓨트 셰이더를 실행할 3차원 그리드(워크그룹)의 X, Y, Z 디스패치 개수입니다.
       * WHY: GPU 하드웨어에 얼마나 많은 스레드 블록을 띄워 연산을 처리할지 스케줄링하기 위해 계산합니다.
       * HOW: 기본값 1로 시작하며, 텐서 크기와 연산 종류에 맞춰 수식이 변동됩니다.
       */
      let dispatchX = 1, dispatchY = 1, dispatchZ = 1;
      
      /**
       * WHAT: 현재 오퍼레이션이 행렬 곱(Matmul) 계열인지 여부를 나타내는 불리언 플래그입니다.
       * WHY: 행렬 곱 연산은 계산 집약적이므로 TDR 방지를 위해 특별한 청크 단위(chunking) 디스패치가 필요하여 이를 구분하기 위해 사용합니다.
       * HOW: inst.op가 'matmul'일 때 true로 설정됩니다.
       */
      let isMatmul = false;
      let B = 1, M = 1, N = 1, K = 1;

      if (inst.op === 'matmul') {
        if (!inst.params || inst.params.length < 3) {
          throw new AMEVAForgeSecurityError(`matmul instruction missing params`);
        }
        [M, N, K] = inst.params;
        wgslCode = MATMUL_WGSL;
        isMatmul = true;
        // TS-H01 Fix: matmul X축도 65535 클램핑 — 초과분은 Z 차원으로 분산
        const rawDispatchX = Math.ceil(N / 8);
        if (rawDispatchX <= 65535) {
          dispatchX = rawDispatchX;
        } else {
          dispatchX = 65535;
          dispatchZ = Math.ceil(rawDispatchX / 65535);
        }
        const maxWorkgroupsM = Math.ceil(M / 8);
        
        dispatchY = Math.min(65535, maxWorkgroupsM);
      } else if (inst.op === 'batched_matmul') {
        if (!inst.params || inst.params.length < 4) {
          throw new AMEVAForgeSecurityError(`batched_matmul instruction missing params`);
        }
        const [B_param, N_param, P_param, M_param] = inst.params;
        B = B_param;
        wgslCode = BATCHED_MATMUL_WGSL;
        
        const rawDispatchX = Math.ceil(P_param / 8);
        if (rawDispatchX <= 65535) {
          dispatchX = rawDispatchX;
        } else {
          throw new AMEVAForgeSecurityError(`batched_matmul dispatchX exceeded limit: ${rawDispatchX}`);
        }
        
        const rawDispatchY = Math.ceil(N_param / 8);
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
        
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array(inst.params));
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
      } else if (inst.op === 'sum_axis') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`sum_axis instruction missing params`);
        }
        const [M_param, N_param] = inst.params;
        wgslCode = SUM_AXIS_WGSL;
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M_param, N_param, 0, 0]));
        dispatchX = Math.ceil(N_param / 64);
      } else if (inst.op === 'fill') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`fill instruction missing params`);
        }
        const numElements = inst.params[0];
        const fillValue = inst.params[1];
        wgslCode = FILL_WGSL;
        const f32arr = new Float32Array([0, fillValue, 0, 0]);
        const u32arr = new Uint32Array(f32arr.buffer);
        u32arr[0] = numElements;
        device.queue.writeBuffer(paramsBuffer, 0, u32arr);
        dispatchX = Math.ceil(numElements / 64);
      } else if (inst.op === 'axpy') {
        if (!inst.params || inst.params.length < 2) {
          throw new AMEVAForgeSecurityError(`axpy instruction missing params`);
        }
        const numElements = inst.params[0];
        const lr = inst.params[1];
        wgslCode = AXPY_WGSL;
        const f32arr = new Float32Array([0, lr, 0, 0]);
        const u32arr = new Uint32Array(f32arr.buffer);
        u32arr[0] = numElements;
        device.queue.writeBuffer(paramsBuffer, 0, u32arr);
        dispatchX = Math.ceil(numElements / 64);
      } else if (inst.op === 'pad') {
        const numElements = byteLength / 4;
        wgslCode = PAD_WGSL;
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
        device.queue.writeBuffer(paramsBuffer, 0, p);
        dispatchX = Math.ceil(numElements / 64);
      } else if (inst.op === 'gather') {
        const numElements = byteLength / 4;
        wgslCode = GATHER_WGSL;
        const p = new Uint32Array(28);
        /**
         * WHAT: 파라미터를 복사하는 짧은 루프입니다.
         * WHY: gather 커널에 필요한 형태와 인덱싱 오프셋 정보들을 전송하기 위해 복사합니다.
         * HOW: 파라미터를 하나씩 Uint32Array에 대입합니다.
         */
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        device.queue.writeBuffer(paramsBuffer, 0, p);
        dispatchX = Math.ceil(numElements / 64);
      } else if (inst.op === 'scatter') {
        const numElements = inst.params![0];
        wgslCode = SCATTER_WGSL;
        const p = new Uint32Array(28);
        /**
         * WHAT: scatter 셰이더 인자를 복사하는 루프입니다.
         * WHY: 분산 배치할 인덱스 스텝 정보를 넘기기 위함입니다.
         * HOW: 파라미터를 하나씩 Uint32Array에 대입합니다.
         */
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        device.queue.writeBuffer(paramsBuffer, 0, p);
        dispatchX = Math.ceil(numElements / 64);
      } else if (inst.op === 'dropout') {
        const numElements = byteLength / 4;
        const seed = inst.params![0];
        const p_val = inst.params![1];
        wgslCode = DROPOUT_WGSL;
        const f32arr = new Float32Array([0, seed, p_val, 0]);
        const u32arr = new Uint32Array(f32arr.buffer);
        u32arr[0] = numElements;
        device.queue.writeBuffer(paramsBuffer, 0, u32arr);
        dispatchX = Math.ceil(numElements / 64);
      } else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d') {
        const numElements = byteLength / 4;
        wgslCode = inst.op === 'maxpool2d' ? MAXPOOL2D_WGSL : AVGPOOL2D_WGSL;
        const p = new Uint32Array(12);
        /**
         * WHAT: 풀링 파라미터를 복사하는 루프입니다.
         * WHY: 윈도우 크기, 스트라이드, 패딩 등 컨볼루션 구조를 셰이더에 넘기기 위함입니다.
         * HOW: 요소별로 배열에 대입합니다.
         */
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        device.queue.writeBuffer(paramsBuffer, 0, p);
        dispatchX = Math.ceil(numElements / 64);
      } else if (inst.op === 'im2col' || inst.op === 'col2im') {
        const numElements = byteLength / 4;
        wgslCode = inst.op === 'im2col' ? IM2COL_WGSL : COL2IM_WGSL;
        const p = new Uint32Array(10);
        /**
         * WHAT: 공간 변환 파라미터를 복사하는 루프입니다.
         * WHY: 이미지 크기와 패치 크기 데이터를 셰이더에 전달하기 위해 수행합니다.
         * HOW: 반복문을 통해 할당합니다.
         */
        for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
        device.queue.writeBuffer(paramsBuffer, 0, p);
        dispatchX = Math.ceil(numElements / 64);
      } else if (inst.op === 'permute') {
        const numElements = byteLength / 4;
        wgslCode = PERMUTE_WGSL;
        const dims = inst.params!;
        const rank = dims.length;
        
        const inHandle = idToHandle[inst.in![0]];
        const inShape = _globalRegistry.get(inHandle).shape;
        
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
        
        const p = new Uint32Array(28);
        p[0] = rank;
        p[1] = numElements;
        
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
        
        const totalWorkgroups = Math.ceil(numElements / 64);
        if (totalWorkgroups <= 65535) {
            dispatchX = totalWorkgroups;
        } else {
            dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
            dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
        }
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
        let numA = 0;
        let numB = 0;
        if (inst.in && inst.in.length >= 2) {
          numA = (idToByteLength[inst.in[0]] ?? byteLength) / 4;
          numB = (idToByteLength[inst.in[1]] ?? byteLength) / 4;
        }
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, dispatchX, numA, numB, 0, 0, 0, 0]));

        if (inst.op === 'cat') {
          if (!inst.params || inst.params.length < 3) {
            throw new AMEVAForgeSecurityError(`cat instruction missing params`);
          }
          const [a_dim, b_dim, stride] = inst.params;
          // Overwrite the params for cat
          device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, dispatchX, a_dim, b_dim, stride, 0, 0, 0]));
        }
      }

      /**
       * WHAT: 파이프라인 캐시에서 가져온 컴파일된 WebGPU 컴퓨트 파이프라인 객체입니다.
       * WHY: 커맨드 인코더가 GPU에서 셰이더를 구동하기 위한 명세(Layout)를 설정하기 위해 참조합니다.
       * HOW: _globalPipelineCache.getPipeline()을 통해 조회 혹은 캐싱 생성하여 얻습니다.
       */
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
          
          /**
           * WHAT: 병렬 리덕션(Reduction) 연산을 위한 다중 패스 트리 루프입니다.
           * WHY: 전체 배열을 하나의 스칼라로 압축하기 위해 여러 번의 컴퓨트 패스를 통해 계층적으로 데이터를 축소시키기 위함입니다.
           * HOW: 원소 수가 1이 될 때까지 while 루프를 돌며, 임시 버퍼를 만들고 리덕션 패스를 제출하여 크기를 줄여 나갑니다.
           */
          while (currentSize > 1) {
              const numWGs = Math.ceil(currentSize / REDUCTION_WG_SIZE);
              const { buffer: passBuf, token: passBufToken } = allocateBuffer(
                  Math.max(4, numWGs * 4),
                  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                  'temporary',
                  `Graph_${instructions[0]?.id}_reduction`
              );
              intermediateAllocations.push({ buffer: passBuf, token: passBufToken });
              
              const { buffer: passParamsBuf, token: passParamsToken } = allocateBuffer(
                  16,
                  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                  'uniform',
                  `Graph_${instructions[0]?.id}_reduction_params`
              );
              intermediateAllocations.push({ buffer: passParamsBuf, token: passParamsToken });
              device.queue.writeBuffer(passParamsBuf, 0, new Uint32Array([currentSize, 0, 0, 0]));
              
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
              passEncoder.dispatchWorkgroups(numWGs);
              passEncoder.end();
              encoderHasCommands = true;
              currentInputBuf = passBuf;
              currentSize = numWGs;
          }
          
          commandEncoder.copyBufferToBuffer(currentInputBuf, 0, outBuffer, 0, 4);
          encoderHasCommands = true;
          
          /**
           * WHAT: 리덕션 연산 중 만들어진 중간 임시 버퍼들을 수집하는 루프입니다.
           * WHY: 작업 완료 후 가비지 컬렉션이나 명시적 해제를 수행하여 메모리 릭을 방지하기 위함입니다.
           * HOW: for...of 구문으로 intermediateBuffers 배열을 순회하여 paramsBuffersToDestroy에 등록합니다.
           */
          for (const alloc of intermediateAllocations) {
              paramsAllocations.push(alloc);
          }
          continue; // skip normal dispatch
      }

      if (inst.op !== 'fill' && (!inst.in || inst.in.length === 0)) {
        throw new AMEVAForgeSecurityError(`Instruction op="${inst.op}" is missing 'in' field.`);
      }

      /**
       * WHAT: 파이프라인 레이아웃에 맞춰 GPUBuffer를 슬롯(binding)에 매핑하는 배열입니다.
       * WHY: 컴퓨트 셰이더 내부의 @group(0) @binding(N) 변수들과 실제 VRAM 메모리를 연결하기 위해 필요합니다.
       * HOW: 연산 종류에 따라 분기하여 각 입력 텐서 버퍼들과 출력 버퍼를 순서대로 할당합니다.
       */
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
          { binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } },
        ];
      } else if (inst.op === 'pad') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: outBuffer } },
        ];
      } else if (inst.op === 'gather' || inst.op === 'scatter') {
        bindGroupEntries = [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: idToBuffer[inst.in![0]] } },
          { binding: 2, resource: { buffer: idToBuffer[inst.in![1]] } },
          { binding: 3, resource: { buffer: outBuffer } },
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

      /**
       * WHAT: 앞서 설정한 bindGroupEntries 리스트를 토대로 생성된 바인드 그룹 객체입니다.
       * WHY: 실제 컴퓨트 패스 인코더에 setBindGroup을 호출하기 위해 WebGPU의 투명한 핸들로 필요합니다.
       * HOW: device.createBindGroup을 사용하여 파이프라인 레이아웃 규칙에 맞춰 버퍼들을 확정(commit)합니다.
       */
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindGroupEntries
      });

      if (isMatmul) {
        const MACS_PER_CHUNK = 2_000_000_000;
        const macsPerRow = N * K;
        let chunkY = Math.max(1, Math.floor(MACS_PER_CHUNK / macsPerRow));
        // TS-H01 Fix: Ensure Y dispatch does not exceed 65535 workgroups
        chunkY = Math.min(chunkY, 65535 * 8);
        chunkY = Math.min(M, chunkY);

        /**
         * WHAT: 행렬 곱셈 연산을 Y축(행) 기준으로 여러 청크(Chunk)로 분할 처리하는 루프입니다.
         * WHY: 단일 행렬 곱 연산이 너무 거대하여 GPU 실행 한계 시간(Timeout)을 초과하는 TDR 현상을 피하기 위해 작업을 작게 나눕니다.
         * HOW: for 루프를 통해 offsetY 변수를 증가시키면서 전체 행(M)을 chunkY만큼씩 잘라 컴퓨트 패스를 큐에 넘깁니다.
         */
        for (let offsetY = 0; offsetY < M; offsetY += chunkY) {
          const currentChunkY = Math.min(chunkY, M - offsetY);
          
          device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, N, K, offsetY]));
          
          const passEncoder = commandEncoder.beginComputePass();
          passEncoder.setPipeline(pipeline);
          passEncoder.setBindGroup(0, bindGroup);
          passEncoder.dispatchWorkgroups(dispatchX, Math.ceil(currentChunkY / 8), dispatchZ);
          passEncoder.end();

          opsInCurrentBatch++;
          workloadElements += (dispatchX * currentChunkY * 8 * 8); 
          
          if (offsetY + currentChunkY < M || workloadElements >= WORKLOAD_BUDGET_ELEMENTS || opsInCurrentBatch >= MAX_OPS_PER_SUBMIT) {
            device.queue.submit([commandEncoder.finish()]);
            commandEncoder = device.createCommandEncoder();
            opsInCurrentBatch = 0;
            workloadElements = 0;
          }
        }
      } else {
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        passEncoder.end();

        opsInCurrentBatch++;
        workloadElements += byteLength / 4;
        if (workloadElements >= WORKLOAD_BUDGET_ELEMENTS || opsInCurrentBatch >= MAX_OPS_PER_SUBMIT) {
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
      try { 
        freeBuffer(alloc.buffer, alloc.token); 
      } catch (e) { 
        _safeLog(`[GraphExecutor] Error freeing param buffer: ${e}`); 
      }
    }
    // pop scopes to prevent leak
    void device.popErrorScope();
    void device.popErrorScope();
    void device.popErrorScope();
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
      try { 
        freeBuffer(alloc.buffer, alloc.token); 
      } catch (e) { 
        _safeLog(`[GraphExecutor] Error freeing param buffer during rollback: ${e}`); 
      }
    }
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
    device.queue.onSubmittedWorkDone().then(() => {
      for (const alloc of paramsAllocations) {
        try { 
          freeBuffer(alloc.buffer, alloc.token); 
        } catch (e) { 
          _safeLog(`[GraphExecutor] Error freeing submitted buffer: ${e}`); 
        }
      }
    }).catch((e) => {
      _safeLog(`[GraphExecutor] onSubmittedWorkDone error: ${e}`);
      for (const alloc of paramsAllocations) {
        try { 
          freeBuffer(alloc.buffer, alloc.token); 
        } catch (err) { 
          _safeLog(`[GraphExecutor] Error freeing submitted buffer on error: ${err}`); 
        }
      }
    });
  }

  return idToHandle;
}
