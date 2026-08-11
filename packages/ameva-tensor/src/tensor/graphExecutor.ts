/**
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
import { _globalRegistry } from "./tensorRegistry";
import { TensorHandle } from "../types";
import { allocateBuffer, writeFloat32Array } from "../webgpu/buffers";
import { _globalPipelineCache } from "../webgpu/pipelineCache";
import { AMEVATensorShapeError, AMEVATensorSecurityError } from "../errors";
import { assertAllowedKernelName } from "../webgpu/shaderGuard";

// kernels
import { MATMUL_WGSL } from "./kernels/matmul.wgsl";
import { RELU_WGSL } from "./kernels/relu.wgsl";
import { ADD_WGSL } from "./kernels/add.wgsl";
import { MUL_WGSL } from "./kernels/mul.wgsl";
import { TRANSPOSE_WGSL } from "./kernels/transpose.wgsl";
import { RELU_BACKWARD_WGSL } from "./kernels/relu_backward.wgsl";

/** 허용된 op 화이트리스트 */
const ALLOWED_OPS = new Set([
  'upload', 'load', 'matmul', 'relu', 'add', 'mul', 'transpose', 'relu_backward'
]);

const MAX_SHAPE_DIM = 8; // NM-06: rank 0~8 허용
const MAX_ELEMENTS = 256 * 1024 * 1024; // 1GB (float32)
const MAX_INSTRUCTIONS = 10_000;
/**
 * TDR 방지를 위한 워크로드 기반 적응형 분할.
 * - 고정 op 수(64)가 아니라, 누적 element 수(GPU 실제 작업량)로 판단.
 * - 100M elements ≈ float32 400MB bandwidth ≈ Windows TDR 2초 타임아웃의 ~25% 마진.
 * - 스칼라 1000개: 한 배치에 모두 처리 (오버헤드 최소화).
 * - matmul 12288²: 단 1개로 즉시 분할 (TDR 방지).
 * - MAX_OPS_PER_SUBMIT: element 예산과 무관하게, 최소 분할 보장용 안전장치.
 */
const WORKLOAD_BUDGET_ELEMENTS = 100_000_000; // 100M elements per submit
const MAX_OPS_PER_SUBMIT = 256; // 안전장치: element 수 관계없이 256 ops마다 강제 분할

interface GraphInstruction {
  op: string;
  id: number;
  shape: number[];
  in?: number[];
  handle?: string;
  params?: number[];
}

/**
 * C-04: 단일 instruction의 무결성을 검증한다.
 */
function validateInstruction(inst: unknown, idx: number): GraphInstruction {
  if (typeof inst !== 'object' || inst === null) {
    throw new AMEVATensorSecurityError(`Instruction[${idx}]: must be an object`);
  }

  const i = inst as Record<string, unknown>;

  if (typeof i.op !== 'string') {
    throw new AMEVATensorSecurityError(`Instruction[${idx}]: op must be a string`);
  }
  if (!ALLOWED_OPS.has(i.op)) {
    throw new AMEVATensorSecurityError(`Instruction[${idx}]: unknown op "${i.op}"`);
  }

  if (!Number.isSafeInteger(i.id) || (i.id as number) < 1) {
    throw new AMEVATensorSecurityError(`Instruction[${idx}]: id must be a positive safe integer`);
  }

  if (!Array.isArray(i.shape)) {
    throw new AMEVATensorShapeError(`Instruction[${idx}]: shape must be an array`);
  }
  // NM-06: rank 0 허용 (스칼라)
  if (i.shape.length > MAX_SHAPE_DIM) {
    throw new AMEVATensorShapeError(
      `Instruction[${idx}]: shape rank must be 0–${MAX_SHAPE_DIM}, got ${i.shape.length}`
    );
  }

  let elements = 1;
  for (const dim of i.shape) {
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new AMEVATensorShapeError(
        `Instruction[${idx}]: shape dim must be a positive safe integer, got ${dim}`
      );
    }
    if (dim > Number.MAX_SAFE_INTEGER / elements) {
      throw new AMEVATensorShapeError(`Instruction[${idx}]: shape product integer overflow`);
    }
    elements *= dim;
  }
  if (elements > MAX_ELEMENTS) {
    throw new AMEVATensorShapeError(
      `Instruction[${idx}]: tensor too large (${elements} elements > ${MAX_ELEMENTS})`
    );
  }

  // NC-06: in 필드가 있으면 배열인지 확인
  if (i.in !== undefined && !Array.isArray(i.in)) {
    throw new AMEVATensorSecurityError(`Instruction[${idx}]: 'in' field must be an array`);
  }

  return i as unknown as GraphInstruction;
}

/**
 * executeGraph — Python 레이지 그래프를 단일 FFI 호출로 GPU에 실행한다.
 */
export function executeGraph(
  instructionsJson: string,
  jsInputs: unknown
): Record<number, TensorHandle> {
  // --- C-04: JSON 파싱 및 전체 검증 ---
  let rawInstructions: unknown[];
  try {
    rawInstructions = JSON.parse(instructionsJson);
  } catch {
    throw new AMEVATensorSecurityError("executeGraph: invalid JSON in instructionsJson");
  }

  if (!Array.isArray(rawInstructions)) {
    throw new AMEVATensorSecurityError("executeGraph: instructionsJson must be a JSON array");
  }
  if (rawInstructions.length > MAX_INSTRUCTIONS) {
    throw new AMEVATensorSecurityError(
      `executeGraph: too many instructions (${rawInstructions.length} > ${MAX_INSTRUCTIONS})`
    );
  }

  const instructions: GraphInstruction[] = rawInstructions.map(validateInstruction);

  // inputs 배열 추출 (Pyodide PyProxy 또는 JS 배열)
  let inputs: unknown[];
  if (jsInputs && typeof (jsInputs as any).toJs === 'function') {
    inputs = (jsInputs as any).toJs();
  } else if (Array.isArray(jsInputs)) {
    inputs = jsInputs;
  } else {
    inputs = [];
  }

  const device = getDevice();

  // NM-05 Fix: 전체 그래프에 대해 error scope 설정
  device.pushErrorScope('validation');

  let commandEncoder = device.createCommandEncoder();
  let opsInCurrentBatch = 0;
  let workloadElements = 0;

  const idToHandle: Record<number, TensorHandle> = {};
  const idToBuffer: Record<number, GPUBuffer> = {};

  let inputIdx = 0;
  const paramsBuffersToDestroy: GPUBuffer[] = [];

  for (const inst of instructions) {
    // byteLength = 검증된 shape에서만 계산
    const byteLength = inst.shape.reduce((a, b) => a * b, 1) * 4;

    // ── load: 기존 GPU 버퍼 참조 ──
    if (inst.op === 'load') {
      const handle = inst.handle;
      if (typeof handle !== 'string') {
        throw new AMEVATensorSecurityError(`load instruction missing handle`);
      }
      idToHandle[inst.id] = handle;
      idToBuffer[inst.id] = _globalRegistry.get(handle).buffer;
      continue;
    }

    // ── upload: 호스트 데이터 → VRAM ──
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
        throw new AMEVATensorSecurityError(
          `upload input[${inputIdx - 1}] is not a Float32Array or convertible type`
        );
      }

      const buffer = allocateBuffer(
        byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      writeFloat32Array(buffer, actualData);

      if (bufProxy) bufProxy.release();

      const handle = _globalRegistry.register({
        buffer,
        shape: inst.shape,
        dtype: "float32",
        byteLength
      });
      idToHandle[inst.id] = handle;
      idToBuffer[inst.id] = buffer;
      continue;
    }

    // ── 연산 op: GPU 커널 디스패치 ──
    // NH-07 Fix: shaderGuard에서 커널 이름 검증
    assertAllowedKernelName(inst.op);

    const outBuffer = allocateBuffer(
      byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const handle = _globalRegistry.register({
      buffer: outBuffer,
      shape: inst.shape,
      dtype: "float32",
      byteLength
    });
    idToHandle[inst.id] = handle;
    idToBuffer[inst.id] = outBuffer;

    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    paramsBuffersToDestroy.push(paramsBuffer);

    let wgslCode = "";
    let dispatchX = 1, dispatchY = 1;
    
    let isMatmul = false;
    let M = 1, N = 1, K = 1;

    if (inst.op === 'matmul') {
      if (!inst.params || inst.params.length < 3) {
        throw new AMEVATensorSecurityError(`matmul instruction missing params`);
      }
      [M, N, K] = inst.params;
      wgslCode = MATMUL_WGSL;
      isMatmul = true;
      dispatchX = Math.ceil(N / 8);
    } else if (inst.op === 'transpose') {
      if (!inst.params || inst.params.length < 2) {
        throw new AMEVATensorSecurityError(`transpose instruction missing params`);
      }
      const [rM, rN] = inst.params;
      wgslCode = TRANSPOSE_WGSL;
      device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([rM, rN, 0, 0]));
      dispatchX = Math.ceil(rM / 8);
      dispatchY = Math.ceil(rN / 8);
    } else {
      const numElements = byteLength / 4;
      wgslCode = inst.op === 'relu'          ? RELU_WGSL :
                 inst.op === 'add'           ? ADD_WGSL :
                 inst.op === 'mul'           ? MUL_WGSL :
                 inst.op === 'relu_backward' ? RELU_BACKWARD_WGSL : '';

      if (!wgslCode) {
        throw new AMEVATensorSecurityError(`Unknown op "${inst.op}"`);
      }
      device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, 0, 0, 0]));
      dispatchX = Math.ceil(numElements / 64);
    }

    const { pipeline } = _globalPipelineCache.getPipeline(inst.op, wgslCode);

    if (!inst.in || inst.in.length === 0) {
      throw new AMEVATensorSecurityError(`Instruction op="${inst.op}" is missing 'in' field.`);
    }

    const bindGroupEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
    ];

    if (inst.in.length > 1) {
      bindGroupEntries.push({ binding: 2, resource: { buffer: idToBuffer[inst.in[1]] } });
      bindGroupEntries.push({ binding: 3, resource: { buffer: outBuffer } });
    } else {
      bindGroupEntries.push({ binding: 2, resource: { buffer: outBuffer } });
    }

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries
    });

    if (isMatmul) {
      // Chunk matmul to prevent Windows TDR (2 seconds timeout)
      // Limit to ~2 billion MACs per chunk
      const MACS_PER_CHUNK = 2_000_000_000;
      const macsPerRow = N * K;
      let chunkY = Math.max(1, Math.floor(MACS_PER_CHUNK / macsPerRow));
      chunkY = Math.min(M, chunkY);

      for (let offsetY = 0; offsetY < M; offsetY += chunkY) {
        const currentChunkY = Math.min(chunkY, M - offsetY);
        
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, N, K, offsetY]));
        
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(dispatchX, Math.ceil(currentChunkY / 8));
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
      passEncoder.dispatchWorkgroups(dispatchX, dispatchY);
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

  if (opsInCurrentBatch > 0) {
    device.queue.submit([commandEncoder.finish()]);
  }

  // NM-05 Fix: error scope pop으로 validation 에러 감지
  void device.popErrorScope().then((error) => {
    if (error) {
      console.error(`[AMEVA] GPU validation error: ${error.message}`);
      // M-NEW-05: GPU 에러를 globalThis에 게시하여 Python에서 감지 가능
      if (typeof globalThis !== 'undefined') {
        (globalThis as any).__ameva_last_gpu_error = error.message;
      }
    }
  });

  // params 버퍼는 GPU 제출 완료 후 소각
  if (paramsBuffersToDestroy.length > 0) {
    device.queue.onSubmittedWorkDone().then(() => {
      paramsBuffersToDestroy.forEach(b => b.destroy());
    });
  }

  return idToHandle;
}
