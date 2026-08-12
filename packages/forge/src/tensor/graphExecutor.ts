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
import { AMEVAForgeShapeError, AMEVAForgeSecurityError } from "../errors";
import { assertAllowedKernelName } from "../webgpu/shaderGuard";

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

/** 허용된 op 화이트리스트 */
const ALLOWED_OPS = new Set([
  'upload', 'load', 'matmul', 'batched_matmul', 'relu', 'add', 'mul', 'transpose', 'relu_backward',
  'sub', 'neg', 'div', 'exp', 'log', 'sigmoid', 'tanh', 'sigmoid_backward', 'tanh_backward',
  'fill', 'sum', 'max', 'sum_axis', 'axpy', 'cat', 'where', 'pad', 'gather', 'scatter', 'maxpool2d', 'avgpool2d',
  'im2col', 'col2im', 'dropout'
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
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: must be an object`);
  }

  const i = inst as Record<string, unknown>;

  if (typeof i.op !== 'string') {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: op must be a string`);
  }
  if (!ALLOWED_OPS.has(i.op)) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: unknown op "${i.op}"`);
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

  let elements = 1;
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
        throw new AMEVAForgeSecurityError(`load instruction missing handle`);
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
        throw new AMEVAForgeSecurityError(
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

    let outBuffer: GPUBuffer;
    if (inst.op === 'axpy') {
      if (!inst.in || inst.in.length < 2) {
        throw new AMEVAForgeSecurityError(`Instruction axpy is missing 'in' fields.`);
      }
      outBuffer = idToBuffer[inst.in[1]];
      idToHandle[inst.id] = idToHandle[inst.in[1]];
      idToBuffer[inst.id] = outBuffer;
    } else {
      outBuffer = allocateBuffer(
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
    }

    let paramsSize = 32;
    if (inst.op === 'pad') paramsSize = 144;
    else if (inst.op === 'gather' || inst.op === 'scatter') paramsSize = 112;
    else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d') paramsSize = 48;
    else if (inst.op === 'im2col' || inst.op === 'col2im') paramsSize = 40;

    const paramsBuffer = device.createBuffer({
      size: paramsSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    paramsBuffersToDestroy.push(paramsBuffer);

    let wgslCode = "";
    let dispatchX = 1, dispatchY = 1, dispatchZ = 1;
    
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
      if (!inst.params || inst.params.length < 7) {
        throw new AMEVAForgeSecurityError(`batched_matmul instruction missing params`);
      }
      [B, M, N, K] = inst.params;
      wgslCode = BATCHED_MATMUL_WGSL;
      dispatchX = Math.ceil(N / 8);
      dispatchY = Math.ceil(M / 8);
      dispatchZ = B;
      device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array(inst.params));
    } else if (inst.op === 'transpose') {
      if (!inst.params || inst.params.length < 3) {
        throw new AMEVAForgeSecurityError(`transpose instruction missing params`);
      }
      const [rM, rN, rB] = inst.params;
      wgslCode = TRANSPOSE_WGSL;
      device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([rM, rN, rB, 0]));
      dispatchX = Math.ceil(rM / 8);
      dispatchY = Math.ceil(rN / 8);
      dispatchZ = rB;
    } else if (inst.op === 'sum_axis') {
      if (!inst.params || inst.params.length < 2) {
        throw new AMEVAForgeSecurityError(`sum_axis instruction missing params`);
      }
      const [M, N] = inst.params;
      wgslCode = SUM_AXIS_WGSL;
      device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, N, 0, 0]));
      dispatchX = Math.ceil(N / 64);
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
      for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
      device.queue.writeBuffer(paramsBuffer, 0, p);
      dispatchX = Math.ceil(numElements / 64);
    } else if (inst.op === 'scatter') {
      const numElements = inst.params![0];
      wgslCode = SCATTER_WGSL;
      const p = new Uint32Array(28);
      for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
      device.queue.writeBuffer(paramsBuffer, 0, p);
      dispatchX = Math.ceil(numElements / 64);
    } else if (inst.op === 'dropout') {
      const numElements = byteLength / 4;
      const seed = inst.params![0];
      const p = inst.params![1];
      wgslCode = DROPOUT_WGSL;
      const f32arr = new Float32Array([0, seed, p, 0]);
      const u32arr = new Uint32Array(f32arr.buffer);
      u32arr[0] = numElements;
      device.queue.writeBuffer(paramsBuffer, 0, u32arr);
      dispatchX = Math.ceil(numElements / 64);
    } else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d') {
      const numElements = byteLength / 4;
      wgslCode = inst.op === 'maxpool2d' ? MAXPOOL2D_WGSL : AVGPOOL2D_WGSL;
      const p = new Uint32Array(12);
      for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
      device.queue.writeBuffer(paramsBuffer, 0, p);
      dispatchX = Math.ceil(numElements / 64);
    } else if (inst.op === 'im2col' || inst.op === 'col2im') {
      const numElements = byteLength / 4;
      wgslCode = inst.op === 'im2col' ? IM2COL_WGSL : COL2IM_WGSL;
      const p = new Uint32Array(10);
      for (let i = 0; i < inst.params!.length; i++) p[i] = inst.params![i];
      device.queue.writeBuffer(paramsBuffer, 0, p);
      dispatchX = Math.ceil(numElements / 64);
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
      device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, dispatchX, 0, 0, 0, 0, 0, 0]));

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
        let currentSize = byteLength / 4;
        let currentInputBuf = idToBuffer[inst.in[0]];
        const intermediateBuffers: GPUBuffer[] = [];
        
        // Use a SINGLE command encoder for ALL passes
        while (currentSize > 1) {
            const numWGs = Math.ceil(currentSize / REDUCTION_WG_SIZE);
            const passBuf = device.createBuffer({
                size: Math.max(4, numWGs * 4),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            intermediateBuffers.push(passBuf);
            
            // To ensure correct params per pass inside the same command encoder, we allocate a new uniform buffer per pass.
            // (If we rewrite paramsBuffer before submit, it might apply universally)
            const passParamsBuf = device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            intermediateBuffers.push(passParamsBuf);
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
            
            currentInputBuf = passBuf;
            currentSize = numWGs;
        }
        
        // Copy final scalar to output buffer
        commandEncoder.copyBufferToBuffer(currentInputBuf, 0, outBuffer, 0, 4);
        
        // Clean up intermediate buffers AFTER submit
        for (const buf of intermediateBuffers) {
            paramsBuffersToDestroy.push(buf);
        }
        continue; // skip normal dispatch
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
      // TS-H01 Fix: Ensure Y dispatch does not exceed 65535 workgroups
      chunkY = Math.min(chunkY, 65535 * 8);
      chunkY = Math.min(M, chunkY);

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
