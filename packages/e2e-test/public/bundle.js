// node_modules/@ameva/forge/dist/index.esm.js
var AMEVAForgeError = class extends Error {
  constructor(message) {
    super(message);
    this.name = new.target.name;
    Object.setPrototypeOf(this, new.target.prototype);
  }
};
var AMEVAForgeShapeError = class extends AMEVAForgeError {
};
var AMEVAForgeDeviceError = class extends AMEVAForgeError {
};
var AMEVAForgeDisposedError = class extends AMEVAForgeError {
};
var AMEVAForgeQuotaExceededError = class extends AMEVAForgeError {
};
var AMEVAForgeWebGPUUnavailableError = class extends AMEVAForgeError {
};
var AMEVAForgeSecurityError = class extends AMEVAForgeError {
};
var device = null;
var adapter = null;
var onDeviceLostCallback = null;
async function initWebGPU(options) {
  if (device)
    return;
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new AMEVAForgeWebGPUUnavailableError("WebGPU is not available in this environment. Ensure you are running in a supported browser with WebGPU enabled.");
  }
  adapter = await navigator.gpu.requestAdapter(options);
  if (!adapter) {
    throw new AMEVAForgeWebGPUUnavailableError("Failed to request a WebGPU adapter. Your GPU may not support WebGPU, or the browser has disabled it.");
  }
  device = await adapter.requestDevice();
  device.lost.then((info) => {
    console.error(`[AMEVA] WebGPU Device Lost: ${info.message} (reason: ${info.reason})`);
    device = null;
    adapter = null;
    if (onDeviceLostCallback) {
      onDeviceLostCallback();
    }
  });
}
function getDevice() {
  if (!device) {
    throw new AMEVAForgeDeviceError("WebGPU device is not initialized. Call await init() first.");
  }
  return device;
}
function getAdapter() {
  return adapter;
}
function getQueue() {
  return getDevice().queue;
}
function isAvailable() {
  return device !== null;
}
function setDeviceLostCallback(callback) {
  onDeviceLostCallback = callback;
}
var device$1 = /* @__PURE__ */ Object.freeze({
  __proto__: null,
  getAdapter,
  getDevice,
  getQueue,
  initWebGPU,
  isAvailable,
  setDeviceLostCallback
});
var QuotaManager = class {
  /** 현재 할당된 총 바이트 (pending 포함) */
  allocatedBytes = 0;
  /** GPU 큐 대기 중인 해제 바이트 (실제로는 아직 점유 중) */
  pendingReleaseBytes = 0;
  hardLimitBytes;
  softLimitBytes;
  constructor(hardLimitBytes = 2 * 1024 * 1024 * 1024, softLimitBytes = Math.floor(1.5 * 1024 * 1024 * 1024)) {
    this._assertValidByteLength(hardLimitBytes);
    this._assertValidByteLength(softLimitBytes);
    if (softLimitBytes > hardLimitBytes) {
      throw new AMEVAForgeQuotaExceededError("softLimitBytes must be <= hardLimitBytes");
    }
    this.hardLimitBytes = hardLimitBytes;
    this.softLimitBytes = softLimitBytes;
  }
  /** H-04: 런타임에 동적으로 쿼터 상한 재설정 */
  setLimits(hardLimitBytes, softLimitBytes) {
    this._assertValidByteLength(hardLimitBytes);
    this._assertValidByteLength(softLimitBytes);
    if (softLimitBytes > hardLimitBytes) {
      throw new AMEVAForgeQuotaExceededError("softLimitBytes must be <= hardLimitBytes");
    }
    this.hardLimitBytes = hardLimitBytes;
    this.softLimitBytes = softLimitBytes;
  }
  _assertValidByteLength(byteLength) {
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
      throw new AMEVAForgeQuotaExceededError(`Invalid allocation size: ${byteLength}`);
    }
  }
  reserve(byteLength) {
    this._assertValidByteLength(byteLength);
    if (byteLength > this.hardLimitBytes - this.allocatedBytes) {
      throw new AMEVAForgeQuotaExceededError(`Quota Exceeded: Cannot allocate ${byteLength} bytes. Current: ${this.allocatedBytes} (${this.pendingReleaseBytes} pending release), Limit: ${this.hardLimitBytes}`);
    }
    this.allocatedBytes += byteLength;
    if (this.allocatedBytes - this.pendingReleaseBytes > this.softLimitBytes) {
      console.warn(`[AMEVA] VRAM soft quota exceeded: ${((this.allocatedBytes - this.pendingReleaseBytes) / 1e9).toFixed(2)}GB / ${(this.softLimitBytes / 1e9).toFixed(2)}GB`);
    }
  }
  /**
   * C-06: dispose() 호출 시 즉시 "해제 예정"으로 표시.
   * 실제 GPU 메모리 해제(release())는 GPU 큐 완료 후에 호출된다.
   */
  markPendingRelease(byteLength) {
    this._assertValidByteLength(byteLength);
    this.pendingReleaseBytes = Math.min(this.pendingReleaseBytes + byteLength, this.allocatedBytes);
  }
  /**
   * GPU 큐 완료 후 실제 해제 확정.
   * tensorRegistry.ts의 onSubmittedWorkDone 콜백에서 호출된다.
   */
  release(byteLength) {
    this._assertValidByteLength(byteLength);
    this.allocatedBytes = Math.max(0, this.allocatedBytes - byteLength);
    this.pendingReleaseBytes = Math.max(0, this.pendingReleaseBytes - byteLength);
  }
  getUsage() {
    return {
      allocatedBytes: this.allocatedBytes,
      pendingReleaseBytes: this.pendingReleaseBytes,
      effectiveBytes: this.allocatedBytes - this.pendingReleaseBytes,
      hardLimitBytes: this.hardLimitBytes,
      softLimitBytes: this.softLimitBytes
    };
  }
  reset() {
    this.allocatedBytes = 0;
    this.pendingReleaseBytes = 0;
  }
};
var _globalQuotaManager = new QuotaManager();
var MAX_ELEMENTS$1 = 256 * 1024 * 1024;
function allocateBuffer(byteLength, usage) {
  _globalQuotaManager.reserve(byteLength);
  try {
    return getDevice().createBuffer({ size: byteLength, usage });
  } catch (e) {
    _globalQuotaManager.release(byteLength);
    throw e;
  }
}
function writeFloat32Array(buffer, data) {
  getQueue().writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
}
async function readBufferToFloat32Array(buffer, byteLength) {
  const device2 = getDevice();
  const stagingBuffer = device2.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });
  try {
    const commandEncoder = device2.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
    device2.queue.submit([commandEncoder.finish()]);
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    try {
      const arrayBuffer = stagingBuffer.getMappedRange();
      return new Float32Array(arrayBuffer.slice(0));
    } finally {
      stagingBuffer.unmap();
    }
  } finally {
    stagingBuffer.destroy();
  }
}
async function mapBufferAsync$1(buffer, byteLength) {
  const device2 = getDevice();
  const stagingBuffer = device2.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });
  const commandEncoder = device2.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
  device2.queue.submit([commandEncoder.finish()]);
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  return stagingBuffer;
}
function readMappedInto$1(stagingBuffer, outArray) {
  try {
    const arrayBuffer = stagingBuffer.getMappedRange();
    outArray.set(new Float32Array(arrayBuffer));
  } finally {
    stagingBuffer.unmap();
    stagingBuffer.destroy();
  }
}
function freeBuffer(buffer, byteLength) {
  _globalQuotaManager.release(byteLength);
  buffer.destroy();
}
var TensorRegistry = class {
  records = /* @__PURE__ */ new Map();
  nextId = 1;
  register(recordOmitHandle) {
    const handle = `tensor_${this.nextId++}_${Date.now()}`;
    const record = {
      ...recordOmitHandle,
      handle,
      disposed: false,
      createdAt: Date.now()
    };
    this.records.set(handle, record);
    return handle;
  }
  get(handle) {
    const record = this.records.get(handle);
    if (!record) {
      throw new AMEVAForgeDisposedError(`Tensor not found: ${handle}`);
    }
    if (record.disposed) {
      throw new AMEVAForgeDisposedError(`Attempted to access disposed tensor: ${handle}`);
    }
    return record;
  }
  has(handle) {
    const record = this.records.get(handle);
    return record !== void 0 && !record.disposed;
  }
  dispose(handle) {
    const record = this.records.get(handle);
    if (!record || record.disposed)
      return;
    record.disposed = true;
    this.records.delete(handle);
    _globalQuotaManager.markPendingRelease(record.byteLength);
    Promise.resolve().then(function() {
      return device$1;
    }).then(({ getDevice: getDevice2 }) => {
      try {
        getDevice2().queue.onSubmittedWorkDone().then(() => {
          freeBuffer(record.buffer, record.byteLength);
        }).catch(() => {
          freeBuffer(record.buffer, record.byteLength);
        });
      } catch {
        _globalQuotaManager.release(record.byteLength);
      }
    });
  }
  clear() {
    const recordsToFree = Array.from(this.records.values()).filter((r) => !r.disposed);
    this.records.clear();
    if (recordsToFree.length === 0)
      return;
    for (const record of recordsToFree) {
      _globalQuotaManager.markPendingRelease(record.byteLength);
    }
    Promise.resolve().then(function() {
      return device$1;
    }).then(({ getDevice: getDevice2 }) => {
      try {
        getDevice2().queue.onSubmittedWorkDone().then(() => {
          for (const record of recordsToFree) {
            freeBuffer(record.buffer, record.byteLength);
          }
        }).catch(() => {
          for (const record of recordsToFree) {
            freeBuffer(record.buffer, record.byteLength);
          }
        });
      } catch {
        _globalQuotaManager.reset();
      }
    });
  }
};
var _globalRegistry = new TensorRegistry();
var PipelineCache = class {
  cache = /* @__PURE__ */ new Map();
  /**
   * 주어진 key(op명)로 캐시를 조회하거나 새로 컴파일하여 반환한다.
   * @param key - 캐시 키 (op 이름, e.g. "matmul")
   * @param wgslCode - WGSL 셰이더 소스
   */
  getPipeline(key, wgslCode) {
    const cached = this.cache.get(key);
    if (cached)
      return cached;
    const device2 = getDevice();
    const shader = device2.createShaderModule({ code: wgslCode });
    const pipeline = device2.createComputePipeline({
      layout: "auto",
      compute: { module: shader, entryPoint: "main" }
    });
    const entry = { shader, pipeline };
    this.cache.set(key, entry);
    return entry;
  }
  /**
   * L-03 Fix: WebGPU device lost 시 캐시 전체 무효화.
   * resetRuntimeMemory()에서 호출된다.
   */
  clear() {
    this.cache.clear();
  }
  get size() {
    return this.cache.size;
  }
};
var _globalPipelineCache = new PipelineCache();
var MATMUL_WGSL = `
struct Params {
  M: u32,
  N: u32,
  K: u32,
  pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.y;
  let col = global_id.x;

  if (row >= params.M || col >= params.N) {
    return;
  }

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + a[row * params.K + k] * b[k * params.N + col];
  }

  c[row * params.N + col] = sum;
}
`;
var RELU_WGSL = `
struct Params {
  size: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.size) {
    return;
  }
  y[idx] = max(x[idx], 0.0);
}
`;
var ADD_WGSL = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx < params.size) {
    out[idx] = a[idx] + b[idx];
  }
}
`;
var TRANSPOSE_WGSL = `
struct Params {
  M: u32,
  N: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  
  if (row < params.M && col < params.N) {
    let in_idx = row * params.N + col;
    let out_idx = col * params.M + row;
    out[out_idx] = input[in_idx];
  }
}
`;
var MUL_WGSL = `
struct Params {
  size: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> A : array<f32>;
@group(0) @binding(2) var<storage, read> B : array<f32>;
@group(0) @binding(3) var<storage, read_write> C : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let index = global_id.x;
  if (index < params.size) {
    C[index] = A[index] * B[index];
  }
}
`;
var RELU_BACKWARD_WGSL = `
struct Params {
  size: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> X : array<f32>;
@group(0) @binding(2) var<storage, read> gradOutput : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradInput : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let index = global_id.x;
  if (index < params.size) {
    if (X[index] > 0.0) {
      gradInput[index] = gradOutput[index];
    } else {
      gradInput[index] = 0.0;
    }
  }
}
`;
var _pendingStagingBuffers = /* @__PURE__ */ new Map();
function resetRuntimeMemory() {
  _globalRegistry.clear();
  _globalQuotaManager.reset();
  _globalPipelineCache.clear();
  _pendingStagingBuffers.clear();
}
async function init(options) {
  setDeviceLostCallback(() => {
    resetRuntimeMemory();
  });
  await initWebGPU(options);
  const adapter2 = getAdapter();
  if (adapter2) {
    const limits = adapter2.limits;
    const maxBuf = limits.maxBufferSize ?? 2 * 1024 * 1024 * 1024;
    const hardLimit = Math.min(maxBuf * 0.8, 8 * 1024 * 1024 * 1024);
    const softLimit = hardLimit * 0.75;
    _globalQuotaManager.setLimits(Math.floor(hardLimit), Math.floor(softLimit));
    console.info(`[AMEVA] GPU quota set: soft=${(softLimit / 1e9).toFixed(2)}GB, hard=${(hardLimit / 1e9).toFixed(2)}GB`);
  }
}
function getTensorInfo(handle) {
  const record = _globalRegistry.get(handle);
  return {
    handle: record.handle,
    shape: [...record.shape],
    dtype: record.dtype,
    byteLength: record.byteLength,
    disposed: record.disposed
  };
}
function read(handle) {
  const record = _globalRegistry.get(handle);
  return readBufferToFloat32Array(record.buffer, record.byteLength);
}
async function mapBufferAsync(handle) {
  const record = _globalRegistry.get(handle);
  const stagingBuffer = await mapBufferAsync$1(record.buffer, record.byteLength);
  _pendingStagingBuffers.set(handle, stagingBuffer);
}
function readMappedInto(handle, outArray) {
  const stagingBuffer = _pendingStagingBuffers.get(handle);
  if (!stagingBuffer) {
    throw new Error(`[AMEVA] No staged buffer for handle "${handle}". Call mapBufferAsync first.`);
  }
  _pendingStagingBuffers.delete(handle);
  let actualData;
  let bufProxy = null;
  if (outArray && typeof outArray.getBuffer === "function") {
    bufProxy = outArray.getBuffer("f32");
    actualData = bufProxy.data;
  } else {
    actualData = outArray;
  }
  readMappedInto$1(stagingBuffer, actualData);
  if (bufProxy) {
    bufProxy.release();
  }
}
function dispose(handle) {
  _globalRegistry.dispose(handle);
}
var ALLOWED_OPS = /* @__PURE__ */ new Set([
  "upload",
  "load",
  "matmul",
  "relu",
  "add",
  "mul",
  "transpose",
  "relu_backward"
]);
var MAX_SHAPE_DIM = 4;
var MAX_ELEMENTS = 256 * 1024 * 1024;
var MAX_INSTRUCTIONS = 1e4;
function validateInstruction(inst, idx) {
  if (typeof inst !== "object" || inst === null) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: must be an object`);
  }
  const i = inst;
  if (typeof i.op !== "string") {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: op must be a string`);
  }
  if (!ALLOWED_OPS.has(i.op)) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: unknown op "${i.op}"`);
  }
  if (!Number.isSafeInteger(i.id) || i.id < 1) {
    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: id must be a positive safe integer`);
  }
  if (!Array.isArray(i.shape)) {
    throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape must be an array`);
  }
  if (i.shape.length < 1 || i.shape.length > MAX_SHAPE_DIM) {
    throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape rank must be 1\u2013${MAX_SHAPE_DIM}, got ${i.shape.length}`);
  }
  let elements = 1;
  for (const dim of i.shape) {
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape dim must be a positive safe integer, got ${dim}`);
    }
    if (dim > Number.MAX_SAFE_INTEGER / elements) {
      throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape product integer overflow`);
    }
    elements *= dim;
  }
  if (elements > MAX_ELEMENTS) {
    throw new AMEVAForgeShapeError(`Instruction[${idx}]: tensor too large (${elements} elements > ${MAX_ELEMENTS})`);
  }
  return i;
}
function executeGraph(instructionsJson, jsInputs) {
  let rawInstructions;
  try {
    rawInstructions = JSON.parse(instructionsJson);
  } catch {
    throw new AMEVAForgeSecurityError("executeGraph: invalid JSON in instructionsJson");
  }
  if (!Array.isArray(rawInstructions)) {
    throw new AMEVAForgeSecurityError("executeGraph: instructionsJson must be a JSON array");
  }
  if (rawInstructions.length > MAX_INSTRUCTIONS) {
    throw new AMEVAForgeSecurityError(`executeGraph: too many instructions (${rawInstructions.length} > ${MAX_INSTRUCTIONS})`);
  }
  const instructions = rawInstructions.map(validateInstruction);
  let inputs;
  if (jsInputs && typeof jsInputs.toJs === "function") {
    inputs = jsInputs.toJs();
  } else if (Array.isArray(jsInputs)) {
    inputs = jsInputs;
  } else {
    inputs = [];
  }
  const device2 = getDevice();
  const commandEncoder = device2.createCommandEncoder();
  const idToHandle = {};
  const idToBuffer = {};
  let inputIdx = 0;
  const paramsBuffersToDestroy = [];
  for (const inst of instructions) {
    const byteLength = inst.shape.reduce((a, b) => a * b, 1) * 4;
    if (inst.op === "load") {
      const handle2 = inst.handle;
      if (typeof handle2 !== "string") {
        throw new AMEVAForgeSecurityError(`load instruction missing handle`);
      }
      idToHandle[inst.id] = handle2;
      idToBuffer[inst.id] = _globalRegistry.get(handle2).buffer;
      continue;
    }
    if (inst.op === "upload") {
      const rawData = inputs[inputIdx++];
      let actualData;
      let bufProxy = null;
      try {
        if (rawData instanceof Float32Array) {
          actualData = rawData;
        } else if (rawData && typeof rawData.getBuffer === "function") {
          bufProxy = rawData.getBuffer("f32");
          actualData = bufProxy.data;
        } else if (rawData && typeof rawData.toJs === "function") {
          const converted = rawData.toJs();
          actualData = converted instanceof Float32Array ? converted : new Float32Array(converted);
        } else if (rawData?.buffer) {
          actualData = new Float32Array(rawData.buffer, rawData.byteOffset || 0, (rawData.byteLength || rawData.length * 4) / 4);
        } else {
          actualData = Float32Array.from(rawData);
        }
      } catch (e) {
        throw new AMEVAForgeSecurityError(`upload input[${inputIdx - 1}] conversion failed: ${e}, rawData typeof: ${typeof rawData}`);
      }
      const buffer = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      writeFloat32Array(buffer, actualData);
      if (bufProxy)
        bufProxy.release();
      const handle2 = _globalRegistry.register({
        buffer,
        shape: inst.shape,
        dtype: "float32",
        byteLength
      });
      idToHandle[inst.id] = handle2;
      idToBuffer[inst.id] = buffer;
      continue;
    }
    const outBuffer = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const handle = _globalRegistry.register({
      buffer: outBuffer,
      shape: inst.shape,
      dtype: "float32",
      byteLength
    });
    idToHandle[inst.id] = handle;
    idToBuffer[inst.id] = outBuffer;
    const passEncoder = commandEncoder.beginComputePass();
    const paramsBuffer = device2.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    paramsBuffersToDestroy.push(paramsBuffer);
    let wgslCode = "";
    let dispatchX = 1, dispatchY = 1;
    if (inst.op === "matmul") {
      const [M, N, K] = inst.params;
      wgslCode = MATMUL_WGSL;
      device2.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, N, K, 0]));
      dispatchX = Math.ceil(N / 8);
      dispatchY = Math.ceil(M / 8);
    } else if (inst.op === "transpose") {
      const [M, N] = inst.params;
      wgslCode = TRANSPOSE_WGSL;
      device2.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, N, 0, 0]));
      dispatchX = Math.ceil(M / 8);
      dispatchY = Math.ceil(N / 8);
    } else {
      const numElements = byteLength / 4;
      wgslCode = inst.op === "relu" ? RELU_WGSL : inst.op === "add" ? ADD_WGSL : inst.op === "mul" ? MUL_WGSL : inst.op === "relu_backward" ? RELU_BACKWARD_WGSL : "";
      if (!wgslCode) {
        throw new AMEVAForgeSecurityError(`Unknown op "${inst.op}"`);
      }
      device2.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, 0, 0, 0]));
      dispatchX = Math.ceil(numElements / 64);
    }
    const { pipeline } = _globalPipelineCache.getPipeline(inst.op, wgslCode);
    passEncoder.setPipeline(pipeline);
    const bindGroupEntries = [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } }
    ];
    if (inst.in.length > 1) {
      bindGroupEntries.push({ binding: 2, resource: { buffer: idToBuffer[inst.in[1]] } });
      bindGroupEntries.push({ binding: 3, resource: { buffer: outBuffer } });
    } else {
      bindGroupEntries.push({ binding: 2, resource: { buffer: outBuffer } });
    }
    const bindGroup = device2.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries
    });
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY);
    passEncoder.end();
  }
  device2.queue.submit([commandEncoder.finish()]);
  if (paramsBuffersToDestroy.length > 0) {
    device2.queue.onSubmittedWorkDone().then(() => {
      paramsBuffersToDestroy.forEach((b) => b.destroy());
    });
  }
  return idToHandle;
}
function disposeBatch(handles) {
  for (const handle of handles) {
    if (handle && _globalRegistry.has(handle)) {
      dispose(handle);
    }
  }
}
function registerPyodideBridge() {
  const api = {
    init,
    read,
    dispose,
    getTensorInfo,
    mapBufferAsync,
    readMappedInto,
    executeGraph,
    disposeBatch
  };
  globalThis.amevaForge = api;
  return api;
}

// app.js
registerPyodideBridge();
async function run() {
  const logDiv = document.getElementById("log");
  const log = (msg) => {
    console.log(msg);
    logDiv.innerHTML += msg + "<br>";
  };
  log("Initializing Pyodide...");
  const pyodide = await loadPyodide();
  log("Loading micropip...");
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  log("Installing forge wheel...");
  await micropip.install("/forge-0.1.0-py3-none-any.whl");
  log("Fetching Python script...");
  const scriptRes = await fetch("/script.py");
  const scriptCode = await scriptRes.text();
  log("Executing Python E2E Test...");
  try {
    pyodide.setStdout({ batched: (msg) => log("[Python] " + msg) });
    await pyodide.runPythonAsync(scriptCode);
    log("E2E_SUCCESS");
  } catch (err) {
    log("E2E_ERROR: " + err);
  }
}
run();
