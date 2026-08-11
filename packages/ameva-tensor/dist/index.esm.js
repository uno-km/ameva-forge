class AMEVATensorError extends Error {
    constructor(message) {
        super(message);
        this.name = new.target.name;
        Object.setPrototypeOf(this, new.target.prototype);
    }
}
class AMEVATensorShapeError extends AMEVATensorError {
}
class AMEVATensorDTypeError extends AMEVATensorError {
}
class AMEVATensorDeviceError extends AMEVATensorError {
}
class AMEVATensorDisposedError extends AMEVATensorError {
}
class AMEVATensorQuotaExceededError extends AMEVATensorError {
}
class AMEVATensorWebGPUUnavailableError extends AMEVATensorError {
}
class AMEVATensorSecurityError extends AMEVATensorError {
}

/**
 * device.ts — WebGPU 싱글톤 디바이스 래퍼
 *
 * H-04 Fix: getAdapter() export 추가 → gpuCore.ts에서 adapter.limits 조회 가능
 * L-03 Fix: device lost 시 onDeviceLostCallback을 통해 pipelineCache도 무효화
 */
function _safeLog$1(msg) {
    try {
        if (typeof globalThis.log === 'function') {
            globalThis.log(msg, 'system');
        }
    }
    catch (e) { }
}
let device = null;
let adapter = null;
let onDeviceLostCallback = null;
async function initWebGPU(options) {
    _safeLog$1(`[device.ts] initWebGPU started. current device=${device ? 'SET' : 'NULL'}`);
    if (device)
        return;
    if (typeof navigator === "undefined" || !navigator.gpu) {
        throw new AMEVATensorWebGPUUnavailableError("WebGPU is not available in this environment. " +
            "Ensure you are running in a supported browser with WebGPU enabled.");
    }
    adapter = await navigator.gpu.requestAdapter(options);
    if (!adapter) {
        throw new AMEVATensorWebGPUUnavailableError("Failed to request a WebGPU adapter. " +
            "Your GPU may not support WebGPU, or the browser has disabled it.");
    }
    const requiredLimits = {};
    if (adapter.limits) {
        requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
        requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
    }
    device = await adapter.requestDevice({ requiredLimits });
    globalThis.__AMEVA_DEVICE__ = device;
    _safeLog$1(`[device.ts] initWebGPU finished. device successfully created.`);
    device.lost.then((info) => {
        const msg = `[AMEVA] WebGPU Device Lost: ${info.message} (reason: ${info.reason})`;
        console.error(msg);
        _safeLog$1(msg);
        device = null;
        globalThis.__AMEVA_DEVICE__ = null;
        adapter = null;
        if (onDeviceLostCallback) {
            onDeviceLostCallback();
        }
    });
}
function getDevice() {
    const globalDev = globalThis.__AMEVA_DEVICE__;
    _safeLog$1(`[device.ts] getDevice called. local device=${device ? 'SET' : 'NULL'}, globalDev=${globalDev ? 'SET' : 'NULL'}`);
    if (!device) {
        const globalExists = typeof globalThis.amevaTensor !== "undefined";
        throw new AMEVATensorDeviceError(`WebGPU device is not initialized. (device is ${device}, __AMEVA_DEVICE__ exists: ${!!globalDev}, globalThis.amevaTensor exists: ${globalExists}). Call await init() first.`);
    }
    return device;
}
/** H-04: adapter.limits 접근용 */
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

function assertWasmRange(offset, byteLength, wasmByteLength) {
    if (!Number.isSafeInteger(offset) || offset < 0) {
        throw new AMEVATensorSecurityError("Invalid offset: must be a non-negative safe integer.");
    }
    if (!Number.isSafeInteger(byteLength) || byteLength < 0) {
        throw new AMEVATensorSecurityError("Invalid byteLength: must be a non-negative safe integer.");
    }
    if (!Number.isSafeInteger(wasmByteLength) || wasmByteLength < 0) {
        throw new AMEVATensorSecurityError("Invalid wasmByteLength: must be a non-negative safe integer.");
    }
    if (offset > wasmByteLength || byteLength > wasmByteLength - offset) {
        throw new AMEVATensorSecurityError("WASM memory range out of bounds");
    }
}

/**
 * quota.ts — VRAM 할당 쿼터 관리자
 *
 * C-06 Fix: quota release 타이밍 불일치 해결 — markPendingRelease + release 2단계.
 * H-04 Fix: setLimits()로 런타임에 동적 쿼터 설정 가능.
 * NH-04 Fix: markPendingRelease() 이중 dispose 시 카운터 음수 방지.
 */
class QuotaManager {
    /** 현재 할당된 총 바이트 (pending 포함) */
    allocatedBytes = 0;
    /** GPU 큐 대기 중인 해제 바이트 (실제로는 아직 GPU 점유 중) */
    pendingReleaseBytes = 0;
    hardLimitBytes;
    softLimitBytes;
    constructor(hardLimitBytes = 1 * 1024 * 1024 * 1024, // H-NEW-02: 기본 1GB (보수적)
    softLimitBytes = 768 * 1024 * 1024 // 768MB
    ) {
        this.hardLimitBytes = hardLimitBytes;
        this.softLimitBytes = softLimitBytes;
    }
    /** H-04: 런타임에 동적으로 쿼터 상한 재설정 */
    setLimits(hardLimitBytes, softLimitBytes) {
        if (!Number.isSafeInteger(hardLimitBytes) || hardLimitBytes <= 0) {
            throw new AMEVATensorQuotaExceededError(`Invalid hard limit: ${hardLimitBytes}`);
        }
        if (!Number.isSafeInteger(softLimitBytes) || softLimitBytes <= 0) {
            throw new AMEVATensorQuotaExceededError(`Invalid soft limit: ${softLimitBytes}`);
        }
        if (softLimitBytes > hardLimitBytes) {
            throw new AMEVATensorQuotaExceededError("softLimitBytes must be <= hardLimitBytes");
        }
        this.hardLimitBytes = hardLimitBytes;
        this.softLimitBytes = softLimitBytes;
    }
    reserve(byteLength) {
        if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
            throw new AMEVATensorQuotaExceededError(`Invalid allocation size: ${byteLength}`);
        }
        // C-06: 여유 공간 = hardLimit - allocatedBytes (pending 포함, 보수적)
        if (byteLength > this.hardLimitBytes - this.allocatedBytes) {
            throw new AMEVATensorQuotaExceededError(`Quota Exceeded: Cannot allocate ${byteLength} bytes. ` +
                `Current: ${this.allocatedBytes} (${this.pendingReleaseBytes} pending release), ` +
                `Limit: ${this.hardLimitBytes}`);
        }
        this.allocatedBytes += byteLength;
        if (this.allocatedBytes - this.pendingReleaseBytes > this.softLimitBytes) {
            console.warn(`[AMEVA] VRAM soft quota exceeded: ` +
                `${((this.allocatedBytes - this.pendingReleaseBytes) / 1e9).toFixed(2)}GB / ` +
                `${(this.softLimitBytes / 1e9).toFixed(2)}GB`);
        }
    }
    /**
     * C-06: dispose() 호출 시 즉시 "해제 예정"으로 표시.
     * NH-04 Fix: 이중 dispose 방지 — allocatedBytes 기준으로 클램핑하되
     *   pendingReleaseBytes가 allocatedBytes를 초과하지 않도록 보장.
     */
    markPendingRelease(byteLength) {
        if (!Number.isSafeInteger(byteLength) || byteLength <= 0)
            return;
        // NH-04: 이미 pending으로 마킹된 bytes + 새로운 bytes가 allocated를 초과하지 않도록
        const newPending = this.pendingReleaseBytes + byteLength;
        this.pendingReleaseBytes = Math.min(newPending, this.allocatedBytes);
    }
    /**
     * GPU 큐 완료 후 실제 해제 확정.
     * NH-04 Fix: 음수 방지를 위해 Math.max(0, ...) 적용.
     */
    release(byteLength) {
        if (!Number.isSafeInteger(byteLength) || byteLength <= 0)
            return;
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
}
const _globalQuotaManager = new QuotaManager();

/**
 * shaderGuard.ts — WGSL 셰이더 보안 가드
 *
 * H-07 Fix: 화이트리스트에 모든 구현된 op 추가.
 * NH-07 Fix: 이 파일의 assertAllowedKernelName()을 graphExecutor.ts와 gpuCore.ts에서
 *   실제로 import하여 사용한다 (이전에는 데드 코드였음).
 */
/** 셰이더 식별자 (함수명, 변수명 등) 유효성 검사 */
function assertSafeShaderIdentifier(identifier) {
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(identifier)) {
        throw new AMEVATensorSecurityError(`Invalid shader identifier: "${identifier}". Only alphanumeric and underscore allowed.`);
    }
}
/** 셰이더에 삽입되는 상수값 유효성 검사 */
function assertAllowedShaderConstant(value) {
    if (!Number.isFinite(value)) {
        throw new AMEVATensorSecurityError(`Invalid shader constant: ${value}. Must be a finite number.`);
    }
}
/**
 * 동적 문자열 보간이 셰이더 소스에 포함되지 않았는지 검사.
 * 템플릿 리터럴 인젝션 공격을 차단한다.
 */
function assertStaticShaderSourceOnly(source) {
    if (source.includes("${") || source.includes("`")) {
        throw new AMEVATensorSecurityError("Dynamic shader source interpolation is forbidden. Use uniform buffers for runtime values.");
    }
}
/**
 * H-07/NH-07 Fix: 모든 구현된 커널 이름을 화이트리스트에 포함.
 * graphExecutor.ts의 ALLOWED_OPS와 반드시 동기화 유지.
 * 이 함수는 gpuCore.ts와 graphExecutor.ts에서 실제로 호출된다.
 */
const ALLOWED_KERNEL_NAMES = new Set([
    "matmul",
    "relu",
    "relu_backward",
    "add",
    "mul",
    "transpose",
]);
function assertAllowedKernelName(name) {
    if (!ALLOWED_KERNEL_NAMES.has(name)) {
        throw new AMEVATensorSecurityError(`Unknown kernel name: "${name}". Allowed: ${[...ALLOWED_KERNEL_NAMES].join(", ")}`);
    }
}
/** 허용된 커널 이름 목록 반환 (외부 동기화 용도) */
function getAllowedKernelNames() {
    return ALLOWED_KERNEL_NAMES;
}

/** dtype별 바이트 크기 */
const BYTES_PER_ELEMENT = {
    "float32": 4,
    // float16: 2 — 셰이더 구현 완료 후 추가 예정
    // int32: 4 — 셰이더 구현 완료 후 추가 예정
};
const MAX_ELEMENTS$1 = 256 * 1024 * 1024;
const MAX_RANK = 8; // NM-06: 스칼라(rank 0) 포함하여 0~8까지 허용
/**
 * validateShape — 텐서 shape의 유효성을 검증하고 총 원소 수를 반환한다.
 *
 * M-01 Fix: dtype별 바이트 크기를 BYTES_PER_ELEMENT 맵으로 정확히 계산.
 * NM-06 Fix: rank 0 스칼라 텐서 허용 (PyTorch/JAX/TF 표준).
 *   rank 0 = shape=[], elements=1, byteLength=4 (단일 float32 스칼라)
 */
function validateShape(shape, dtype, expectedByteLength) {
    if (!Array.isArray(shape)) {
        throw new AMEVATensorShapeError("Shape must be an array.");
    }
    // NM-06 Fix: rank 0 (shape=[]) 허용 — 스칼라 텐서
    if (shape.length > MAX_RANK) {
        throw new AMEVATensorShapeError(`Shape rank must be between 0 and ${MAX_RANK}, got ${shape.length}.`);
    }
    let elements = 1;
    for (const dim of shape) {
        if (!Number.isSafeInteger(dim) || dim <= 0) {
            throw new AMEVATensorShapeError(`Each shape dimension must be a positive safe integer, got: ${dim}`);
        }
        if (dim > Number.MAX_SAFE_INTEGER / elements) {
            throw new AMEVATensorShapeError("Shape product overflows safe integer limit.");
        }
        elements *= dim;
    }
    if (elements > MAX_ELEMENTS$1) {
        throw new AMEVATensorShapeError(`Tensor size exceeds max elements limit: ${elements} > ${MAX_ELEMENTS$1}`);
    }
    if (expectedByteLength !== undefined) {
        const bytesPerElement = BYTES_PER_ELEMENT[dtype];
        if (bytesPerElement === undefined) {
            throw new AMEVATensorDTypeError(`Unsupported dtype for byte size calculation: "${dtype}". ` +
                `Supported: ${Object.keys(BYTES_PER_ELEMENT).join(', ')}`);
        }
        const calculatedBytes = elements * bytesPerElement;
        if (calculatedBytes !== expectedByteLength) {
            throw new AMEVATensorShapeError(`Shape/data size mismatch: shape ${JSON.stringify(shape)} (${dtype}) ` +
                `implies ${calculatedBytes} bytes, but data is ${expectedByteLength} bytes.`);
        }
    }
    return elements;
}

function validateDType(dtype) {
    if (dtype !== "float32") {
        throw new AMEVATensorDTypeError(`Unsupported dtype: ${dtype}. Only float32 is supported.`);
    }
}

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
function allocateBuffer(byteLength, usage) {
    _globalQuotaManager.reserve(byteLength);
    try {
        return getDevice().createBuffer({ size: byteLength, usage });
    }
    catch (e) {
        _globalQuotaManager.release(byteLength);
        throw e;
    }
}
function writeFloat32Array(buffer, data) {
    getQueue().writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
}
/** 전통적인 full-copy readback */
async function readBufferToFloat32Array(buffer, byteLength) {
    const device = getDevice();
    const stagingBuffer = device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    try {
        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
        device.queue.submit([commandEncoder.finish()]);
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        try {
            const arrayBuffer = stagingBuffer.getMappedRange();
            return new Float32Array(arrayBuffer.slice(0));
        }
        finally {
            stagingBuffer.unmap();
        }
    }
    finally {
        stagingBuffer.destroy();
    }
}
/**
 * C-05 Fix: mapBufferAsync는 staging buffer를 직접 반환한다.
 * 전역 Map 없이 호출자가 staging buffer 참조를 들고 있어 동시 readback 안전.
 *
 * GPU→CPU 전송 과정:
 * 1. VRAM의 compute buffer를 RAM의 staging buffer로 복사 (GPU copyBufferToBuffer)
 * 2. staging buffer를 WASM에서 읽을 수 있도록 맵핑 (mapAsync)
 * 이 함수는 staging buffer를 반환하며, readMappedInto()에서 최종 읽기를 수행.
 */
async function mapBufferAsync$1(buffer, byteLength) {
    const device = getDevice();
    const stagingBuffer = device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
    device.queue.submit([commandEncoder.finish()]);
    // GPU 연산 완료 + 메모리 맵핑 완료까지 대기
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    return stagingBuffer;
}
/**
 * C-05 Fix: staging buffer를 인자로 받아 데이터 읽기 수행 후 소각.
 *
 * NH-05 Note: 이 함수는 1번의 메모리 copy를 수행한다.
 * outArray.set()은 staging buffer → WASM 힙으로의 copy다.
 * WebGPU 스펙상 GPU 메모리와 WASM 힙의 직접 공유(진짜 Zero-Copy)는 불가능하다.
 * 이 1번의 copy는 구조적으로 불가피한 최소값이다.
 */
function readMappedInto$1(stagingBuffer, outArray) {
    try {
        const arrayBuffer = stagingBuffer.getMappedRange();
        // GPU staging → WASM 힙 copy (1번, 불가피)
        outArray.set(new Float32Array(arrayBuffer));
    }
    finally {
        stagingBuffer.unmap();
        stagingBuffer.destroy();
    }
}
function freeBuffer(buffer, byteLength) {
    _globalQuotaManager.release(byteLength);
    buffer.destroy();
}

/**
 * tensorRegistry.ts — GPU 텐서 생명주기 레지스트리
 *
 * C-06 Fix: dispose() 시 _globalQuotaManager.markPendingRelease() 즉시 호출.
 * NC-07 Fix: dynamic import() 제거 → 정적 import 사용 + device.destroy() 보장.
 * NL-03 Fix: Date.now() 제거 → 단조증가 ID만 사용 (타이밍 정보 노출 방지).
 */
class TensorRegistry {
    records = new Map();
    nextId = 1;
    register(recordOmitHandle) {
        // NL-03 Fix: Date.now() 제거, 단순 카운터만 사용
        const handle = `tensor_${this.nextId++}`;
        const record = {
            ...recordOmitHandle,
            handle,
            disposed: false,
            createdAt: this.nextId - 1 // NL-03 Fix: monotonic counter, not timestamp
        };
        this.records.set(handle, record);
        return handle;
    }
    get(handle) {
        const record = this.records.get(handle);
        if (!record) {
            throw new AMEVATensorDisposedError(`Tensor not found: ${handle}`);
        }
        if (record.disposed) {
            throw new AMEVATensorDisposedError(`Attempted to access disposed tensor: ${handle}`);
        }
        return record;
    }
    has(handle) {
        const record = this.records.get(handle);
        return record !== undefined && !record.disposed;
    }
    dispose(handle) {
        const record = this.records.get(handle);
        if (!record || record.disposed)
            return;
        record.disposed = true;
        this.records.delete(handle);
        // C-06 Fix: 즉시 "해제 예약" 표시
        _globalQuotaManager.markPendingRelease(record.byteLength);
        // NC-07 Fix: 정적 import된 getDevice() 사용 (dynamic import 제거)
        try {
            const device = getDevice();
            device.queue.onSubmittedWorkDone().then(() => {
                freeBuffer(record.buffer, record.byteLength);
            }).catch(() => {
                // GPU 큐 실패 → 즉시 소각
                _safeDestroyBuffer(record);
            });
        }
        catch {
            // device가 없거나 lost → 즉시 quota 해제 + buffer 소각
            _safeDestroyBuffer(record);
        }
    }
    clear() {
        const recordsToFree = Array.from(this.records.values()).filter(r => !r.disposed);
        this.records.clear();
        if (recordsToFree.length === 0)
            return;
        for (const record of recordsToFree) {
            _globalQuotaManager.markPendingRelease(record.byteLength);
        }
        try {
            const device = getDevice();
            device.queue.onSubmittedWorkDone().then(() => {
                for (const record of recordsToFree) {
                    freeBuffer(record.buffer, record.byteLength);
                }
            }).catch(() => {
                for (const record of recordsToFree) {
                    _safeDestroyBuffer(record);
                }
            });
        }
        catch {
            // device already lost
            for (const record of recordsToFree) {
                _safeDestroyBuffer(record);
            }
            _globalQuotaManager.reset();
        }
    }
}
/**
 * NC-07 Fix: device lost 상황에서도 buffer.destroy()를 보장하고 quota를 해제.
 */
function _safeDestroyBuffer(record) {
    try {
        record.buffer.destroy();
    }
    catch {
        // buffer가 이미 destroyed
    }
    _globalQuotaManager.release(record.byteLength);
}
const _globalRegistry = new TensorRegistry();

/**
 * pipelineCache.ts — WGSL 컴파일 파이프라인 캐시
 *
 * L-03 Fix: clear() 메서드를 통해 device lost 시 캐시 무효화.
 * NL-02 Fix: 캐시 키에 WGSL 해시를 포함하여 동일 op명으로 다른 WGSL 지원.
 */
/**
 * NL-02 Fix: 간단한 문자열 해시 함수 (djb2 변형).
 * 같은 op명으로 다른 WGSL 코드가 전달될 때 캐시 충돌 방지.
 */
function hashString(str) {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) + hash) ^ str.charCodeAt(i);
        hash = hash & hash; // 32bit integer
    }
    return (hash >>> 0).toString(16);
}
class PipelineCache {
    cache = new Map();
    /**
     * 주어진 key(op명)와 wgslCode 해시로 캐시를 조회하거나 새로 컴파일하여 반환한다.
     * NL-02 Fix: 캐시 키 = `${key}:${hashString(wgslCode)}`
     */
    getPipeline(key, wgslCode) {
        const cacheKey = `${key}:${hashString(wgslCode)}`;
        const cached = this.cache.get(cacheKey);
        if (cached)
            return cached;
        const device = getDevice();
        const shader = device.createShaderModule({ code: wgslCode });
        const pipeline = device.createComputePipeline({
            layout: "auto",
            compute: { module: shader, entryPoint: "main" },
        });
        const entry = { shader, pipeline };
        this.cache.set(cacheKey, entry);
        return entry;
    }
    /**
     * H-NEW-08: 비동기 파이프라인 사전 컴파일 (UI freeze 방지).
     * init() 직후 호출하여 첫 연산 시 동기 컴파일 블로킹을 방지한다.
     */
    async warmup(entries) {
        const device = getDevice();
        const promises = entries
            .filter(e => !this.cache.has(`${e.key}:${hashString(e.wgslCode)}`))
            .map(async (e) => {
            const cacheKey = `${e.key}:${hashString(e.wgslCode)}`;
            const shader = device.createShaderModule({ code: e.wgslCode });
            const pipeline = await device.createComputePipelineAsync({
                layout: "auto",
                compute: { module: shader, entryPoint: "main" },
            });
            this.cache.set(cacheKey, { shader, pipeline });
        });
        await Promise.all(promises);
    }
    /**
     * L-03 Fix: WebGPU device lost 시 캐시 전체 무효화.
     */
    clear() {
        this.cache.clear();
    }
    get size() {
        return this.cache.size;
    }
}
const _globalPipelineCache = new PipelineCache();

const MATMUL_WGSL = `
struct Params {
  M: u32,
  N: u32,
  K: u32,
  offsetY: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.y + params.offsetY;
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

const RELU_WGSL = `
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

const ADD_WGSL = `
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

const TRANSPOSE_WGSL = `
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

const MUL_WGSL = `
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

const RELU_BACKWARD_WGSL = `
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
/**
 * 커널 레지스트리: 새 커널 추가 시 import 1줄 + 여기 1줄만 추가하면
 * warmupKernels(), graphExecutor 모두 자동으로 반영된다.
 */
const KERNEL_REGISTRY = new Map([
    ['matmul', MATMUL_WGSL],
    ['relu', RELU_WGSL],
    ['add', ADD_WGSL],
    ['mul', MUL_WGSL],
    ['transpose', TRANSPOSE_WGSL],
    ['relu_backward', RELU_BACKWARD_WGSL],
]);
// ── 핸들별 staging buffer 관리 (C-05) ──
const _pendingStagingBuffers = new Map();
function resetRuntimeMemory() {
    _globalRegistry.clear();
    _globalQuotaManager.reset();
    _globalPipelineCache.clear(); // L-03 Fix: device lost 시 파이프라인 캐시도 무효화
    // L-NEW-02: 미처리된 staging buffer도 소각
    for (const [, buf] of _pendingStagingBuffers) {
        try {
            buf.unmap();
        }
        catch { /* already unmapped */ }
        try {
            buf.destroy();
        }
        catch { /* already destroyed */ }
    }
    _pendingStagingBuffers.clear();
}
/**
 * NH-03 Fix: 초기화 시 adapter.limits에서 GPU 정보를 조회.
 * maxBufferSize는 단일 버퍼 최대 크기이고 VRAM 총 용량이 아님.
 * maxStorageBufferBindingSize가 실제 단일 바인딩에서 사용 가능한 최대 크기.
 * 총 VRAM 쿼터는 사용자 설정 + 어댑터 힌트로 보수적으로 설정.
 */
function _safeLog(msg) {
    try {
        if (typeof globalThis.log === 'function') {
            globalThis.log(msg, 'system');
        }
    }
    catch (e) { }
}
async function init(options) {
    _safeLog(`[gpuCore.ts] init started`);
    setDeviceLostCallback(() => {
        resetRuntimeMemory();
    });
    try {
        _safeLog(`[gpuCore.ts] calling initWebGPU...`);
        await initWebGPU(options);
        _safeLog(`[gpuCore.ts] initWebGPU finished`);
    }
    catch (e) {
        _safeLog(`[gpuCore.ts] initWebGPU threw error: ${e.message}`);
        throw e;
    }
    // NH-03: 실제 GPU 제한 조회 후 쿼터 조정
    const adapter = getAdapter();
    if (adapter) {
        const limits = adapter.limits;
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
        console.info(`[AMEVA] GPU quota set: soft=${(softLimit / 1e9).toFixed(2)}GB, ` +
            `hard=${(hardLimit / 1e9).toFixed(2)}GB ` +
            `(maxStorageBindingSize=${(maxBinding / 1e9).toFixed(2)}GB)`);
    }
    // H-NEW-08: 비동기 파이프라인 사전 컴파일
    await warmupKernels();
}
/**
 * 모든 커널 파이프라인을 비동기로 사전 컴파일한다.
 * KERNEL_REGISTRY를 동적으로 순회하므로, 새 커널 추가 시 여기를 수정할 필요 없다.
 */
async function warmupKernels() {
    const entries = Array.from(KERNEL_REGISTRY.entries()).map(([key, wgslCode]) => ({ key, wgslCode }));
    await _globalPipelineCache.warmup(entries);
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
/**
 * C-05 Fix: staging buffer를 핸들 키로 _pendingStagingBuffers에 저장.
 */
async function mapBufferAsync(handle) {
    const record = _globalRegistry.get(handle);
    const stagingBuffer = await mapBufferAsync$1(record.buffer, record.byteLength);
    _pendingStagingBuffers.set(handle, stagingBuffer);
}
/**
 * C-05 Fix: 핸들 키로 staging buffer를 조회하여 읽기.
 */
function readMappedInto(handle, outArray) {
    const stagingBuffer = _pendingStagingBuffers.get(handle);
    if (!stagingBuffer) {
        throw new Error(`[AMEVA] No staged buffer for handle "${handle}". Call mapBufferAsync first.`);
    }
    _pendingStagingBuffers.delete(handle);
    let bufProxy = null;
    try {
        let actualData;
        if (outArray && typeof outArray.getBuffer === 'function') {
            bufProxy = outArray.getBuffer("f32");
            actualData = bufProxy.data;
        }
        else {
            actualData = outArray;
        }
        readMappedInto$1(stagingBuffer, actualData);
    }
    finally {
        // H-NEW-06: bufProxy.release() 실패 시에도 리소스 정리 보장
        if (bufProxy) {
            try {
                bufProxy.release();
            }
            catch { /* ignore */ }
        }
    }
}
function dispose(handle) {
    _globalRegistry.dispose(handle);
}
function dispatchKernel(opts) {
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
    const entries = [
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
function random(shape, dtype = "float32") {
    validateDType(dtype);
    const elements = validateShape(shape, dtype);
    const data = new Float32Array(elements);
    for (let i = 0; i < elements; i++)
        data[i] = Math.random();
    const byteLength = elements * 4;
    const buffer = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    writeFloat32Array(buffer, data);
    return _globalRegistry.register({ buffer, shape, dtype, byteLength });
}
function uploadFloat32Array(data, shape) {
    let actualData;
    let bufProxy = null;
    if (data && typeof data.getBuffer === 'function') {
        bufProxy = data.getBuffer("f32");
        actualData = bufProxy.data;
    }
    else {
        actualData = data;
    }
    const elements = validateShape(shape, "float32", actualData.byteLength);
    const byteLength = elements * 4;
    const buffer = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    writeFloat32Array(buffer, actualData);
    if (bufProxy)
        bufProxy.release();
    return _globalRegistry.register({ buffer, shape, dtype: "float32", byteLength });
}
function matmul(handleA, handleB) {
    const a = _globalRegistry.get(handleA);
    const b = _globalRegistry.get(handleB);
    if (a.shape.length !== 2 || b.shape.length !== 2)
        throw new AMEVATensorShapeError("Matmul requires 2D tensors");
    if (a.dtype !== "float32" || b.dtype !== "float32")
        throw new AMEVATensorDTypeError("Matmul requires float32 tensors");
    const M = a.shape[0], K = a.shape[1], K2 = b.shape[0], N = b.shape[1];
    if (K !== K2)
        throw new AMEVATensorShapeError(`Inner dim mismatch: ${K} != ${K2}`);
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
function relu(handle) {
    const x = _globalRegistry.get(handle);
    if (x.dtype !== "float32")
        throw new AMEVATensorDTypeError("ReLU requires float32");
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
function add(handleA, handleB) {
    const a = _globalRegistry.get(handleA);
    const b = _globalRegistry.get(handleB);
    if (a.byteLength !== b.byteLength)
        throw new AMEVATensorShapeError("Add requires tensors of the same shape");
    if (a.dtype !== "float32" || b.dtype !== "float32")
        throw new AMEVATensorDTypeError("Add requires float32");
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
function mul(handleA, handleB) {
    const a = _globalRegistry.get(handleA);
    const b = _globalRegistry.get(handleB);
    if (a.byteLength !== b.byteLength)
        throw new AMEVATensorShapeError("Mul requires tensors of the same shape");
    if (a.dtype !== "float32" || b.dtype !== "float32")
        throw new AMEVATensorDTypeError("Mul requires float32");
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
function transpose(handle) {
    const x = _globalRegistry.get(handle);
    if (x.shape.length !== 2)
        throw new AMEVATensorShapeError("Transpose requires 2D tensors");
    if (x.dtype !== "float32")
        throw new AMEVATensorDTypeError("Transpose requires float32");
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
function relu_backward(handleX, handleGrad) {
    const x = _globalRegistry.get(handleX);
    const grad = _globalRegistry.get(handleGrad);
    if (x.byteLength !== grad.byteLength)
        throw new AMEVATensorShapeError("ReLU backward: shape mismatch");
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
/**
 * C-04: 단일 instruction의 무결성을 검증한다.
 */
function validateInstruction(inst, idx) {
    if (typeof inst !== 'object' || inst === null) {
        throw new AMEVATensorSecurityError(`Instruction[${idx}]: must be an object`);
    }
    const i = inst;
    if (typeof i.op !== 'string') {
        throw new AMEVATensorSecurityError(`Instruction[${idx}]: op must be a string`);
    }
    if (!ALLOWED_OPS.has(i.op)) {
        throw new AMEVATensorSecurityError(`Instruction[${idx}]: unknown op "${i.op}"`);
    }
    if (!Number.isSafeInteger(i.id) || i.id < 1) {
        throw new AMEVATensorSecurityError(`Instruction[${idx}]: id must be a positive safe integer`);
    }
    if (!Array.isArray(i.shape)) {
        throw new AMEVATensorShapeError(`Instruction[${idx}]: shape must be an array`);
    }
    // NM-06: rank 0 허용 (스칼라)
    if (i.shape.length > MAX_SHAPE_DIM) {
        throw new AMEVATensorShapeError(`Instruction[${idx}]: shape rank must be 0–${MAX_SHAPE_DIM}, got ${i.shape.length}`);
    }
    let elements = 1;
    for (const dim of i.shape) {
        if (!Number.isSafeInteger(dim) || dim <= 0) {
            throw new AMEVATensorShapeError(`Instruction[${idx}]: shape dim must be a positive safe integer, got ${dim}`);
        }
        if (dim > Number.MAX_SAFE_INTEGER / elements) {
            throw new AMEVATensorShapeError(`Instruction[${idx}]: shape product integer overflow`);
        }
        elements *= dim;
    }
    if (elements > MAX_ELEMENTS) {
        throw new AMEVATensorShapeError(`Instruction[${idx}]: tensor too large (${elements} elements > ${MAX_ELEMENTS})`);
    }
    // NC-06: in 필드가 있으면 배열인지 확인
    if (i.in !== undefined && !Array.isArray(i.in)) {
        throw new AMEVATensorSecurityError(`Instruction[${idx}]: 'in' field must be an array`);
    }
    return i;
}
/**
 * executeGraph — Python 레이지 그래프를 단일 FFI 호출로 GPU에 실행한다.
 */
function executeGraph(instructionsJson, jsInputs) {
    // --- C-04: JSON 파싱 및 전체 검증 ---
    let rawInstructions;
    try {
        rawInstructions = JSON.parse(instructionsJson);
    }
    catch {
        throw new AMEVATensorSecurityError("executeGraph: invalid JSON in instructionsJson");
    }
    if (!Array.isArray(rawInstructions)) {
        throw new AMEVATensorSecurityError("executeGraph: instructionsJson must be a JSON array");
    }
    if (rawInstructions.length > MAX_INSTRUCTIONS) {
        throw new AMEVATensorSecurityError(`executeGraph: too many instructions (${rawInstructions.length} > ${MAX_INSTRUCTIONS})`);
    }
    const instructions = rawInstructions.map(validateInstruction);
    // inputs 배열 추출 (Pyodide PyProxy 또는 JS 배열)
    let inputs;
    if (jsInputs && typeof jsInputs.toJs === 'function') {
        inputs = jsInputs.toJs();
    }
    else if (Array.isArray(jsInputs)) {
        inputs = jsInputs;
    }
    else {
        inputs = [];
    }
    const device = getDevice();
    // NM-05 Fix: 전체 그래프에 대해 error scope 설정
    device.pushErrorScope('validation');
    let commandEncoder = device.createCommandEncoder();
    let opsInCurrentBatch = 0;
    let workloadElements = 0;
    const idToHandle = {};
    const idToBuffer = {};
    let inputIdx = 0;
    const paramsBuffersToDestroy = [];
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
            let actualData;
            let bufProxy = null;
            if (rawData && typeof rawData.getBuffer === 'function') {
                bufProxy = rawData.getBuffer("f32");
                actualData = bufProxy.data;
            }
            else if (rawData instanceof Float32Array) {
                actualData = rawData;
            }
            else if (rawData && typeof rawData.toJs === 'function') {
                const converted = rawData.toJs();
                actualData = converted instanceof Float32Array ? converted : new Float32Array(converted);
            }
            else {
                throw new AMEVATensorSecurityError(`upload input[${inputIdx - 1}] is not a Float32Array or convertible type`);
            }
            const buffer = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
            writeFloat32Array(buffer, actualData);
            if (bufProxy)
                bufProxy.release();
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
        const outBuffer = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
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
        }
        else if (inst.op === 'transpose') {
            if (!inst.params || inst.params.length < 2) {
                throw new AMEVATensorSecurityError(`transpose instruction missing params`);
            }
            const [rM, rN] = inst.params;
            wgslCode = TRANSPOSE_WGSL;
            device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([rM, rN, 0, 0]));
            dispatchX = Math.ceil(rM / 8);
            dispatchY = Math.ceil(rN / 8);
        }
        else {
            const numElements = byteLength / 4;
            wgslCode = inst.op === 'relu' ? RELU_WGSL :
                inst.op === 'add' ? ADD_WGSL :
                    inst.op === 'mul' ? MUL_WGSL :
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
        const bindGroupEntries = [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
        ];
        if (inst.in.length > 1) {
            bindGroupEntries.push({ binding: 2, resource: { buffer: idToBuffer[inst.in[1]] } });
            bindGroupEntries.push({ binding: 3, resource: { buffer: outBuffer } });
        }
        else {
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
        }
        else {
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
                globalThis.__ameva_last_gpu_error = error.message;
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

/**
 * safeCopy.ts — Pyodide PyProxy → Float32Array 안전 변환
 *
 * H-05 Fix: ensureFloat32Array에서 불필요한 new Float32Array(jsView) deep copy 제거.
 *   Float32Array가 이미 WASM 힙을 가리키고 있으면 그대로 반환 (Zero-Copy).
 *   복사가 실제로 필요한 경우에만 cloneToFloat32Array()를 명시적으로 호출.
 */
function hasToJs(input) {
    return (typeof input === "object" &&
        input !== null &&
        "toJs" in input &&
        typeof input.toJs === "function");
}
/**
 * H-05 Fix: 입력이 이미 Float32Array면 복사 없이 반환.
 * PyProxy인 경우 toJs() 결과를 확인하고 역시 복사 없이 뷰를 반환.
 */
function ensureFloat32Array(input) {
    if (input instanceof Float32Array) {
        return input; // H-05: 복사 제거 — 이미 올바른 타입
    }
    if (hasToJs(input)) {
        const jsView = input.toJs();
        if (jsView instanceof Float32Array) {
            return jsView; // H-05: 복사 제거 — WASM 힙 뷰 그대로 반환
        }
        if (jsView instanceof ArrayBuffer) {
            return new Float32Array(jsView);
        }
    }
    throw new Error("Invalid input type: expected Float32Array or a Pyodide proxy coercible to Float32Array.");
}
/**
 * 명시적 deep copy가 필요한 경우 (예: 버퍼 소유권 이전).
 * 일반 데이터 읽기에는 사용하지 말 것.
 */
function cloneToFloat32Array(input) {
    const view = ensureFloat32Array(input);
    return new Float32Array(view);
}

/**
 * pyodideBridge.ts — globalThis.amevaTensor API 등록자
 *
 * H-02 연동: 단일 실행 경로(graphExecutor.ts)로 통합.
 *   executeGraph 시그니처: (instructionsJson: string, jsInputs: unknown) => Record
 *
 * M-06 연동: disposeBatch 추가 (bridge.py의 js_dispose_batch 지원)
 */
function disposeBatch(handles) {
    for (const handle of handles) {
        if (handle) {
            try {
                dispose(handle);
            }
            catch { /* already disposed */ }
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
        warmupKernels,
        disposeBatch,
    };
    globalThis.amevaTensor = api;
    return api;
}

export { AMEVATensorDTypeError, AMEVATensorDeviceError, AMEVATensorDisposedError, AMEVATensorError, AMEVATensorQuotaExceededError, AMEVATensorSecurityError, AMEVATensorShapeError, AMEVATensorWebGPUUnavailableError, KERNEL_REGISTRY, QuotaManager, add, assertAllowedKernelName, assertAllowedShaderConstant, assertSafeShaderIdentifier, assertStaticShaderSourceOnly, assertWasmRange, cloneToFloat32Array, dispose, ensureFloat32Array, executeGraph, getAdapter, getAllowedKernelNames, getDevice, getQueue, getTensorInfo, init, initWebGPU, isAvailable, mapBufferAsync, matmul, mul, random, read, readMappedInto, registerPyodideBridge, relu, relu_backward, resetRuntimeMemory, setDeviceLostCallback, transpose, uploadFloat32Array, validateDType, validateShape, warmupKernels };
//# sourceMappingURL=index.esm.js.map
