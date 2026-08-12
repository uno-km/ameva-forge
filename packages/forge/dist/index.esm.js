class AMEVAForgeError extends Error {
    constructor(message) {
        super(message);
        this.name = new.target.name;
        Object.setPrototypeOf(this, new.target.prototype);
    }
}
class AMEVAForgeShapeError extends AMEVAForgeError {
}
class AMEVAForgeDTypeError extends AMEVAForgeError {
}
class AMEVAForgeDeviceError extends AMEVAForgeError {
}
class AMEVAForgeDisposedError extends AMEVAForgeError {
}
class AMEVAForgeQuotaExceededError extends AMEVAForgeError {
}
class AMEVAForgeWebGPUUnavailableError extends AMEVAForgeError {
}
class AMEVAForgeSecurityError extends AMEVAForgeError {
}

/**
 * device.ts — WebGPU 싱글톤 디바이스 래퍼
 *
 * H-04 Fix: getAdapter() export 추가 → gpuCore.ts에서 adapter.limits 조회 가능
 * L-03 Fix: device lost 시 onDeviceLostCallback을 통해 pipelineCache도 무효화
 */
function _safeLog$1(msg) {
    try {
        // VUL-015 Fix: Only log in development or explicit debug modes
        const isDev = (typeof process !== 'undefined' && process.env && process.env.NODE_ENV !== 'production') ||
            (typeof globalThis.AMEVA_DEBUG !== 'undefined' && globalThis.AMEVA_DEBUG) ||
            (typeof globalThis.__DEV__ !== 'undefined' && globalThis.__DEV__);
        // Vite/ESBuild injects import.meta.env, wrap in try-catch to avoid syntax errors in older environments
        let isViteDev = false;
        try {
            isViteDev = import.meta.env && import.meta.env.MODE !== 'production';
        }
        catch (e) { }
        if (!isDev && !isViteDev)
            return;
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
        throw new AMEVAForgeWebGPUUnavailableError("WebGPU is not available in this environment. " +
            "Ensure you are running in a supported browser with WebGPU enabled.");
    }
    adapter = await navigator.gpu.requestAdapter(options);
    if (!adapter) {
        throw new AMEVAForgeWebGPUUnavailableError("Failed to request a WebGPU adapter. " +
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
    // _safeLog(`[device.ts] getDevice called. local device=${device ? 'SET' : 'NULL'}, globalDev=${globalDev ? 'SET' : 'NULL'}`);
    if (!device) {
        const globalExists = typeof globalThis.amevaForge !== "undefined";
        throw new AMEVAForgeDeviceError(`WebGPU device is not initialized. (device is ${device}, __AMEVA_DEVICE__ exists: ${!!globalDev}, globalThis.amevaForge exists: ${globalExists}). Call await init() first.`);
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
        throw new AMEVAForgeSecurityError("Invalid offset: must be a non-negative safe integer.");
    }
    if (!Number.isSafeInteger(byteLength) || byteLength < 0) {
        throw new AMEVAForgeSecurityError("Invalid byteLength: must be a non-negative safe integer.");
    }
    if (!Number.isSafeInteger(wasmByteLength) || wasmByteLength < 0) {
        throw new AMEVAForgeSecurityError("Invalid wasmByteLength: must be a non-negative safe integer.");
    }
    if (offset > wasmByteLength || byteLength > wasmByteLength - offset) {
        throw new AMEVAForgeSecurityError("WASM memory range out of bounds");
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
            throw new AMEVAForgeQuotaExceededError(`Invalid hard limit: ${hardLimitBytes}`);
        }
        if (!Number.isSafeInteger(softLimitBytes) || softLimitBytes <= 0) {
            throw new AMEVAForgeQuotaExceededError(`Invalid soft limit: ${softLimitBytes}`);
        }
        if (softLimitBytes > hardLimitBytes) {
            throw new AMEVAForgeQuotaExceededError("softLimitBytes must be <= hardLimitBytes");
        }
        this.hardLimitBytes = hardLimitBytes;
        this.softLimitBytes = softLimitBytes;
    }
    reserve(byteLength) {
        if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
            throw new AMEVAForgeQuotaExceededError(`Invalid allocation size: ${byteLength}`);
        }
        // C-06: 여유 공간 = hardLimit - allocatedBytes (pending 포함, 보수적)
        if (byteLength > this.hardLimitBytes - this.allocatedBytes) {
            throw new AMEVAForgeQuotaExceededError(`Quota Exceeded: Cannot allocate ${byteLength} bytes. ` +
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
    track(byteLength) {
        this.reserve(byteLength);
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
        throw new AMEVAForgeSecurityError(`Invalid shader identifier: "${identifier}". Only alphanumeric and underscore allowed.`);
    }
}
/** 셰이더에 삽입되는 상수값 유효성 검사 */
function assertAllowedShaderConstant(value) {
    if (!Number.isFinite(value)) {
        throw new AMEVAForgeSecurityError(`Invalid shader constant: ${value}. Must be a finite number.`);
    }
}
/**
 * 동적 문자열 보간이 셰이더 소스에 포함되지 않았는지 검사.
 * 템플릿 리터럴 인젝션 공격을 차단한다.
 */
function assertStaticShaderSourceOnly(source) {
    if (source.includes("${") || source.includes("`")) {
        throw new AMEVAForgeSecurityError("Dynamic shader source interpolation is forbidden. Use uniform buffers for runtime values.");
    }
}
/**
 * H-07/NH-07 Fix: 모든 구현된 커널 이름을 화이트리스트에 포함.
 * graphExecutor.ts의 ALLOWED_OPS와 반드시 동기화 유지.
 * 이 함수는 gpuCore.ts와 graphExecutor.ts에서 실제로 호출된다.
 */
let ALLOWED_KERNEL_NAMES = new Set([
    "matmul",
    "relu",
    "relu_backward",
    "add",
    "mul",
    "transpose",
    // v2.0: 학습 기능에 필요한 커널 추가 (VUL-001 Fix)
    "sub",
    "neg",
    "div",
    "exp",
    "log",
    "sigmoid",
    "tanh",
    "sigmoid_backward",
    "tanh_backward",
    "fill",
    "sum",
    "max",
    "sum_axis",
    "axpy",
    "pad",
    "gather",
    "scatter",
    "dropout",
    "maxpool2d",
    "avgpool2d",
    "im2col",
    "col2im",
]);
function registerKernelNames(names) {
    ALLOWED_KERNEL_NAMES = new Set(names);
}
function assertAllowedKernelName(name) {
    if (!ALLOWED_KERNEL_NAMES.has(name)) {
        throw new AMEVAForgeSecurityError(`Unknown kernel name: "${name}". Allowed: ${[...ALLOWED_KERNEL_NAMES].join(", ")}`);
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
        throw new AMEVAForgeShapeError("Shape must be an array.");
    }
    // NM-06 Fix: rank 0 (shape=[]) 허용 — 스칼라 텐서
    if (shape.length > MAX_RANK) {
        throw new AMEVAForgeShapeError(`Shape rank must be between 0 and ${MAX_RANK}, got ${shape.length}.`);
    }
    let elements = 1;
    for (let i = 0; i < shape.length; i++) {
        const dim = shape[i];
        if (!Number.isSafeInteger(dim) || dim <= 0) {
            throw new AMEVAForgeShapeError(`shape[${i}] must be positive, got ${dim}`);
        }
        if (dim > Number.MAX_SAFE_INTEGER / elements) {
            throw new AMEVAForgeShapeError("Shape product overflows safe integer limit.");
        }
        elements *= dim;
    }
    if (elements > MAX_ELEMENTS$1) {
        throw new AMEVAForgeShapeError(`Tensor size exceeds max elements limit: ${elements} > ${MAX_ELEMENTS$1}`);
    }
    if (expectedByteLength !== undefined) {
        const bytesPerElement = BYTES_PER_ELEMENT[dtype];
        if (bytesPerElement === undefined) {
            throw new AMEVAForgeDTypeError(`Unsupported dtype for byte size calculation: "${dtype}". ` +
                `Supported: ${Object.keys(BYTES_PER_ELEMENT).join(', ')}`);
        }
        const calculatedBytes = elements * bytesPerElement;
        if (calculatedBytes !== expectedByteLength) {
            throw new AMEVAForgeShapeError(`Shape/data size mismatch: shape ${JSON.stringify(shape)} (${dtype}) ` +
                `implies ${calculatedBytes} bytes, but data is ${expectedByteLength} bytes.`);
        }
    }
    return elements;
}

function validateDType(dtype) {
    if (dtype !== "float32") {
        throw new AMEVAForgeDTypeError(`Unsupported dtype: ${dtype}. Only float32 is supported.`);
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
            throw new AMEVAForgeDisposedError(`Tensor not found: ${handle}`);
        }
        if (record.disposed) {
            throw new AMEVAForgeDisposedError(`Attempted to access disposed tensor: ${handle}`);
        }
        return record;
    }
    has(handle) {
        const record = this.records.get(handle);
        return record !== undefined && !record.disposed;
    }
    dispose(handle) {
        if (!this.records.has(handle)) {
            return; // TS-H04: 이중 dispose 방어 — 이미 해제된 핸들 무시
        }
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
        const pendingEntries = entries.filter(e => !this.cache.has(`${e.key}:${hashString(e.wgslCode)}`));
        const promises = pendingEntries.map(async (e) => {
            const cacheKey = `${e.key}:${hashString(e.wgslCode)}`;
            const shader = device.createShaderModule({ code: e.wgslCode });
            const pipeline = await device.createComputePipelineAsync({
                layout: "auto",
                compute: { module: shader, entryPoint: "main" },
            });
            this.cache.set(cacheKey, { shader, pipeline });
        });
        const results = await Promise.allSettled(promises);
        results.forEach((res, i) => {
            if (res.status === 'rejected') {
                console.warn(`[AMEVA] Warmup failed for ${pendingEntries[i].key}:`, res.reason);
            }
        });
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
  // global_id.z 를 X축 타일 오프셋으로 사용
  // dispatcher가 z = ceil(N / (65535*8))만큼 dispatch
  let col = global_id.x + global_id.z * 65535u * 8u;
  let row = global_id.y + params.offsetY;

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
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx >= num_elements) {
    return;
  }
  y[idx] = max(x[idx], 0.0);
}
`;

const ADD_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    out[idx] = a[idx] + b[idx];
  }
}
`;

const TRANSPOSE_WGSL = `
struct Params {
  M: u32,
  N: u32,
  B: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  let batch = global_id.z;
  
  if (row < params.M && col < params.N && batch < params.B) {
    let in_idx = batch * (params.M * params.N) + row * params.N + col;
    let out_idx = batch * (params.M * params.N) + col * params.M + row;
    out[out_idx] = input[in_idx];
  }
}
`;

const MUL_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> A : array<f32>;
@group(0) @binding(2) var<storage, read> B : array<f32>;
@group(0) @binding(3) var<storage, read_write> C : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let index = global_id.x + global_id.y * workgroups_x * 64u;
  if (index < num_elements) {
    C[index] = A[index] * B[index];
  }
}
`;

const RELU_BACKWARD_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> X : array<f32>;
@group(0) @binding(2) var<storage, read> gradOutput : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradInput : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let index = global_id.x + global_id.y * workgroups_x * 64u;
  if (index < num_elements) {
    if (X[index] > 0.0) {
      gradInput[index] = gradOutput[index];
    } else {
      gradInput[index] = 0.0;
    }
  }
}
`;

const SUB_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    out[idx] = a[idx] - b[idx];
  }
}
`;

const NEG_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx >= num_elements) {
    return;
  }
  y[idx] = -x[idx];
}
`;

const DIV_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    out[idx] = a[idx] / b[idx];
  }
}
`;

const EXP_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx >= num_elements) {
    return;
  }
  y[idx] = exp(x[idx]);
}
`;

const LOG_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx >= num_elements) {
    return;
  }
  y[idx] = log(x[idx]);
}
`;

const SIGMOID_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx >= num_elements) {
    return;
  }
  y[idx] = 1.0 / (1.0 + exp(-x[idx]));
}
`;

const TANH_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx >= num_elements) {
    return;
  }
  y[idx] = tanh(x[idx]);
}
`;

const SIGMOID_BACKWARD_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read> sigmoid_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx >= num_elements) {
    return;
  }
  output[idx] = grad[idx] * sigmoid_output[idx] * (1.0 - sigmoid_output[idx]);
}
`;

const TANH_BACKWARD_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read> tanh_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx >= num_elements) {
    return;
  }
  output[idx] = grad[idx] * (1.0 - tanh_output[idx] * tanh_output[idx]);
}
`;

const FILL_WGSL = `
struct Params {
  numElements: u32,
  value: f32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.numElements;
  let idx = global_id.x;
  if (idx >= num_elements) {
    return;
  }
  output[idx] = params.value;
}
`;

const SUM_WGSL = `
struct Params {
  numElements: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  let gid = global_id.x;
  let lid = local_id.x;
  let wid = workgroup_id.x;
  
  if (gid < params.numElements) {
    shared[lid] = input[gid];
  } else {
    shared[lid] = 0.0;
  }
  
  workgroupBarrier();
  
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      shared[lid] += shared[lid + s];
    }
    workgroupBarrier();
  }
  
  if (lid == 0u) {
    output[wid] = shared[0];
  }
}
`;

const MAX_WGSL = `
struct Params {
  numElements: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  let gid = global_id.x;
  let lid = local_id.x;
  let wid = workgroup_id.x;
  
  if (gid < params.numElements) {
    shared[lid] = input[gid];
  } else {
    shared[lid] = -3.402823e+38;
  }
  
  workgroupBarrier();
  
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      shared[lid] = max(shared[lid], shared[lid + s]);
    }
    workgroupBarrier();
  }
  
  if (lid == 0u) {
    output[wid] = shared[0];
  }
}
`;

const SUM_AXIS_WGSL = `
struct Params {
  M: u32,
  N: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let col = global_id.x;
  if (col >= params.N) {
    return;
  }
  
  var sum = 0.0;
  for (var row = 0u; row < params.M; row = row + 1u) {
    sum += input[row * params.N + col];
  }
  
  output[col] = sum;
}
`;

const AXPY_WGSL = `
struct Params {
  numElements: u32,
  lr: f32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> param: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.numElements) {
    return;
  }
  param[idx] = param[idx] - params.lr * grad[idx];
}
`;

const PAD_WGSL = `
struct Params {
  num_elements: u32,
  rank: u32,
  pad_val: f32,
  _pad: u32,
  in_strides: array<u32, 8>,
  out_strides: array<u32, 8>,
  pad_before: array<u32, 8>,
  in_shape: array<u32, 8>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_elements) { return; }

  var temp = idx;
  var in_idx = 0u;
  var in_bounds = true;

  for (var i = 0u; i < params.rank; i = i + 1u) {
    let coord = temp / params.out_strides[i];
    temp = temp % params.out_strides[i];
    
    if (coord < params.pad_before[i] || coord >= params.pad_before[i] + params.in_shape[i]) {
      in_bounds = false;
      break;
    }
    let in_coord = coord - params.pad_before[i];
    in_idx = in_idx + in_coord * params.in_strides[i];
  }

  if (in_bounds) {
    output[idx] = input[in_idx];
  } else {
    output[idx] = params.pad_val;
  }
}
`;

const GATHER_WGSL = `
struct Params {
  num_elements: u32,
  dim: u32,
  rank: u32,
  _pad: u32,
  x_strides: array<u32, 8>,
  out_strides: array<u32, 8>,
  x_shape: array<u32, 8>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_elements) { return; }

  var temp = idx;
  var in_idx = 0u;

  for (var i = 0u; i < params.rank; i = i + 1u) {
    let coord = temp / params.out_strides[i];
    temp = temp % params.out_strides[i];
    
    if (i == params.dim) {
      let idx_val = u32(index[idx]);
      in_idx = in_idx + idx_val * params.x_strides[i];
    } else {
      in_idx = in_idx + coord * params.x_strides[i];
    }
  }

  output[idx] = input[in_idx];
}
`;

const SCATTER_WGSL = `
struct Params {
  num_elements: u32,
  dim: u32,
  rank: u32,
  _pad: u32,
  x_strides: array<u32, 8>,
  idx_strides: array<u32, 8>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> index: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_elements) { return; }

  var temp = idx;
  var out_idx = 0u;

  for (var i = 0u; i < params.rank; i = i + 1u) {
    let coord = temp / params.idx_strides[i];
    temp = temp % params.idx_strides[i];
    
    if (i == params.dim) {
      let idx_val = u32(index[idx]);
      out_idx = out_idx + idx_val * params.x_strides[i];
    } else {
      out_idx = out_idx + coord * params.x_strides[i];
    }
  }

  // Not strictly atomic, but for simple scatter where indices are unique it's fine.
  output[out_idx] = src[idx];
}
`;

const CAT_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  a_dim: u32,
  b_dim: u32,
  stride: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  if (idx >= num_elements) {
    return;
  }
  
  let stride = params.stride;
  let a_dim = params.a_dim;
  let b_dim = params.b_dim;
  
  let out_dim_size = a_dim + b_dim;
  let chunk_size = out_dim_size * stride;
  
  let batch_idx = idx / chunk_size;
  let rem = idx % chunk_size;
  let dim_idx = rem / stride;
  let stride_idx = rem % stride;
  
  if (dim_idx < a_dim) {
    let a_index = batch_idx * (a_dim * stride) + dim_idx * stride + stride_idx;
    out[idx] = a[a_index];
  } else {
    let b_dim_idx = dim_idx - a_dim;
    let b_index = batch_idx * (b_dim * stride) + b_dim_idx * stride + stride_idx;
    out[idx] = b[b_index];
  }
}
`;

const WHERE_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
  pad4: u32,
  pad5: u32,
  pad6: u32,
  pad7: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx >= num_elements) {
    return;
  }
  if (cond[idx] > 0.0) {
    out[idx] = x[idx];
  } else {
    out[idx] = y[idx];
  }
}
`;

const DROPOUT_WGSL = `
struct Params {
  num_elements: u32,
  seed: f32,
  p: f32,
  padding: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(hash: u32) -> f32 {
    return f32(hash) / 4294967295.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + global_id.y * 65535u * 64u;
    if (index >= params.num_elements) {
        return;
    }
    
    let hash = pcg_hash(index + u32(params.seed * 10000.0));
    let rand = rand_f32(hash);
    
    if (rand < params.p) {
        out[index] = 0.0;
    } else {
        out[index] = x[index] * (1.0 / (1.0 - params.p));
    }
}
`;

const MAXPOOL2D_WGSL = `
struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kH: u32,
    kW: u32,
    sH: u32,
    sW: u32,
    pH: u32,
    pW: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch * params.channels * params.out_h * params.out_w;
    if (idx >= total) {
        return;
    }
    
    let ow = idx % params.out_w;
    let oh = (idx / params.out_w) % params.out_h;
    let c = (idx / (params.out_w * params.out_h)) % params.channels;
    let b = idx / (params.out_w * params.out_h * params.channels);
    
    let h_start = i32(oh * params.sH) - i32(params.pH);
    let w_start = i32(ow * params.sW) - i32(params.pW);
    
    var max_val = -3.402823466e+38; // -FLT_MAX
    
    for (var kh = 0u; kh < params.kH; kh++) {
        for (var kw = 0u; kw < params.kW; kw++) {
            let h = h_start + i32(kh);
            let w = w_start + i32(kw);
            
            if (h >= 0 && h < i32(params.in_h) && w >= 0 && w < i32(params.in_w)) {
                let in_idx = ((b * params.channels + c) * params.in_h + u32(h)) * params.in_w + u32(w);
                let val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    output[idx] = max_val;
}
`;

const AVGPOOL2D_WGSL = `
struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kH: u32,
    kW: u32,
    sH: u32,
    sW: u32,
    pH: u32,
    pW: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch * params.channels * params.out_h * params.out_w;
    if (idx >= total) {
        return;
    }
    
    let ow = idx % params.out_w;
    let oh = (idx / params.out_w) % params.out_h;
    let c = (idx / (params.out_w * params.out_h)) % params.channels;
    let b = idx / (params.out_w * params.out_h * params.channels);
    
    let h_start = i32(oh * params.sH) - i32(params.pH);
    let w_start = i32(ow * params.sW) - i32(params.pW);
    
    var sum = 0.0;
    var count = 0.0;
    
    for (var kh = 0u; kh < params.kH; kh++) {
        for (var kw = 0u; kw < params.kW; kw++) {
            let h = h_start + i32(kh);
            let w = w_start + i32(kw);
            
            if (h >= 0 && h < i32(params.in_h) && w >= 0 && w < i32(params.in_w)) {
                let in_idx = ((b * params.channels + c) * params.in_h + u32(h)) * params.in_w + u32(w);
                sum += input[in_idx];
                count += 1.0;
            }
        }
    }
    
    if (count > 0.0) {
        output[idx] = sum / count;
    } else {
        output[idx] = 0.0;
    }
}
`;

const IM2COL_WGSL = `
struct Params {
  N: u32,
  C: u32,
  H: u32,
  W: u32,
  K_h: u32,
  K_w: u32,
  stride: u32,
  padding: u32,
  H_out: u32,
  W_out: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let num_elements = params.N * params.H_out * params.W_out * params.C * params.K_h * params.K_w;
  if (idx >= num_elements) { return; }

  var temp = idx;
  let c_kw_kh = temp % (params.C * params.K_h * params.K_w);
  temp = temp / (params.C * params.K_h * params.K_w);
  let h_out_w_out = temp % (params.H_out * params.W_out);
  temp = temp / (params.H_out * params.W_out);
  let n = temp % params.N;

  let k_w = c_kw_kh % params.K_w;
  let k_h = (c_kw_kh / params.K_w) % params.K_h;
  let c = c_kw_kh / (params.K_w * params.K_h);

  let w_out = h_out_w_out % params.W_out;
  let h_out = h_out_w_out / params.W_out;

  let h_in = i32(h_out * params.stride) - i32(params.padding) + i32(k_h);
  let w_in = i32(w_out * params.stride) - i32(params.padding) + i32(k_w);

  if (h_in >= 0 && h_in < i32(params.H) && w_in >= 0 && w_in < i32(params.W)) {
    let in_idx = ((n * params.C + c) * params.H + u32(h_in)) * params.W + u32(w_in);
    output[idx] = input[in_idx];
  } else {
    output[idx] = 0.0;
  }
}
`;

const COL2IM_WGSL = `
struct Params {
  N: u32,
  C: u32,
  H: u32,
  W: u32,
  K_h: u32,
  K_w: u32,
  stride: u32,
  padding: u32,
  H_out: u32,
  W_out: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad_x_col: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let num_elements = params.N * params.C * params.H * params.W;
  if (idx >= num_elements) { return; }

  var temp = idx;
  let w = temp % params.W;
  temp = temp / params.W;
  let h = temp % params.H;
  temp = temp / params.H;
  let c = temp % params.C;
  let n = temp / params.C;

  var val = 0.0;
  
  for (var k_h = 0u; k_h < params.K_h; k_h = k_h + 1u) {
    let h_plus_pad = h + params.padding;
    if (h_plus_pad >= k_h) {
      let h_rem = h_plus_pad - k_h;
      if (h_rem % params.stride == 0u) {
        let h_out = h_rem / params.stride;
        if (h_out < params.H_out) {
          
          for (var k_w = 0u; k_w < params.K_w; k_w = k_w + 1u) {
            let w_plus_pad = w + params.padding;
            if (w_plus_pad >= k_w) {
              let w_rem = w_plus_pad - k_w;
              if (w_rem % params.stride == 0u) {
                let w_out = w_rem / params.stride;
                if (w_out < params.W_out) {
                  let n_out = n;
                  let hw_out = h_out * params.W_out + w_out;
                  let c_kw_kh = (c * params.K_h + k_h) * params.K_w + k_w;
                  
                  let col_idx = (n_out * (params.H_out * params.W_out) + hw_out) * (params.C * params.K_h * params.K_w) + c_kw_kh;
                  val = val + grad_x_col[col_idx];
                }
              }
            }
          }
          
        }
      }
    }
  }

  grad_x[idx] = val;
}
`;

const BATCHED_MATMUL_WGSL = `
struct Params {
  B: u32,
  M: u32,
  N: u32,
  K: u32,
  strideA: u32,
  strideB: u32,
  strideC: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let col = global_id.x;
  let row = global_id.y;
  let batch = global_id.z;

  if (row >= params.M || col >= params.N || batch >= params.B) {
    return;
  }

  let a_offset = batch * params.strideA + row * params.K;
  let b_offset = batch * params.strideB + col;
  let c_offset = batch * params.strideC + row * params.N + col;

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + a[a_offset + k] * b[b_offset + k * params.N];
  }

  c[c_offset] = sum;
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
    ['batched_matmul', BATCHED_MATMUL_WGSL],
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
    ['dropout', DROPOUT_WGSL],
    ['maxpool2d', MAXPOOL2D_WGSL],
    ['avgpool2d', AVGPOOL2D_WGSL],
    ['im2col', IM2COL_WGSL],
    ['col2im', COL2IM_WGSL],
]);
// VUL-001 Fix: Register kernel names automatically to keep whitelist in sync
registerKernelNames(KERNEL_REGISTRY.keys());
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
    _globalQuotaManager.track(stagingBuffer.size);
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
        _globalQuotaManager.release(stagingBuffer.size);
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
        throw new AMEVAForgeShapeError("Matmul requires 2D tensors");
    if (a.dtype !== "float32" || b.dtype !== "float32")
        throw new AMEVAForgeDTypeError("Matmul requires float32 tensors");
    const M = a.shape[0], K = a.shape[1], K2 = b.shape[0], N = b.shape[1];
    if (K !== K2)
        throw new AMEVAForgeShapeError(`Inner dim mismatch: ${K} != ${K2}`);
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
        throw new AMEVAForgeDTypeError("ReLU requires float32");
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
function mul(handleA, handleB) {
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
function transpose(handle) {
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
function relu_backward(handleX, handleGrad) {
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
/**
 * C-04: 단일 instruction의 무결성을 검증한다.
 */
function validateInstruction(inst, idx) {
    if (typeof inst !== 'object' || inst === null) {
        throw new AMEVAForgeSecurityError(`Instruction[${idx}]: must be an object`);
    }
    const i = inst;
    if (typeof i.op !== 'string') {
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
    // NM-06: rank 0 허용 (스칼라)
    if (i.shape.length > MAX_SHAPE_DIM) {
        throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape rank must be 0–${MAX_SHAPE_DIM}, got ${i.shape.length}`);
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
    // NC-06: in 필드가 있으면 배열인지 확인
    if (i.in !== undefined && !Array.isArray(i.in)) {
        throw new AMEVAForgeSecurityError(`Instruction[${idx}]: 'in' field must be an array`);
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
        throw new AMEVAForgeSecurityError("executeGraph: invalid JSON in instructionsJson");
    }
    if (!Array.isArray(rawInstructions)) {
        throw new AMEVAForgeSecurityError("executeGraph: instructionsJson must be a JSON array");
    }
    if (rawInstructions.length > MAX_INSTRUCTIONS) {
        throw new AMEVAForgeSecurityError(`executeGraph: too many instructions (${rawInstructions.length} > ${MAX_INSTRUCTIONS})`);
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
                throw new AMEVAForgeSecurityError(`load instruction missing handle`);
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
                throw new AMEVAForgeSecurityError(`upload input[${inputIdx - 1}] is not a Float32Array or convertible type`);
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
        let outBuffer;
        if (inst.op === 'axpy') {
            if (!inst.in || inst.in.length < 2) {
                throw new AMEVAForgeSecurityError(`Instruction axpy is missing 'in' fields.`);
            }
            outBuffer = idToBuffer[inst.in[1]];
            idToHandle[inst.id] = idToHandle[inst.in[1]];
            idToBuffer[inst.id] = outBuffer;
        }
        else {
            outBuffer = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
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
        if (inst.op === 'pad')
            paramsSize = 144;
        else if (inst.op === 'gather' || inst.op === 'scatter')
            paramsSize = 112;
        else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d')
            paramsSize = 48;
        else if (inst.op === 'im2col' || inst.op === 'col2im')
            paramsSize = 40;
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
            }
            else {
                dispatchX = 65535;
                dispatchZ = Math.ceil(rawDispatchX / 65535);
            }
            const maxWorkgroupsM = Math.ceil(M / 8);
            dispatchY = Math.min(65535, maxWorkgroupsM);
        }
        else if (inst.op === 'batched_matmul') {
            if (!inst.params || inst.params.length < 7) {
                throw new AMEVAForgeSecurityError(`batched_matmul instruction missing params`);
            }
            [B, M, N, K] = inst.params;
            wgslCode = BATCHED_MATMUL_WGSL;
            dispatchX = Math.ceil(N / 8);
            dispatchY = Math.ceil(M / 8);
            dispatchZ = B;
            device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array(inst.params));
        }
        else if (inst.op === 'transpose') {
            if (!inst.params || inst.params.length < 3) {
                throw new AMEVAForgeSecurityError(`transpose instruction missing params`);
            }
            const [rM, rN, rB] = inst.params;
            wgslCode = TRANSPOSE_WGSL;
            device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([rM, rN, rB, 0]));
            dispatchX = Math.ceil(rM / 8);
            dispatchY = Math.ceil(rN / 8);
            dispatchZ = rB;
        }
        else if (inst.op === 'sum_axis') {
            if (!inst.params || inst.params.length < 2) {
                throw new AMEVAForgeSecurityError(`sum_axis instruction missing params`);
            }
            const [M, N] = inst.params;
            wgslCode = SUM_AXIS_WGSL;
            device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, N, 0, 0]));
            dispatchX = Math.ceil(N / 64);
        }
        else if (inst.op === 'fill') {
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
        }
        else if (inst.op === 'axpy') {
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
        }
        else if (inst.op === 'pad') {
            const numElements = byteLength / 4;
            wgslCode = PAD_WGSL;
            const p = new Uint32Array(36);
            for (let i = 0; i < inst.params.length; i++) {
                if (i === 2)
                    new Float32Array(p.buffer)[2] = inst.params[2];
                else
                    p[i] = inst.params[i];
            }
            device.queue.writeBuffer(paramsBuffer, 0, p);
            dispatchX = Math.ceil(numElements / 64);
        }
        else if (inst.op === 'gather') {
            const numElements = byteLength / 4;
            wgslCode = GATHER_WGSL;
            const p = new Uint32Array(28);
            for (let i = 0; i < inst.params.length; i++)
                p[i] = inst.params[i];
            device.queue.writeBuffer(paramsBuffer, 0, p);
            dispatchX = Math.ceil(numElements / 64);
        }
        else if (inst.op === 'scatter') {
            const numElements = inst.params[0];
            wgslCode = SCATTER_WGSL;
            const p = new Uint32Array(28);
            for (let i = 0; i < inst.params.length; i++)
                p[i] = inst.params[i];
            device.queue.writeBuffer(paramsBuffer, 0, p);
            dispatchX = Math.ceil(numElements / 64);
        }
        else if (inst.op === 'dropout') {
            const numElements = byteLength / 4;
            const seed = inst.params[0];
            const p = inst.params[1];
            wgslCode = DROPOUT_WGSL;
            const f32arr = new Float32Array([0, seed, p, 0]);
            const u32arr = new Uint32Array(f32arr.buffer);
            u32arr[0] = numElements;
            device.queue.writeBuffer(paramsBuffer, 0, u32arr);
            dispatchX = Math.ceil(numElements / 64);
        }
        else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d') {
            const numElements = byteLength / 4;
            wgslCode = inst.op === 'maxpool2d' ? MAXPOOL2D_WGSL : AVGPOOL2D_WGSL;
            const p = new Uint32Array(12);
            for (let i = 0; i < inst.params.length; i++)
                p[i] = inst.params[i];
            device.queue.writeBuffer(paramsBuffer, 0, p);
            dispatchX = Math.ceil(numElements / 64);
        }
        else if (inst.op === 'im2col' || inst.op === 'col2im') {
            const numElements = byteLength / 4;
            wgslCode = inst.op === 'im2col' ? IM2COL_WGSL : COL2IM_WGSL;
            const p = new Uint32Array(10);
            for (let i = 0; i < inst.params.length; i++)
                p[i] = inst.params[i];
            device.queue.writeBuffer(paramsBuffer, 0, p);
            dispatchX = Math.ceil(numElements / 64);
        }
        else if (inst.op === 'sum' || inst.op === 'max') {
            // Handled entirely dynamically below, but we need to bypass normal flow
            wgslCode = inst.op === 'sum' ? SUM_WGSL : MAX_WGSL;
        }
        else {
            const numElements = byteLength / 4;
            wgslCode = inst.op === 'relu' ? RELU_WGSL :
                inst.op === 'add' ? ADD_WGSL :
                    inst.op === 'mul' ? MUL_WGSL :
                        inst.op === 'sub' ? SUB_WGSL :
                            inst.op === 'neg' ? NEG_WGSL :
                                inst.op === 'div' ? DIV_WGSL :
                                    inst.op === 'relu_backward' ? RELU_BACKWARD_WGSL :
                                        inst.op === 'exp' ? EXP_WGSL :
                                            inst.op === 'log' ? LOG_WGSL :
                                                inst.op === 'sigmoid' ? SIGMOID_WGSL :
                                                    inst.op === 'tanh' ? TANH_WGSL :
                                                        inst.op === 'sigmoid_backward' ? SIGMOID_BACKWARD_WGSL :
                                                            inst.op === 'tanh_backward' ? TANH_BACKWARD_WGSL :
                                                                inst.op === 'cat' ? CAT_WGSL :
                                                                    inst.op === 'where' ? WHERE_WGSL :
                                                                        inst.op === 'dropout' ? DROPOUT_WGSL : '';
            if (!wgslCode) {
                throw new AMEVAForgeSecurityError(`Unknown op "${inst.op}"`);
            }
            const totalWorkgroups = Math.ceil(numElements / 64);
            // TS-C01 Fix: 65535 초과 시 2D 그리드로 분산
            if (totalWorkgroups <= 65535) {
                dispatchX = totalWorkgroups;
                dispatchY = 1;
            }
            else {
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
            const intermediateBuffers = [];
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
        let bindGroupEntries = [];
        if (inst.op === 'fill') {
            bindGroupEntries = [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: outBuffer } },
            ];
        }
        else if (inst.op === 'axpy') {
            bindGroupEntries = [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                { binding: 2, resource: { buffer: idToBuffer[inst.in[1]] } },
            ];
        }
        else if (inst.op === 'pad') {
            bindGroupEntries = [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                { binding: 2, resource: { buffer: outBuffer } },
            ];
        }
        else if (inst.op === 'gather' || inst.op === 'scatter') {
            bindGroupEntries = [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                { binding: 2, resource: { buffer: idToBuffer[inst.in[1]] } },
                { binding: 3, resource: { buffer: outBuffer } },
            ];
        }
        else if (inst.op === 'where') {
            bindGroupEntries = [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                { binding: 2, resource: { buffer: idToBuffer[inst.in[1]] } },
                { binding: 3, resource: { buffer: idToBuffer[inst.in[2]] } },
                { binding: 4, resource: { buffer: outBuffer } },
            ];
        }
        else if (inst.op === 'dropout') {
            bindGroupEntries = [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                { binding: 2, resource: { buffer: outBuffer } },
            ];
        }
        else {
            bindGroupEntries = [
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
        }
        else {
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
 * pyodideBridge.ts — globalThis.amevaForge API 등록자
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
    globalThis.amevaForge = api;
    return api;
}

export { AMEVAForgeDTypeError, AMEVAForgeDeviceError, AMEVAForgeDisposedError, AMEVAForgeError, AMEVAForgeQuotaExceededError, AMEVAForgeSecurityError, AMEVAForgeShapeError, AMEVAForgeWebGPUUnavailableError, KERNEL_REGISTRY, QuotaManager, add, assertAllowedKernelName, assertAllowedShaderConstant, assertSafeShaderIdentifier, assertStaticShaderSourceOnly, assertWasmRange, cloneToFloat32Array, dispose, ensureFloat32Array, executeGraph, getAdapter, getAllowedKernelNames, getDevice, getQueue, getTensorInfo, init, initWebGPU, isAvailable, mapBufferAsync, matmul, mul, random, read, readMappedInto, registerKernelNames, registerPyodideBridge, relu, relu_backward, resetRuntimeMemory, setDeviceLostCallback, transpose, uploadFloat32Array, validateDType, validateShape, warmupKernels };
//# sourceMappingURL=index.esm.js.map
