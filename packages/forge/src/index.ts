/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *   - 2026-08-17: Security Hardening: Enforce strict public API boundary, isolate raw GPUDevice
 * 
 * WHAT: 라이브러리의 외부 공개용(Public) API를 모두 한 곳으로 모아 내보내는 진입점(엔트리포인트) 파일입니다.
 * WHY: 패키지 사용자가 내부 디렉토리 구조를 일일이 알 필요 없이 일관된 단일 경로에서 모듈을 쉽게 임포트할 수 있도록 편의성을 제공하기 위함입니다.
 * HOW: 내부의 여러 모듈들에 정의된 클래스, 타입, 함수 등을 export 및 re-export 키워드를 활용하여 다시 바깥으로 통합 추출합니다.
 */

import { getDevice, initWebGPU, isAvailable, _resetDeviceForTesting } from "./webgpu/device";
import { _globalQuotaManager, QuotaManager, getQuotaSnapshot } from "./webgpu/quota";
import { AMEVAForgeValidationError } from "./errors";

declare var process: any;

export const VERSION = "1.0.1";

export * from "./errors";
export * from "./types";

// Security Hardened WebGPU device exports (raw GPUDevice / Queue / Adapter NOT publicly exposed)
export { initWebGPU, isAvailable };
export { assertWasmRange } from "./webgpu/validateWasmRange";
export { QuotaManager, getQuotaSnapshot };
export { flushGC, clearStagingPool } from "./webgpu/buffers";
export * from "./webgpu/shaderGuard";

export * from "./tensor/validateShape";
export * from "./tensor/validateDType";
export * from "./tensor/dispatchShape";
export * from "./tensor/broadcastParams";
export * from "./tensor/gpuCore";
export { executeGraph, configureRuntime, getRuntimeConfig, type ForgeRuntimeConfig } from "./tensor/graphExecutor";

export * from "./bridge/safeCopy";
export * from "./bridge/pyodideBridge";
export * from "./devtools/inspector";
export * from "./loader/ggufStreamer";
export * from "./loader/ggufTensorMapper";
export * from "./tensor/kernels/silu.wgsl";
export * from "./tensor/kernels/upsample2d.wgsl";
export * from "./tensor/kernels/group_norm.wgsl";
export * from "./tensor/kernels/stt_mel.wgsl";
export * from "./tensor/kernels/stt_stft.wgsl";
export * from "./tensor/kernels/tts_synth.wgsl";
export * from "./diffusion/scheduler";
export * from "./diffusion/vaeDecoder";
export * from "./diffusion/resnetBlock";
export * from "./diffusion/autoencoderKL";
export * from "./diffusion/clipTokenizer";
export * from "./diffusion/clipTextEncoder";
export * from "./diffusion/unetGraph";
export * from "./diffusion/pipeline";
export * from "./vision/classicalCV";
export * from "./vision/clipVisionEncoder";
export * from "./vision/vlmEngine";
export * from "./audio/sttEngine";
export * from "./audio/ttsEngine";
export * from "./llm/llmEngine";
export * from "./llm/sampler";
export * from "./llm/llmTextGenerator";
export * from "./tokenizer/bpeTokenizer";
export * from "./loader/modelLoader";
export * from "./worker/inferenceWorker";
export * from "./worker/workerClient";
export * from "./orchestrator/allModalOrchestrator";

/**
 * WHAT: 테스트 환경(E2E / Jest)에서만 제어 가능한 결함 주입(Fault Injection) 훅입니다.
 * WHY: 프로덕션 환경에 raw GPUDevice를 노출하지 않으면서도 OOM, Validation, Device Lost 복구력을 엄격히 검증하기 위함입니다.
 */
export const __testing = (
  (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'test') ||
  (typeof globalThis !== 'undefined' && ((globalThis as any).__AMEVA_TEST_MODE__ || (globalThis as any).testReady))
) ? Object.freeze({
  destroyDevice: () => {
    try {
      getDevice().destroy();
    } catch (e) { console.warn(`[__testing] destroyDevice failed: ${e}`); }
    try {
      _resetDeviceForTesting();
    } catch (e) { console.warn(`[__testing] _resetDeviceForTesting failed: ${e}`); }
  },
  triggerValidationError: async () => {
    const dev = getDevice();
    dev.pushErrorScope('validation');
    try {
      dev.createBuffer({
        size: 1024,
        usage: 0 as GPUBufferUsageFlags, // Usage 0 is an unconditional WebGPU validation fault
      });
    } finally {
      const err = await dev.popErrorScope();
      if (err) {
        throw new AMEVAForgeValidationError(`GPU Validation Error: ${err.message}`);
      }
    }
  },
  setQuotaLimit: (maxBytes: number) => {
    _globalQuotaManager.setLimits(maxBytes, maxBytes);
  },
  getDeviceInternal: getDevice,
}) : undefined;
