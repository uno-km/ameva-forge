/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: Grand Unified All-Modal Orchestrator WebGPU Native Tests (SCRUM-334 & SCRUM-335)
 */

import { AllModalOrchestrator, ALL_MODAL_CAPABILITIES } from '../src/orchestrator/allModalOrchestrator';
import { _setDeviceForTesting, _resetDeviceForTesting } from '../src/webgpu/device';
import { AMEVAForgeDeviceError } from '../src/errors';
import { LLMEngine, LLMWeights } from '../src/llm/llmEngine';
import { CLIPVisionWeights } from '../src/vision/clipVisionEncoder';

function createMockDevice(): any {
  return {
    createShaderModule: jest.fn(() => ({})),
    createComputePipeline: jest.fn(() => ({
      getBindGroupLayout: jest.fn(() => ({})),
    })),
    createBuffer: jest.fn((desc: any) => ({
      size: desc.size,
      usage: desc.usage,
      destroy: jest.fn(),
      mapAsync: jest.fn().mockResolvedValue(undefined),
      getMappedRange: jest.fn((offset?: number, size?: number) => new ArrayBuffer(size !== undefined ? size : desc.size)),
      unmap: jest.fn(),
    })),
    createBindGroupLayout: jest.fn(() => ({})),
    createBindGroup: jest.fn(() => ({})),
    createCommandEncoder: jest.fn(() => ({
      beginComputePass: jest.fn(() => ({
        setPipeline: jest.fn(),
        setBindGroup: jest.fn(),
        dispatchWorkgroups: jest.fn(),
        end: jest.fn(),
      })),
      copyBufferToBuffer: jest.fn(),
      finish: jest.fn(() => ({})),
    })),
    pushErrorScope: jest.fn(),
    popErrorScope: jest.fn().mockResolvedValue(null),
    queue: {
      writeBuffer: jest.fn(),
      submit: jest.fn(),
      onSubmittedWorkDone: jest.fn().mockResolvedValue(undefined),
    },
  };
}

describe('AllModalOrchestrator Grand Unified WebGPU Native Tests (SCRUM-334 & SCRUM-335)', () => {
  let orchestrator: AllModalOrchestrator;
  let mockDevice: any;

  beforeEach(() => {
    orchestrator = new AllModalOrchestrator();
    mockDevice = createMockDevice();
  });

  afterEach(() => {
    _resetDeviceForTesting();
  });

  it('manifests full 5-modality capabilities truthfully', () => {
    expect(ALL_MODAL_CAPABILITIES.modalities).toEqual(['stt', 'llm', 'vision', 'tts', 'diffusion']);
    expect(ALL_MODAL_CAPABILITIES.zero_silent_fallback_enforced).toBe(true);
    expect(ALL_MODAL_CAPABILITIES.webgpu_compute_accelerated).toBe(true);
    expect(ALL_MODAL_CAPABILITIES.on_device_runtime).toBe('WebGPU-Vulkan-Native-Unified');
  });

  it('executes STT (listen) converting audio PCM to 80-bin mel spectrogram', () => {
    const pcm = new Float32Array(16000 * 0.2).fill(0.1);
    const { mels, numFrames } = orchestrator.listen(pcm, 16000);
    expect(numFrames).toBeGreaterThan(5);
    expect(mels.length).toBe(80 * numFrames);
  });

  it('executes STT GPU (listenGPU) on WebGPU hardware acceleration', async () => {
    _setDeviceForTesting(mockDevice);
    const pcm = new Float32Array(16000 * 0.2).fill(0.1);
    const { mels, numFrames } = await orchestrator.listenGPU(pcm, 16000);
    expect(numFrames).toBeGreaterThan(5);
    expect(mels.length).toBe(80 * numFrames);
    expect(mockDevice.queue.submit).toHaveBeenCalled();
  });

  it('executes Vision (seeEdges) detecting image boundaries with Canny 8-direction BFS', () => {
    const rgba = new Uint8ClampedArray(16 * 16 * 4).fill(200);
    const edges = orchestrator.seeEdges(rgba, 16, 16);
    expect(edges.length).toBe(16 * 16);
  });

  it('executes Vision GPU (seeEmbeddingsGPU) computing ViT patch embeddings', async () => {
    _setDeviceForTesting(mockDevice);
    const rgb = new Float32Array(3 * 16 * 16).fill(0.2);
    const weights: CLIPVisionWeights = {
      patchConvWeight: new Float32Array(768 * 3 * 16 * 16).fill(0.01),
      classEmbedding: new Float32Array(768).fill(0.01),
      positionEmbedding: new Float32Array(2 * 768).fill(0.01),
      preNormGamma: new Float32Array(768).fill(1.0),
      preNormBeta: new Float32Array(768).fill(0.0),
      postNormGamma: new Float32Array(768).fill(1.0),
      postNormBeta: new Float32Array(768).fill(0.0),
      layers: [],
    };
    const embedding = await orchestrator.seeEmbeddingsGPU(rgb, 16, 16, weights);
    expect(embedding.length).toBe(768);
  });

  it('executes TTS (speak) synthesizing natural PCM audio using formant resonators', () => {
    const { pcm, durationSeconds } = orchestrator.speak('welcome to ameva forge');
    expect(pcm.length).toBeGreaterThan(1000);
    expect(durationSeconds).toBeGreaterThan(0.5);
  });

  it('executes TTS GPU (speakGPU) synthesizing audio on WebGPU', async () => {
    _setDeviceForTesting(mockDevice);
    const { pcm, durationSeconds } = await orchestrator.speakGPU('webgpu native speech');
    expect(pcm.length).toBeGreaterThan(1000);
    expect(durationSeconds).toBeGreaterThan(0.5);
    expect(mockDevice.queue.submit).toHaveBeenCalled();
  });

  it('strictly throws AMEVAForgeDeviceError in runGrandMultimodalGPU when WebGPU is unavailable', async () => {
    _resetDeviceForTesting(); // Ensure no device

    await expect(
      orchestrator.runGrandMultimodalGPU({
        audioPcm: new Float32Array(1000).fill(0.1),
        llmTokens: [1, 2],
        llmWeights: {
          tokenEmbedding: new Float32Array(100 * 64),
          layers: [],
          finalNormGamma: new Float32Array(64).fill(1.0),
          lmHeadWeight: new Float32Array(100 * 64),
        },
        visionRgb: new Float32Array(3 * 16 * 16),
        visionWidth: 16,
        visionHeight: 16,
        visionWeights: {
          patchConvWeight: new Float32Array(768 * 3 * 16 * 16),
          classEmbedding: new Float32Array(768),
          positionEmbedding: new Float32Array(2 * 768),
          preNormGamma: new Float32Array(768).fill(1.0),
          preNormBeta: new Float32Array(768).fill(0.0),
          postNormGamma: new Float32Array(768).fill(1.0),
          postNormBeta: new Float32Array(768).fill(0.0),
          layers: [],
        },
        ttsText: 'test fail fast',
        diffusionOptions: {
          prompt: 'test prompt',
          numSteps: 1,
          width: 8,
          height: 8,
          latentChannels: 4,
          backend: 'webgpu',
        } as any,
      })
    ).rejects.toThrow(AMEVAForgeDeviceError);
  });
});
