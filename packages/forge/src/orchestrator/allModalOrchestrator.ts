/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-334 & SCRUM-335 Grand Unified All-Modal On-Device AI Orchestrator
 *
 * WHAT: STT(귀), LLM(뇌), Vision(눈), TTS(입), Diffusion(손) 5대 모달리티를
 *      WebGPU WGSL 컴퓨트 셰이더 기반 하드웨어 런타임으로 직결하여 구동하는 올모달 오케스트레이터입니다.
 * WHY: CPU 침묵 폴백 없이 브라우저 WebGPU 하드웨어를 100% 활용하는 차세대 온디바이스 AI 런타임 표준을 확립하기 위함입니다.
 * HOW: STTEngine(STT_MEL_WGSL) + LLMEngine(FlashAttention/Tiled GEMM) + CLIPVisionEncoder(GPU GEMM) + TTSEngine(TTS_SYNTH_WGSL) + WebGPUDiffusionPipeline.
 */

import { STTEngine } from '../audio/sttEngine';
import { TTSEngine } from '../audio/ttsEngine';
import { ClassicalCV } from '../vision/classicalCV';
import { CLIPVisionEncoder, CLIPVisionWeights } from '../vision/clipVisionEncoder';
import { LLMEngine, LLMWeights, KVCache } from '../llm/llmEngine';
import { WebGPUDiffusionPipeline, GenerationOptions } from '../diffusion/pipeline';
import { DecodedImage } from '../diffusion/vaeDecoder';
import { getDevice } from '../webgpu/device';
import { AMEVAForgeDeviceError } from '../errors';

export interface AllModalCapabilities {
  readonly modalities: readonly ['stt', 'llm', 'vision', 'tts', 'diffusion'];
  readonly zero_silent_fallback_enforced: boolean;
  readonly on_device_runtime: string;
  readonly webgpu_compute_accelerated: boolean;
}

export const ALL_MODAL_CAPABILITIES: AllModalCapabilities = Object.freeze({
  modalities: ['stt', 'llm', 'vision', 'tts', 'diffusion'] as const,
  zero_silent_fallback_enforced: true,
  on_device_runtime: 'WebGPU-Vulkan-Native-Unified',
  webgpu_compute_accelerated: true,
});

export class AllModalOrchestrator {
  public diffusionPipeline: WebGPUDiffusionPipeline;

  constructor() {
    this.diffusionPipeline = new WebGPUDiffusionPipeline();
  }

  /**
   * 1. STT (귀): 16kHz PCM 오디오를 80채널 로그 멜-스펙트로그램으로 변환하여 분석합니다.
   */
  public listen(pcm: Float32Array, sampleRate: number = 16000): { mels: Float32Array; numFrames: number } {
    return STTEngine.computeLogMelSpectrogram(pcm, sampleRate);
  }

  public async listenGPU(pcm: Float32Array, sampleRate: number = 16000): Promise<{ mels: Float32Array; numFrames: number }> {
    return STTEngine.computeLogMelSpectrogramGPU(pcm, sampleRate);
  }

  /**
   * 2. LLM (뇌): RoPE, RMSNorm, SwiGLU 기반 트랜스포머 디코더로 토큰 로짓을 예측합니다.
   */
  public think(tokenId: number, pos: number, weights: LLMWeights, kvCaches: KVCache[], dim: number = LLMEngine.DIM, vocabSize: number = 32000): Float32Array {
    return LLMEngine.forwardToken(tokenId, pos, weights, kvCaches, dim, vocabSize);
  }

  public async thinkGPU(tokenId: number, pos: number, weights: LLMWeights, kvCaches: KVCache[], dim: number = LLMEngine.DIM, vocabSize: number = 32000): Promise<Float32Array> {
    return LLMEngine.forwardTokenGPU(tokenId, pos, weights, kvCaches, dim, vocabSize);
  }

  /**
   * 3. Vision (눈): RGBA 이미지로부터 에지를 검출하거나 ViT를 통해 768차원 시맨틱 특징 벡터를 추출합니다.
   */
  public seeEdges(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): Uint8Array {
    const gray = ClassicalCV.toGrayscale(rgba, width, height);
    return ClassicalCV.canny(gray, width, height);
  }

  public seeEmbeddings(rgb: Float32Array, width: number, height: number, weights: CLIPVisionWeights): Float32Array {
    const { imageEmbedding } = CLIPVisionEncoder.forward(rgb, width, height, weights);
    return imageEmbedding;
  }

  public async seeEmbeddingsGPU(rgb: Float32Array, width: number, height: number, weights: CLIPVisionWeights): Promise<Float32Array> {
    const { imageEmbedding } = await CLIPVisionEncoder.forwardGPU(rgb, width, height, weights);
    return imageEmbedding;
  }

  /**
   * 4. TTS (입): 로젠버그 성문 펄스와 5-밴드 바이쿼드 필터로 실시간 PCM 오디오를 합성합니다.
   */
  public speak(text: string, sampleRate: number = 22050): { pcm: Float32Array; durationSeconds: number } {
    const res = TTSEngine.synthesize(text, sampleRate);
    return { pcm: res.pcm, durationSeconds: res.durationSeconds };
  }

  public async speakGPU(text: string, sampleRate: number = 22050): Promise<{ pcm: Float32Array; durationSeconds: number }> {
    const res = await TTSEngine.synthesizeGPU(text, sampleRate);
    return { pcm: res.pcm, durationSeconds: res.durationSeconds };
  }

  /**
   * 5. Diffusion (손): 텍스트 프롬프트로부터 온디바이스 신경망 디퓨전 파이프라인으로 이미지를 그립니다.
   * 기본 백엔드는 엄격히 WebGPU이며, 미가용 시 침묵 CPU 폴백 없이 즉각 Fail-Fast 예외를 분출합니다.
   */
  public async draw(options: GenerationOptions): Promise<DecodedImage> {
    return this.diffusionPipeline.generate(options);
  }

  public async drawGPU(options: GenerationOptions): Promise<DecodedImage> {
    return this.diffusionPipeline.generate({ ...options, backend: 'webgpu' });
  }

  /**
   * 🏛️ WebGPU 네이티브 5대 모달리티 대통합 실행 파이프라인 (Grand Unified All-Modal WebGPU Pipeline)
   * VRAM 내에서 STT -> LLM -> Vision -> Diffusion -> TTS 전 과정을 하드웨어 가속으로 순차 구동합니다.
   */
  public async runGrandMultimodalGPU(config: {
    audioPcm: Float32Array;
    llmTokens: number[];
    llmWeights: LLMWeights;
    visionRgb: Float32Array;
    visionWidth: number;
    visionHeight: number;
    visionWeights: CLIPVisionWeights;
    ttsText: string;
    diffusionOptions: GenerationOptions;
  }): Promise<{
    sttMels: Float32Array;
    llmLogits: Float32Array;
    visionEmbedding: Float32Array;
    ttsPcm: Float32Array;
    diffusionImage: DecodedImage;
  }> {
    const dev = getDevice();
    if (!dev) {
      throw new AMEVAForgeDeviceError('[AllModalOrchestrator] WebGPU device is strictly required for runGrandMultimodalGPU. Refusing silent fallback to CPU.');
    }

    // 1. STT GPU
    const { mels } = await this.listenGPU(config.audioPcm);

    // 2. LLM GPU
    const kvCaches: KVCache[] = config.llmWeights.layers.map(() => ({
      k: new Float32Array(LLMEngine.MAX_SEQ_LEN * LLMEngine.DIM),
      v: new Float32Array(LLMEngine.MAX_SEQ_LEN * LLMEngine.DIM),
      length: 0,
    }));
    let lastLogits = new Float32Array(100);
    for (let pos = 0; pos < config.llmTokens.length; pos++) {
      const logits = await this.thinkGPU(config.llmTokens[pos], pos, config.llmWeights, kvCaches, LLMEngine.DIM, 100);
      lastLogits = new Float32Array(logits);
    }

    // 3. Vision GPU
    const visionEmbedding = await this.seeEmbeddingsGPU(
      config.visionRgb,
      config.visionWidth,
      config.visionHeight,
      config.visionWeights
    );

    // 4. Diffusion GPU
    const diffusionImage = await this.drawGPU(config.diffusionOptions);

    // 5. TTS GPU
    const { pcm: ttsPcm } = await this.speakGPU(config.ttsText);

    return {
      sttMels: mels,
      llmLogits: lastLogits,
      visionEmbedding,
      ttsPcm,
      diffusionImage,
    };
  }
}
