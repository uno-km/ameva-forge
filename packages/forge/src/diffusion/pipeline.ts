/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-330 Real In-Browser WebGPU Diffusion Pipeline Orchestrator
 *
 * WHAT: CLIP 텍스트 인코더, UNet 신경망 실행 그래프, 오일러 스케줄러, VAE 디코더를 비동기로 조율하는 완제품 오케스트레이터입니다.
 * WHY: 가짜 decay 수식이나 가짜 가중치 침묵 생성을 원천 박멸하고,
 *      실제 신경망 순전파와 페일패스트(Fail-Fast) 오류 검증을 100% 집행하기 위해 존재합니다.
 * HOW: Tokenizer -> CLIPTextEncoder -> Multi-step UNetGraph -> EulerDiscreteScheduler -> VAEDecoder.
 */

import { GGUFStreamer, GGUFHeader } from '../loader/ggufStreamer';
import { EulerDiscreteScheduler } from './scheduler';
import { VAEDecoder, DecodedImage, VAEDecoderWeights } from './vaeDecoder';
import { CLIPTokenizer } from './clipTokenizer';
import { CLIPTextEncoder, CLIPTextEncoderWeights } from './clipTextEncoder';
import { UNetGraph, UNetWeights } from './unetGraph';

export enum DiffusionPipelineErrorCode {
  UNET_FORWARD_NOT_IMPLEMENTED = 'UNET_FORWARD_NOT_IMPLEMENTED',
  CLIP_ENCODER_NOT_IMPLEMENTED = 'CLIP_ENCODER_NOT_IMPLEMENTED',
  VAE_WEIGHTS_REQUIRED = 'VAE_WEIGHTS_REQUIRED',
  MODEL_NOT_LOADED = 'MODEL_NOT_LOADED',
}

export class DiffusionPipelineError extends Error {
  public readonly code: DiffusionPipelineErrorCode;

  constructor(code: DiffusionPipelineErrorCode, message: string, options?: { cause?: unknown }) {
    super(`[WebGPUDiffusionPipeline:${code}] ${message}`, options);
    this.name = 'DiffusionPipelineError';
    this.code = code;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export interface GenerationOptions {
  prompt: string;
  negativePrompt?: string;
  numSteps?: number;
  width?: number;
  height?: number;
  seed?: number;
  guidanceScale?: number;
  vaeWeights?: VAEDecoderWeights;
  unetWeights?: UNetWeights;
  clipWeights?: CLIPTextEncoderWeights;
  onProgress?: (progress: GenerationProgress) => void;
}

export interface GenerationProgress {
  step: number;
  totalSteps: number;
  percentage: number;
  elapsedMs: number;
}

export class WebGPUDiffusionPipeline {
  public modelHeader?: GGUFHeader;
  public scheduler: EulerDiscreteScheduler;
  public tokenizer: CLIPTokenizer;
  public isModelLoaded: boolean = false;

  constructor() {
    this.scheduler = new EulerDiscreteScheduler(1);
    this.tokenizer = new CLIPTokenizer();
  }

  /**
   * GGUF 모델 헤더를 로드하고 가중치 오프셋 테이블을 구축합니다.
   */
  public async loadModel(headerBuffer: ArrayBuffer): Promise<GGUFHeader> {
    this.modelHeader = GGUFStreamer.parseHeader(headerBuffer);
    this.isModelLoaded = true;
    return this.modelHeader;
  }

  /**
   * 텍스트 프롬프트로부터 이미지를 생성하는 완전한 순전파 파이프라인.
   * 가중치 누락이나 결함 시 침묵 가짜 시뮬레이션 없이 즉시 Fail-Fast 예외를 분출합니다.
   */
  public async generate(options: GenerationOptions): Promise<DecodedImage> {
    const startTime = performance.now();
    const {
      prompt,
      numSteps = 1,
      width = 64,
      height = 64,
      seed = 42,
      onProgress,
    } = options;

    // 1. 엄격한 사전 가중치 유효성 검사 (Zero Silent Fallback)
    if (!options.vaeWeights) {
      throw new DiffusionPipelineError(
        DiffusionPipelineErrorCode.VAE_WEIGHTS_REQUIRED,
        'vaeWeights are strictly required to decode latent to RGB.'
      );
    }
    if (!options.unetWeights) {
      throw new DiffusionPipelineError(
        DiffusionPipelineErrorCode.UNET_FORWARD_NOT_IMPLEMENTED,
        'UNet weights are strictly required. Refusing to simulate denoising with heuristic decay formulas. Real UNet execution graph required.'
      );
    }
    if (!options.clipWeights) {
      throw new DiffusionPipelineError(
        DiffusionPipelineErrorCode.CLIP_ENCODER_NOT_IMPLEMENTED,
        'CLIP weights are strictly required for text conditioning. Refusing to silently ignore text prompt.'
      );
    }

    const latentH = Math.floor(height / 8);
    const latentW = Math.floor(width / 8);
    const latentChannels = 4;

    // 2. CLIP BPE 토큰화 및 텍스트 인코딩
    const { tokenIds } = this.tokenizer.encode(prompt);
    const textContext = CLIPTextEncoder.forward(tokenIds, options.clipWeights);

    // 3. 디노이징 스케줄러 타임스텝 설정 및 초기 가우시안 잠재 노이즈 생성
    this.scheduler.setTimesteps(numSteps);
    let latents = this.scheduler.generateInitialNoise(latentChannels, latentH, latentW, seed);

    // 4. Multi-Step Denoising Loop (Yielding to prevent browser TDR)
    for (let step = 0; step < numSteps; step++) {
      const t = this.scheduler.timesteps[step];

      // Real UNet Multi-block execution graph
      const predNoise = UNetGraph.forward(
        latents,
        t,
        textContext,
        options.unetWeights,
        latentH,
        latentW,
        32
      );

      // Scheduler Euler Step update
      const { prevSample } = this.scheduler.step(predNoise, step, latents);
      latents = prevSample;

      // 브라우저 렌더 이벤트 루프에 제어권 양보 (TDR 크래시 원천 차단)
      await this.scheduler.yieldToMainThread();

      if (onProgress) {
        const elapsedMs = performance.now() - startTime;
        onProgress({
          step: step + 1,
          totalSteps: numSteps,
          percentage: Math.round(((step + 1) / numSteps) * 100),
          elapsedMs,
        });
      }
    }

    // 5. VAE Decoder: Latent -> RGB Canvas ImageData 변환
    const decoded = VAEDecoder.decodeLatentToRGB(latents, latentW, latentH, width, height, options.vaeWeights);

    return decoded;
  }
}
