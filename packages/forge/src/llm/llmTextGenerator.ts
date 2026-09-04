/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-343 Autoregressive LLM Text Streaming Generator
 *
 * WHAT: BPETokenizer, LLMEngine, Sampler를 하나로 유기적으로 결합하여,
 *      자연어 프롬프트로부터 실시간 스트리밍 텍스트 토큰을 생성하고 방출하는 완제품 생성기입니다.
 * WHY: 메인 스레드 락(UI 프리징)과 TDR 크래시를 방지하고, WebGPU 하드웨어 가속을 통해
 *      사용자에게 ChatGPT와 동등한 타자기 효과의 실시간 텍스트 스트리밍 경험을 제공하기 위함입니다.
 * HOW: Tokenize -> Prefill KV-Cache -> Autoregressive Decode Loop (with Event-Loop Yield) -> Sampler -> Decode Chunk.
 */

import { BPETokenizer } from '../tokenizer/bpeTokenizer';
import { LLMEngine, LLMWeights, KVCache } from './llmEngine';
import { Sampler, SamplingOptions } from './sampler';
import { getDevice } from '../webgpu/device';

export interface TextGenerationOptions extends SamplingOptions {
  maxNewTokens?: number;
  stopTokens?: number[];
  onToken?: (token: string, progress: TextGenerationProgress) => void;
  abortSignal?: AbortSignal;
  backend?: 'webgpu' | 'cpu';
}

export interface TextGenerationProgress {
  tokenCount: number;
  maxTokens: number;
  tps: number; // Tokens Per Second
  elapsedMs: number;
  done: boolean;
}

export class LLMTextGenerator {
  public tokenizer: BPETokenizer;
  public weights?: LLMWeights;
  public dim: number;
  public vocabSize: number;

  constructor(tokenizer: BPETokenizer, weights?: LLMWeights, dim: number = LLMEngine.DIM, vocabSize: number = 32000) {
    this.tokenizer = tokenizer;
    this.weights = weights;
    this.dim = dim;
    this.vocabSize = vocabSize;
  }

  /**
   * 브라우저 렌더 이벤트 루프에 제어권을 양보하여 UI 멈춤 및 TDR을 방지합니다.
   */
  private static async yieldToEventLoop(): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, 0));
  }

  /**
   * 프롬프트 문자열로부터 텍스트를 실시간 스트리밍으로 생성합니다.
   */
  public async generateStream(
    prompt: string,
    options: TextGenerationOptions = {}
  ): Promise<string> {
    if (!this.weights) {
      throw new Error('[LLMTextGenerator] Model weights must be loaded before generating text.');
    }
    if (!this.tokenizer.isInitialized) {
      throw new Error('[LLMTextGenerator] Tokenizer must be initialized before generating text.');
    }

    const {
      maxNewTokens = 128,
      stopTokens = [this.tokenizer.eosTokenId],
      onToken,
      abortSignal,
      backend = 'webgpu',
    } = options;

    const useGPU = backend === 'webgpu' && !!getDevice();

    // 1. 프롬프트 인코딩
    const promptTokens = this.tokenizer.encode(prompt, true);
    if (promptTokens.length === 0) {
      return '';
    }

    // 2. KV-Cache 초기화
    const kvCaches: KVCache[] = this.weights.layers.map(() => ({
      k: new Float32Array(LLMEngine.MAX_SEQ_LEN * this.dim),
      v: new Float32Array(LLMEngine.MAX_SEQ_LEN * this.dim),
      length: 0,
    }));

    // 3. Prefill 단계: 프롬프트 토큰들을 순차적으로 KV-Cache에 주입
    let lastLogits: Float32Array<ArrayBufferLike> = new Float32Array(this.vocabSize);
    for (let pos = 0; pos < promptTokens.length; pos++) {
      if (useGPU) {
        lastLogits = await LLMEngine.forwardTokenGPU(
          promptTokens[pos],
          pos,
          this.weights,
          kvCaches,
          this.dim,
          this.vocabSize
        );
      } else {
        lastLogits = LLMEngine.forwardToken(
          promptTokens[pos],
          pos,
          this.weights,
          kvCaches,
          this.dim,
          this.vocabSize
        );
      }
    }

    // 4. Autoregressive Decode 루프
    const generatedTokens: number[] = [];
    const allContextTokens = [...promptTokens];
    let generatedText = '';
    const startTime = performance.now();

    for (let step = 0; step < maxNewTokens; step++) {
      if (abortSignal?.aborted) {
        break;
      }

      // 샘플러를 통해 다음 토큰 결정
      const nextTokenId = Sampler.sampleToken(lastLogits as Float32Array, allContextTokens, options);

      // 종료 토큰 검사
      if (stopTokens.includes(nextTokenId)) {
        break;
      }

      generatedTokens.push(nextTokenId);
      allContextTokens.push(nextTokenId);

      // 토큰 텍스트 디코딩
      const piece = this.tokenizer.decode([nextTokenId], true);
      generatedText += piece;

      const elapsedMs = performance.now() - startTime;
      const tps = generatedTokens.length / (Math.max(1, elapsedMs) / 1000.0);

      if (onToken) {
        onToken(piece, {
          tokenCount: generatedTokens.length,
          maxTokens: maxNewTokens,
          tps: Math.round(tps * 10) / 10,
          elapsedMs: Math.round(elapsedMs),
          done: false,
        });
      }

      const nextPos = promptTokens.length + step;
      if (nextPos >= LLMEngine.MAX_SEQ_LEN - 1) {
        // 최대 컨텍스트 초과 시 안전 중단
        break;
      }

      // 다음 토큰 순전파
      if (useGPU) {
        lastLogits = await LLMEngine.forwardTokenGPU(
          nextTokenId,
          nextPos,
          this.weights,
          kvCaches,
          this.dim,
          this.vocabSize
        );
      } else {
        lastLogits = LLMEngine.forwardToken(
          nextTokenId,
          nextPos,
          this.weights,
          kvCaches,
          this.dim,
          this.vocabSize
        );
      }

      // 브라우저 렌더러에 제어권 양보 (UI 프리징 방지)
      if (step % 2 === 0) {
        await LLMTextGenerator.yieldToEventLoop();
      }
    }

    if (onToken) {
      const elapsedMs = performance.now() - startTime;
      const tps = generatedTokens.length / (Math.max(1, elapsedMs) / 1000.0);
      onToken('', {
        tokenCount: generatedTokens.length,
        maxTokens: maxNewTokens,
        tps: Math.round(tps * 10) / 10,
        elapsedMs: Math.round(elapsedMs),
        done: true,
      });
    }

    return generatedText;
  }
}
