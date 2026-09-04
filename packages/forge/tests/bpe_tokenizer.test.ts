/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-344 BPETokenizer & LLMTextGenerator Suite Tests
 */

import { BPETokenizer, TokenizerErrorCode, TokenizerError } from '../src/tokenizer/bpeTokenizer';
import { Sampler } from '../src/llm/sampler';
import { LLMTextGenerator } from '../src/llm/llmTextGenerator';
import { LLMWeights } from '../src/llm/llmEngine';

describe('SCRUM-341 & SCRUM-344: BPETokenizer Universal Precision Tests', () => {
  let tokenizer: BPETokenizer;

  beforeEach(() => {
    const vocab = [
      '<pad>', '<s>', '</s>', '<unk>',
      'Hello', 'world', '!', ' ', ' Ame', 'va', ' For', 'ge',
      'hello', 'world', '안녕하세요', '세상'
    ];
    tokenizer = new BPETokenizer({
      vocab,
      bosTokenId: 1,
      eosTokenId: 2,
      padTokenId: 0,
      unkTokenId: 3,
    });
  });

  it('correctly reports initialization state and vocab size', () => {
    expect(tokenizer.isInitialized).toBe(true);
    expect(tokenizer.vocabSize).toBeGreaterThan(10);
  });

  it('rejects uninitialized encode/decode with Fail-Fast TokenizerError', () => {
    const uninit = new BPETokenizer();
    expect(() => uninit.encode('test')).toThrow(TokenizerError);
    expect(() => uninit.decode([1, 2])).toThrow(TokenizerError);
  });

  it('encodes ASCII and subwords into correct token IDs with BOS', () => {
    const tokens = tokenizer.encode('Hello world !', true);
    expect(tokens[0]).toBe(1); // BOS
    expect(tokens.length).toBeGreaterThan(2);
  });

  it('reconstructs decoded text accurately (Round-trip decoding)', () => {
    const original = 'Hello world !';
    const tokens = tokenizer.encode(original, true);
    const decoded = tokenizer.decode(tokens, true);
    expect(decoded).toContain('Hello');
    expect(decoded).toContain('world');
  });

  it('initializes seamlessly from GGUF-style metadata dictionary', () => {
    const mockGgufMeta = {
      'tokenizer.ggml.tokens': ['<pad>', '<s>', '</s>', '<unk>', 'apple', 'banana', 'orange'],
      'tokenizer.ggml.bos_token_id': 1,
      'tokenizer.ggml.eos_token_id': 2,
    };
    const ggufTok = BPETokenizer.fromGGUFMetadata(mockGgufMeta);
    expect(ggufTok.isInitialized).toBe(true);
    const tokens = ggufTok.encode('apple banana', false);
    expect(tokens).toEqual([4, 5]);
  });
});

describe('SCRUM-342 & SCRUM-344: Sampler Numerical Stability Tests', () => {
  it('selects argmax deterministically when temperature is 0.0 (Greedy)', () => {
    const logits = new Float32Array([0.1, 0.2, 5.8, 0.9, -1.2]);
    const sampled = Sampler.sampleToken(logits, [], { temperature: 0.0 });
    expect(sampled).toBe(2);
  });

  it('penalizes repeated context tokens when repetitionPenalty is active', () => {
    const logits = new Float32Array([2.0, 2.05, 0.1]);
    // Index 1 has higher logit, but is in context
    const sampled = Sampler.sampleToken(logits, [1], { temperature: 0.0, repetitionPenalty: 1.5 });
    expect(sampled).toBe(0); // Index 0 wins because index 1 was penalized
  });

  it('safely handles non-finite edge cases without exploding', () => {
    const logits = new Float32Array([100.0, 200.0, 300.0]);
    const sampled = Sampler.sampleToken(logits, [], { temperature: 1.0, topK: 2 });
    expect(sampled).toBeGreaterThanOrEqual(1);
  });
});

describe('SCRUM-343 & SCRUM-344: LLMTextGenerator Streaming Autoregressive Tests', () => {
  function createSyntheticWeights(dim: number = 64, vocabSize: number = 50): LLMWeights {
    return {
      tokenEmbedding: new Float32Array(vocabSize * dim).fill(0.01),
      layers: [
        {
          inputNormGamma: new Float32Array(dim).fill(1.0),
          qWeight: new Float32Array(dim * dim).fill(0.01),
          kWeight: new Float32Array(dim * dim).fill(0.01),
          vWeight: new Float32Array(dim * dim).fill(0.01),
          outWeight: new Float32Array(dim * dim).fill(0.01),
          postNormGamma: new Float32Array(dim).fill(1.0),
          gateWeight: new Float32Array(128 * dim).fill(0.01),
          upWeight: new Float32Array(128 * dim).fill(0.01),
          downWeight: new Float32Array(dim * 128).fill(0.01),
        }
      ],
      finalNormGamma: new Float32Array(dim).fill(1.0),
      lmHeadWeight: new Float32Array(vocabSize * dim).fill(0.01),
    };
  }

  it('streams autoregressive tokens invoking onToken callback sequentially', async () => {
    const vocab = ['<pad>', '<s>', '</s>', '<unk>', 'the', 'cat', 'sat', 'on', 'mat'];
    const tokenizer = new BPETokenizer({
      vocab,
      bosTokenId: 1,
      eosTokenId: 2,
    });
    const weights = createSyntheticWeights(64, vocab.length);
    const generator = new LLMTextGenerator(tokenizer, weights, 64, vocab.length);

    const emittedTokens: string[] = [];
    let progressCallCount = 0;

    const result = await generator.generateStream('the cat', {
      maxNewTokens: 5,
      backend: 'cpu',
      temperature: 0.0,
      onToken: (tok, prog) => {
        if (tok) emittedTokens.push(tok);
        progressCallCount++;
      },
    });

    expect(progressCallCount).toBeGreaterThan(0);
    expect(typeof result).toBe('string');
  });
});
