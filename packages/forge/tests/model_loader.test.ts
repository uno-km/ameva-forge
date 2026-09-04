/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-348 ModelLoader & GGUF Plug & Play Session Suite Tests
 */

import { ModelLoader, LoadedModelSession, OFFICIAL_MODEL_PRESETS } from '../src/loader/modelLoader';
import { GGUFHeader, GGMLType } from '../src/loader/ggufStreamer';

describe('SCRUM-345 ~ SCRUM-348: ModelLoader Plug & Play Engine Tests', () => {
  function createMockGGUFBuffer(): ArrayBuffer {
    // 최소 유효 GGUF 헤더 바이너리 생성 (Magic: 'GGUF' = 0x46554747, Version: 3)
    const buf = new ArrayBuffer(2048);
    const view = new DataView(buf);

    let offset = 0;
    // Magic: GGUF
    view.setUint32(offset, 0x46554747, true); offset += 4;
    // Version: 3
    view.setUint32(offset, 3, true); offset += 4;
    // Tensor Count: 0 (uint64)
    view.setBigUint64(offset, BigInt(0), true); offset += 8;
    // Metadata KV Count: 2 (uint64)
    view.setBigUint64(offset, BigInt(2), true); offset += 8;

    // KV 1: "general.architecture" = "llama"
    const k1 = 'general.architecture';
    view.setBigUint64(offset, BigInt(k1.length), true); offset += 8;
    for (let i = 0; i < k1.length; i++) view.setUint8(offset + i, k1.charCodeAt(i));
    offset += k1.length;
    // Type: String (8)
    view.setUint32(offset, 8, true); offset += 4;
    const v1 = 'llama';
    view.setBigUint64(offset, BigInt(v1.length), true); offset += 8;
    for (let i = 0; i < v1.length; i++) view.setUint8(offset + i, v1.charCodeAt(i));
    offset += v1.length;

    // KV 2: "llama.embedding_length" = 64 (uint32 type = 4)
    const k2 = 'llama.embedding_length';
    view.setBigUint64(offset, BigInt(k2.length), true); offset += 8;
    for (let i = 0; i < k2.length; i++) view.setUint8(offset + i, k2.charCodeAt(i));
    offset += k2.length;
    view.setUint32(offset, 4, true); offset += 4; // uint32
    view.setUint32(offset, 64, true); offset += 4;

    return buf;
  }

  it('exposes official high-quality micro model presets truthfully', () => {
    expect(OFFICIAL_MODEL_PRESETS.length).toBeGreaterThanOrEqual(3);
    const smollm = OFFICIAL_MODEL_PRESETS.find(p => p.id === 'smollm-135m-q4');
    expect(smollm).toBeDefined();
    expect(smollm?.url).toContain('huggingface.co');
  });

  it('loads valid GGUF ArrayBuffer directly and instantiates LoadedModelSession', async () => {
    const buffer = createMockGGUFBuffer();
    const progressLog: string[] = [];

    const session = await ModelLoader.loadModel(buffer, {
      onProgress: (p) => progressLog.push(p.stage),
    });

    expect(session).toBeInstanceOf(LoadedModelSession);
    expect(session.architecture).toBe('llama');
    expect(session.tokenizer).toBeDefined();
    expect(session.generator).toBeDefined();
    expect(progressLog).toContain('parsing_header');
    expect(progressLog).toContain('loading_tensors');
    expect(progressLog).toContain('ready');
  });

  it('generates text output via session.prompt API seamlessly', async () => {
    const buffer = createMockGGUFBuffer();
    const session = await ModelLoader.loadModel(buffer);

    const emittedChunks: string[] = [];
    const text = await session.prompt('Hello World', {
      maxNewTokens: 3,
      backend: 'cpu',
      onToken: (tok) => {
        if (tok) emittedChunks.push(tok);
      },
    });

    expect(typeof text).toBe('string');
  });

  it('fails fast when fed with invalid or corrupted binary source', async () => {
    const garbageBuffer = new ArrayBuffer(64);
    await expect(ModelLoader.loadModel(garbageBuffer)).rejects.toThrow();
  });
});
