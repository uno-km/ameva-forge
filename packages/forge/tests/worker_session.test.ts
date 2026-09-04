/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-352 InferenceWorkerHandler & Worker Protocol Tests
 */

import { InferenceWorkerHandler, WorkerInboundMessage, WorkerOutboundMessage } from '../src/worker/inferenceWorker';

describe('SCRUM-349 ~ SCRUM-352: Web Worker Background Runner Protocol Tests', () => {
  function createMockGGUFBuffer(): ArrayBuffer {
    const buf = new ArrayBuffer(1024);
    const view = new DataView(buf);
    view.setUint32(0, 0x46554747, true); // 'GGUF'
    view.setUint32(4, 3, true);          // Version 3
    view.setBigUint64(8, BigInt(0), true); // 0 tensors
    view.setBigUint64(16, BigInt(0), true); // 0 metadata
    return buf;
  }

  it('dispatches load and progress events truthfully over postMessage', async () => {
    const emittedMessages: WorkerOutboundMessage[] = [];
    const handler = new InferenceWorkerHandler((msg) => {
      emittedMessages.push(msg);
    });

    const buffer = createMockGGUFBuffer();
    await handler.handleMessage({
      type: 'LOAD_MODEL',
      id: 'req_1',
      payload: { source: buffer },
    });

    expect(emittedMessages.length).toBeGreaterThanOrEqual(1);
    const doneMsg = emittedMessages.find(m => m.type === 'LOAD_DONE');
    expect(doneMsg).toBeDefined();
    expect(doneMsg?.id).toBe('req_1');
  });

  it('generates text chunks and terminates with GENERATE_DONE event', async () => {
    const emittedMessages: WorkerOutboundMessage[] = [];
    const handler = new InferenceWorkerHandler((msg) => {
      emittedMessages.push(msg);
    });

    const buffer = createMockGGUFBuffer();
    await handler.handleMessage({
      type: 'LOAD_MODEL',
      id: 'load_1',
      payload: { source: buffer },
    });

    await handler.handleMessage({
      type: 'GENERATE',
      id: 'gen_1',
      payload: {
        prompt: 'test prompt',
        maxNewTokens: 2,
        backend: 'cpu',
      },
    });

    const genDoneMsg = emittedMessages.find(m => m.type === 'GENERATE_DONE');
    expect(genDoneMsg).toBeDefined();
    expect(genDoneMsg?.id).toBe('gen_1');
  });

  it('emits typed ERROR message without crashing if unknown message or error occurs', async () => {
    const emittedMessages: WorkerOutboundMessage[] = [];
    const handler = new InferenceWorkerHandler((msg) => {
      emittedMessages.push(msg);
    });

    await handler.handleMessage({
      type: 'INVALID_OP' as any,
      id: 'err_1',
    });

    const errorMsg = emittedMessages.find(m => m.type === 'ERROR');
    expect(errorMsg).toBeDefined();
    expect(errorMsg?.id).toBe('err_1');
  });
});
