/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: Grand Unified All-Modal Orchestrator E2E Tests
 */

import { AllModalOrchestrator, ALL_MODAL_CAPABILITIES } from '../src/orchestrator/allModalOrchestrator';

describe('AllModalOrchestrator Grand Unified Tests (SCRUM-334)', () => {
  let orchestrator: AllModalOrchestrator;

  beforeEach(() => {
    orchestrator = new AllModalOrchestrator();
  });

  it('manifests full 5-modality capabilities truthfully', () => {
    expect(ALL_MODAL_CAPABILITIES.modalities).toEqual(['stt', 'llm', 'vision', 'tts', 'diffusion']);
    expect(ALL_MODAL_CAPABILITIES.zero_silent_fallback_enforced).toBe(true);
  });

  it('executes STT (listen) converting audio PCM to 80-bin mel spectrogram', () => {
    const pcm = new Float32Array(16000 * 0.2).fill(0.1);
    const { mels, numFrames } = orchestrator.listen(pcm, 16000);
    expect(numFrames).toBeGreaterThan(5);
    expect(mels.length).toBe(80 * numFrames);
  });

  it('executes Vision (seeEdges) detecting image boundaries with Canny 8-direction BFS', () => {
    const rgba = new Uint8ClampedArray(16 * 16 * 4).fill(200);
    const edges = orchestrator.seeEdges(rgba, 16, 16);
    expect(edges.length).toBe(16 * 16);
  });

  it('executes TTS (speak) synthesizing natural PCM audio using formant resonators', () => {
    const { pcm, durationSeconds } = orchestrator.speak('welcome to ameva forge');
    expect(pcm.length).toBeGreaterThan(1000);
    expect(durationSeconds).toBeGreaterThan(0.5);
  });
});
