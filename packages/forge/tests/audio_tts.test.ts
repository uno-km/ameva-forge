/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: TTS Modality Unit Tests (Formant DSP Synthesis)
 */

import { TTSEngine, TTSError, TTSErrorCode } from '../src/audio/ttsEngine';

describe('TTS Modality Tests (SCRUM-334)', () => {
  it('synthesizes real PCM audio waveform from text using Rosenberg Glottal Flow model', () => {
    const text = 'hello world';
    const { pcm, sampleRate, durationSeconds } = TTSEngine.synthesize(text, 22050, 140.0);

    expect(pcm.length).toBeGreaterThan(1000);
    expect(sampleRate).toBe(22050);
    expect(durationSeconds).toBeGreaterThan(0.5);

    // Verify waveform energy and finiteness
    let energy = 0.0;
    for (let i = 0; i < pcm.length; i++) {
      expect(Number.isFinite(pcm[i])).toBe(true);
      expect(Math.abs(pcm[i])).toBeLessThanOrEqual(1.0);
      energy += pcm[i] * pcm[i];
    }
    expect(energy).toBeGreaterThan(0.1);
  });

  it('strictly throws TTS_TEXT_EMPTY when given empty or whitespace text', () => {
    expect(() => {
      TTSEngine.synthesize('   ');
    }).toThrow('empty text');
  });

  it('strictly throws TTS_INVALID_SAMPLE_RATE when sample rate is outside [8000, 48000]', () => {
    expect(() => {
      TTSEngine.synthesize('test', 5000);
    }).toThrow('Sample rate');
  });
});
