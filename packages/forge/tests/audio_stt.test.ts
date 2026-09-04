/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: STT Modality Unit Tests (Mel-Spectrogram & Whisper Conv)
 */

import { STTEngine, STTError, STTErrorCode } from '../src/audio/sttEngine';

describe('STT Modality Tests (SCRUM-334)', () => {
  it('computes 80-channel log mel-spectrogram from 16kHz PCM waveform', () => {
    const sampleRate = 16000;
    const duration = 0.5; // 0.5 seconds
    const numSamples = Math.floor(sampleRate * duration);
    const pcm = new Float32Array(numSamples);
    // 440Hz sine wave
    for (let i = 0; i < numSamples; i++) {
      pcm[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }

    const { mels, numFrames } = STTEngine.computeLogMelSpectrogram(pcm, sampleRate);
    expect(numFrames).toBeGreaterThan(10);
    expect(mels.length).toBe(80 * numFrames);
    for (let i = 0; i < mels.length; i++) {
      expect(Number.isFinite(mels[i])).toBe(true);
    }
  });

  it('strictly throws STT_INVALID_SAMPLE_RATE when sample rate is not 16000Hz', () => {
    const pcm = new Float32Array(1000).fill(0.1);
    expect(() => {
      STTEngine.computeLogMelSpectrogram(pcm, 44100);
    }).toThrow('16000Hz');
  });

  it('strictly throws STT_NON_FINITE_AUDIO when input contains NaN', () => {
    const pcm = new Float32Array(1000).fill(0.1);
    pcm[50] = NaN;
    expect(() => {
      STTEngine.computeLogMelSpectrogram(pcm, 16000);
    }).toThrow('Non-finite');
  });

  it('executes Whisper-compatible audio convolution downsampling by 4x', () => {
    const numFrames = 32;
    const mels = new Float32Array(80 * numFrames).fill(0.5);
    const dModel = 128;
    const weights = {
      conv1Weight: new Float32Array(dModel * 80 * 3).fill(0.01),
      conv2Weight: new Float32Array(dModel * dModel * 3).fill(0.01),
      positionEmbedding: new Float32Array(1500 * dModel).fill(0.01),
      normGamma: new Float32Array(dModel).fill(1.0),
      normBeta: new Float32Array(dModel).fill(0.0),
    };

    const tokens = STTEngine.forwardAudioEncoder(mels, numFrames, weights, dModel);
    const expectedTokens = Math.floor((Math.floor((numFrames + 1) / 2) + 1) / 2);
    expect(tokens.length).toBe(expectedTokens * dModel);
  });
});
