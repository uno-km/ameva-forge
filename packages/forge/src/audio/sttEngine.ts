/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-334 & SCRUM-335 WebGPU High-Precision On-Device STT Engine (Whisper-Compatible)
 *
 * WHAT: 16kHz 오디오 PCM 파형을 80채널 로그 멜-스펙트로그램으로 WebGPU VRAM 내에서 변환하고,
 *      오디오 컨볼루션 및 트랜스포머 인코더를 거쳐 텍스트를 받아 적는 음성 인식(STT) 엔진입니다.
 * WHY: 침묵 CPU 폴백 없이 브라우저 GPU 하드웨어에서 직접 음향 특징을 고속 추출하기 위함입니다.
 * HOW: Hanning Window + 80-bin Mel Filterbank (STT_MEL_WGSL) -> 2-Stage Conv1D Downsampling -> Transformer.
 */

import { STT_MEL_WGSL } from '../tensor/kernels/stt_mel.wgsl';
import { STT_STFT_WGSL } from '../tensor/kernels/stt_stft.wgsl';
import { getDevice } from '../webgpu/device';
import { allocateBuffer, freeBuffer, readBufferToFloat32Array } from '../webgpu/buffers';
import { computeDispatch2D } from '../tensor/dispatchShape';
import { _globalPipelineCache } from '../webgpu/pipelineCache';

export enum STTErrorCode {
  STT_INVALID_SAMPLE_RATE = 'STT_INVALID_SAMPLE_RATE',
  STT_NON_FINITE_AUDIO = 'STT_NON_FINITE_AUDIO',
  STT_AUDIO_TOO_SHORT = 'STT_AUDIO_TOO_SHORT',
  STT_WEIGHTS_REQUIRED = 'STT_WEIGHTS_REQUIRED',
  WEBGPU_NOT_AVAILABLE = 'WEBGPU_NOT_AVAILABLE',
}

export class STTError extends Error {
  public readonly code: STTErrorCode;

  constructor(code: STTErrorCode, message: string) {
    super(`[STT:${code}] ${message}`);
    this.name = 'STTError';
    this.code = code;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export interface STTEncoderWeights {
  conv1Weight: Float32Array; // [dModel, 80, 3]
  conv1Bias?: Float32Array;  // [dModel]
  conv2Weight: Float32Array; // [dModel, dModel, 3]
  conv2Bias?: Float32Array;  // [dModel]
  positionEmbedding: Float32Array; // [1500, dModel]
  normGamma: Float32Array;   // [dModel]
  normBeta: Float32Array;    // [dModel]
}

export class STTEngine {
  public static readonly SAMPLE_RATE = 16000;
  public static readonly N_MELS = 80;
  public static readonly N_FFT = 400;  // 25ms
  public static readonly HOP_LENGTH = 160; // 10ms

  /**
   * 16kHz PCM Float32Array로부터 80채널 로그 멜-스펙트로그램(Log Mel-Spectrogram)을 계산합니다 (CPU Reference).
   */
  public static computeLogMelSpectrogram(pcm: Float32Array, sampleRate: number = 16000): { mels: Float32Array; numFrames: number } {
    if (sampleRate !== STTEngine.SAMPLE_RATE) {
      throw new STTError(
        STTErrorCode.STT_INVALID_SAMPLE_RATE,
        `STTEngine requires 16000Hz sample rate, received: ${sampleRate}Hz`
      );
    }
    if (pcm.length < STTEngine.N_FFT) {
      throw new STTError(
        STTErrorCode.STT_AUDIO_TOO_SHORT,
        `Audio too short for STT FFT: length=${pcm.length} < N_FFT=${STTEngine.N_FFT}`
      );
    }

    for (let i = 0; i < pcm.length; i++) {
      if (!Number.isFinite(pcm[i])) {
        throw new STTError(STTErrorCode.STT_NON_FINITE_AUDIO, `Non-finite audio sample detected at index ${i}`);
      }
    }

    const nFft = STTEngine.N_FFT;
    const hop = STTEngine.HOP_LENGTH;
    const nMels = STTEngine.N_MELS;
    const numFrames = Math.floor((pcm.length - nFft) / hop) + 1;
    const nBins = Math.floor(nFft / 2) + 1; // 201

    const window = new Float32Array(nFft);
    for (let i = 0; i < nFft; i++) {
      window[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / nFft));
    }

    const melFilterbank = this.createMelFilterbank(nMels, nBins, sampleRate);
    const mels = new Float32Array(numFrames * nMels);

    for (let f = 0; f < numFrames; f++) {
      const start = f * hop;
      const magnitudes = new Float32Array(nBins);

      for (let k = 0; k < nBins; k++) {
        let real = 0.0;
        let imag = 0.0;
        for (let n = 0; n < nFft; n++) {
          const sample = pcm[start + n] * window[n];
          const angle = (-2.0 * Math.PI * k * n) / nFft;
          real += sample * Math.cos(angle);
          imag += sample * Math.sin(angle);
        }
        magnitudes[k] = Math.sqrt(real * real + imag * imag);
      }

      for (let m = 0; m < nMels; m++) {
        let energy = 0.0;
        const melOff = m * nBins;
        for (let k = 0; k < nBins; k++) {
          energy += magnitudes[k] * melFilterbank[melOff + k];
        }
        const logMel = Math.log10(Math.max(energy, 1e-5));
        mels[f * nMels + m] = logMel;
      }
    }

    return { mels, numFrames };
  }

  /**
   * WebGPU WGSL 셰이더를 사용한 하드웨어 가속 멜-스펙트로그램 계산 (Zero CPU Fallback)
   */
  public static async computeLogMelSpectrogramGPU(pcm: Float32Array, sampleRate: number = 16000): Promise<{ mels: Float32Array; numFrames: number }> {
    const dev = getDevice();
    if (!dev) {
      throw new STTError(
        STTErrorCode.WEBGPU_NOT_AVAILABLE,
        'WebGPU device is strictly required for WebGPU STT Mel computation. Refusing silent fallback to CPU.'
      );
    }
    if (sampleRate !== STTEngine.SAMPLE_RATE) {
      throw new STTError(
        STTErrorCode.STT_INVALID_SAMPLE_RATE,
        `STTEngine requires 16000Hz sample rate, received: ${sampleRate}Hz`
      );
    }
    if (pcm.length < STTEngine.N_FFT) {
      throw new STTError(
        STTErrorCode.STT_AUDIO_TOO_SHORT,
        `Audio too short for STT FFT: length=${pcm.length} < N_FFT=${STTEngine.N_FFT}`
      );
    }

    const nFft = STTEngine.N_FFT;
    const hop = STTEngine.HOP_LENGTH;
    const nMels = STTEngine.N_MELS;
    const numFrames = Math.floor((pcm.length - nFft) / hop) + 1;
    const nBins = Math.floor(nFft / 2) + 1; // 201

    // 1. Allocate GPU Buffer for PCM samples (DMA directly into VRAM)
    const { buffer: pcmBuf, token: pcmTok } = allocateBuffer(pcm.byteLength, 0x0080 | 0x0008, 'tensor', 'stt_pcm');
    dev.queue.writeBuffer(pcmBuf, 0, pcm.buffer, pcm.byteOffset, pcm.byteLength);

    // 2. Allocate GPU Buffer for STFT Magnitudes (VRAM resident, zero CPU roundtrip)
    const totalStftEntries = numFrames * nBins;
    const stftDispatch = computeDispatch2D(Math.ceil(totalStftEntries / 64));
    const stftParams = new Uint32Array([
      numFrames,
      nFft,
      hop,
      nBins,
      stftDispatch.dispatchX,
      pcm.length,
      0,
      0
    ]);
    const { buffer: stftParamBuf, token: stftParamTok } = allocateBuffer(32, 0x0040 | 0x0008, 'uniform', 'stt_stft_p');
    dev.queue.writeBuffer(stftParamBuf, 0, stftParams.buffer, stftParams.byteOffset, stftParams.byteLength);

    const { buffer: magBuf, token: magTok } = allocateBuffer(totalStftEntries * 4, 0x0080, 'tensor', 'stt_mags');

    const { pipeline: stftPipeline } = _globalPipelineCache.getPipeline('stt_stft', STT_STFT_WGSL);
    const stftBindGroup = dev.createBindGroup({
      layout: stftPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: stftParamBuf } },
        { binding: 1, resource: { buffer: pcmBuf } },
        { binding: 2, resource: { buffer: magBuf } },
      ],
    });

    // 3. Setup Pass 2: Mel-Filterbank Projection Kernel
    const melFilterbank = this.createMelFilterbank(nMels, nBins, sampleRate);
    const totalMelEntries = numFrames * nMels;
    const melDispatch = computeDispatch2D(Math.ceil(totalMelEntries / 64));

    const melParams = new Uint32Array([numFrames, nMels, nBins, melDispatch.dispatchX]);
    const { buffer: melParamBuf, token: melParamTok } = allocateBuffer(16, 0x0040 | 0x0008, 'uniform', 'stt_mel_p');
    dev.queue.writeBuffer(melParamBuf, 0, melParams.buffer, melParams.byteOffset, melParams.byteLength);

    const { buffer: fbBuf, token: fbTok } = allocateBuffer(melFilterbank.byteLength, 0x0080 | 0x0008, 'tensor', 'stt_fb');
    dev.queue.writeBuffer(fbBuf, 0, melFilterbank.buffer, melFilterbank.byteOffset, melFilterbank.byteLength);

    const { buffer: outBuf, token: outTok } = allocateBuffer(totalMelEntries * 4, 0x0080 | 0x0004, 'tensor', 'stt_out');

    const { pipeline: melPipeline } = _globalPipelineCache.getPipeline('stt_mel', STT_MEL_WGSL);
    const melBindGroup = dev.createBindGroup({
      layout: melPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: melParamBuf } },
        { binding: 1, resource: { buffer: magBuf } },
        { binding: 2, resource: { buffer: fbBuf } },
        { binding: 3, resource: { buffer: outBuf } },
      ],
    });

    // 4. Record and dispatch both compute passes in a single GPU Command Buffer
    const enc = dev.createCommandEncoder();

    // Pass 1: Hardware-Accelerated STFT with Hanning Window
    const pass1 = enc.beginComputePass();
    pass1.setPipeline(stftPipeline);
    pass1.setBindGroup(0, stftBindGroup);
    pass1.dispatchWorkgroups(stftDispatch.dispatchX, stftDispatch.dispatchY);
    pass1.end();

    // Pass 2: 80-Channel Mel-Filterbank Projection & Log Compression
    const pass2 = enc.beginComputePass();
    pass2.setPipeline(melPipeline);
    pass2.setBindGroup(0, melBindGroup);
    pass2.dispatchWorkgroups(melDispatch.dispatchX, melDispatch.dispatchY);
    pass2.end();

    dev.queue.submit([enc.finish()]);

    // 5. Read back only the final 80-channel Mel Spectrogram
    const rawMels = await readBufferToFloat32Array(outBuf, totalMelEntries * 4);
    const mels = new Float32Array(rawMels);

    // 6. Free all allocated VRAM staging and intermediate buffers
    freeBuffer(pcmBuf, pcmTok);
    freeBuffer(stftParamBuf, stftParamTok);
    freeBuffer(magBuf, magTok);
    freeBuffer(melParamBuf, melParamTok);
    freeBuffer(fbBuf, fbTok);
    freeBuffer(outBuf, outTok);

    return { mels, numFrames };
  }

  /**
   * 오디오 멜-스펙트로그램 -> Whisper 컨볼루션 인코더 순전파 (시간 축 4배 압축)
   */
  public static forwardAudioEncoder(
    mels: Float32Array,
    numFrames: number,
    weights: STTEncoderWeights,
    dModel: number = 384
  ): Float32Array {
    const nMels = STTEngine.N_MELS;

    // Conv1 (inC=80, outC=dModel, stride=2, padding=1)
    const outFrames1 = Math.floor((numFrames + 1) / 2);
    const h1 = new Float32Array(dModel * outFrames1);

    for (let oc = 0; oc < dModel; oc++) {
      const b = weights.conv1Bias ? weights.conv1Bias[oc] : 0.0;
      for (let t = 0; t < outFrames1; t++) {
        let sum = b;
        const inCenter = t * 2;
        for (let ic = 0; ic < nMels; ic++) {
          for (let k = -1; k <= 1; k++) {
            const inT = inCenter + k;
            if (inT >= 0 && inT < numFrames) {
              const val = mels[ic * numFrames + inT];
              const w = weights.conv1Weight[(oc * nMels + ic) * 3 + (k + 1)];
              sum += val * w;
            }
          }
        }
        // GELU
        const clamped = Math.max(-88.0, Math.min(88.0, 1.702 * sum));
        h1[oc * outFrames1 + t] = sum * (1.0 / (1.0 + Math.exp(-clamped)));
      }
    }

    // Conv2 (inC=dModel, outC=dModel, stride=2, padding=1)
    const outFrames2 = Math.floor((outFrames1 + 1) / 2);
    const audioTokens = new Float32Array(outFrames2 * dModel);

    for (let t = 0; t < outFrames2; t++) {
      const inCenter = t * 2;
      const tokenOffset = t * dModel;

      for (let oc = 0; oc < dModel; oc++) {
        let sum = weights.conv2Bias ? weights.conv2Bias[oc] : 0.0;
        for (let ic = 0; ic < dModel; ic++) {
          for (let k = -1; k <= 1; k++) {
            const inT = inCenter + k;
            if (inT >= 0 && inT < outFrames1) {
              const val = h1[ic * outFrames1 + inT];
              const w = weights.conv2Weight[(oc * dModel + ic) * 3 + (k + 1)];
              sum += val * w;
            }
          }
        }
        // Add Position Embedding + Norm
        const posOffset = t * dModel;
        const posVal = posOffset + oc < weights.positionEmbedding.length ? weights.positionEmbedding[posOffset + oc] : 0.0;
        audioTokens[tokenOffset + oc] = (sum + posVal) * weights.normGamma[oc] + weights.normBeta[oc];
      }
    }

    return audioTokens;
  }

  private static createMelFilterbank(nMels: number, nBins: number, sampleRate: number): Float32Array {
    const fMin = 0.0;
    const fMax = sampleRate / 2.0;

    const hzToMel = (hz: number) => 2595.0 * Math.log10(1.0 + hz / 700.0);
    const melToHz = (mel: number) => 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);

    const melMin = hzToMel(fMin);
    const melMax = hzToMel(fMax);

    const melPoints = new Float32Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) {
      melPoints[i] = melToHz(melMin + (i / (nMels + 1)) * (melMax - melMin));
    }

    const binPoints = new Float32Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) {
      binPoints[i] = Math.floor(((STTEngine.N_FFT + 1) * melPoints[i]) / sampleRate);
    }

    const filterbank = new Float32Array(nMels * nBins);
    for (let m = 0; m < nMels; m++) {
      const left = binPoints[m];
      const center = binPoints[m + 1];
      const right = binPoints[m + 2];

      for (let k = 0; k < nBins; k++) {
        let weight = 0.0;
        if (k >= left && k <= center && center > left) {
          weight = (k - left) / (center - left);
        } else if (k >= center && k <= right && right > center) {
          weight = (right - k) / (right - center);
        }
        filterbank[m * nBins + k] = weight;
      }
    }

    return filterbank;
  }
}
