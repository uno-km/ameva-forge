/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: High-Precision On-Device STT Engine (Whisper-Compatible)
 *
 * WHAT: 16kHz 마이크/오디오 PCM 파형을 80채널 로그 멜-스펙트로그램으로 변환하고,
 *      오디오 컨볼루션 및 트랜스포머 인코더를 거쳐 텍스트를 받아 적는 음성 인식(STT) 엔진입니다.
 * WHY: 외부 클라우드 통신 없는 제로 레이턴시 온디바이스 음성 명령 및 음성 인식을 지원하기 위함입니다.
 * HOW: Hanning Window + 80-bin Mel Filterbank -> 2-Stage Conv1D Downsampling (4x) -> Transformer Encoder -> Decoder.
 */
export declare enum STTErrorCode {
    STT_INVALID_SAMPLE_RATE = "STT_INVALID_SAMPLE_RATE",
    STT_NON_FINITE_AUDIO = "STT_NON_FINITE_AUDIO",
    STT_AUDIO_TOO_SHORT = "STT_AUDIO_TOO_SHORT",
    STT_WEIGHTS_REQUIRED = "STT_WEIGHTS_REQUIRED"
}
export declare class STTError extends Error {
    readonly code: STTErrorCode;
    constructor(code: STTErrorCode, message: string);
}
export interface STTEncoderWeights {
    conv1Weight: Float32Array;
    conv1Bias?: Float32Array;
    conv2Weight: Float32Array;
    conv2Bias?: Float32Array;
    positionEmbedding: Float32Array;
    normGamma: Float32Array;
    normBeta: Float32Array;
}
export declare class STTEngine {
    static readonly SAMPLE_RATE = 16000;
    static readonly N_MELS = 80;
    static readonly N_FFT = 400;
    static readonly HOP_LENGTH = 160;
    /**
     * 16kHz PCM Float32Array로부터 80채널 로그 멜-스펙트로그램(Log Mel-Spectrogram)을 계산합니다.
     */
    static computeLogMelSpectrogram(pcm: Float32Array, sampleRate?: number): {
        mels: Float32Array;
        numFrames: number;
    };
    /**
     * 오디오 멜-스펙트로그램 -> Whisper 컨볼루션 인코더 순전파 (시간 축 4배 압축)
     */
    static forwardAudioEncoder(mels: Float32Array, numFrames: number, weights: STTEncoderWeights, dModel?: number): Float32Array;
}
