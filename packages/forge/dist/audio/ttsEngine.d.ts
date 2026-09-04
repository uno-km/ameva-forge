/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-334 & SCRUM-335 WebGPU High-Precision DSP Formant Speech Synthesizer (TTS)
 *
 * WHAT: 텍스트 및 음소로부터 Rosenberg 성문 펄스와 5-밴드 바이쿼드 공진기를
 *      WebGPU WGSL 컴퓨트 셰이더 및 VRAM에서 직접 계산하는 온디바이스 음성 합성 엔진입니다.
 * WHY: 침묵 CPU 폴백 없이 브라우저 GPU 하드웨어를 100% 활용하여 초고속 실시간 발화를 실행하기 위함입니다.
 * HOW: Rosenberg Glottal Flow Model -> WebGPU TTS_SYNTH_WGSL -> PCM Waveform.
 */
export declare enum TTSErrorCode {
    TTS_TEXT_EMPTY = "TTS_TEXT_EMPTY",
    TTS_INVALID_SAMPLE_RATE = "TTS_INVALID_SAMPLE_RATE",
    TTS_NON_FINITE_AUDIO = "TTS_NON_FINITE_AUDIO",
    WEBGPU_NOT_AVAILABLE = "WEBGPU_NOT_AVAILABLE"
}
export declare class TTSError extends Error {
    readonly code: TTSErrorCode;
    constructor(code: TTSErrorCode, message: string);
}
export interface FormantPreset {
    f1: number;
    f2: number;
    f3: number;
    f4: number;
    f5: number;
    bw1: number;
    bw2: number;
}
export declare class TTSEngine {
    static readonly DEFAULT_SAMPLE_RATE = 22050;
    private static readonly VOWEL_FORMANTS;
    /**
     * 텍스트 문자열을 실제 음성 파형(Float32Array PCM)으로 합성합니다 (CPU Reference).
     */
    static synthesize(text: string, sampleRate?: number, f0?: number): {
        pcm: Float32Array;
        sampleRate: number;
        durationSeconds: number;
    };
    /**
     * WebGPU WGSL 셰이더를 사용한 하드웨어 가속 음성 합성 (Zero CPU Fallback)
     */
    static synthesizeGPU(text: string, sampleRate?: number, f0?: number): Promise<{
        pcm: Float32Array;
        sampleRate: number;
        durationSeconds: number;
    }>;
    private static calculateResonator;
}
