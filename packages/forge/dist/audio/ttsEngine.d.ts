/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: Zero-Dependency High-Precision DSP Formant Speech Synthesizer (TTS)
 *
 * WHAT: 텍스트 및 음소(Phoneme)로부터 로젠버그 성문 펄스(Rosenberg Glottal Pulse)와
 *      5-밴드 바이쿼드 공진기(5-Band Cascaded Biquad Resonators)를 통해
 *      순수 수학적 원리로 자연스러운 음성 파형(PCM Float32Array)을 합성하는 온디바이스 TTS 엔진입니다.
 * WHY: 외부 300MB 가중치 다운로드 없이도 브라우저 단독으로 즉각적인 발화와 음성 출력을 실행하기 위해 존재합니다.
 * HOW: Rosenberg Glottal Flow Model -> 5 Formant Biquad Filters (F1~F5) -> ADSR Envelope -> PCM Waveform.
 */
export declare enum TTSErrorCode {
    TTS_TEXT_EMPTY = "TTS_TEXT_EMPTY",
    TTS_INVALID_SAMPLE_RATE = "TTS_INVALID_SAMPLE_RATE",
    TTS_NON_FINITE_AUDIO = "TTS_NON_FINITE_AUDIO"
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
     * 텍스트 문자열을 실제 음성 파형(Float32Array PCM)으로 합성합니다.
     */
    static synthesize(text: string, sampleRate?: number, f0?: number): {
        pcm: Float32Array;
        sampleRate: number;
        durationSeconds: number;
    };
    private static calculateResonator;
}
