/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-331 CLIP BPE Tokenizer for WebGPU Text Conditioning
 *
 * WHAT: 텍스트 프롬프트를 Stable Diffusion 표준 77개 정수 토큰 시퀀스(Int32Array[77])로 변환하는 BPE 토크나이저입니다.
 * WHY: 침묵 가짜 프롬프트 무시를 박멸하고, 실제 사용자의 텍스트 입력을 CLIP 임베딩 벡터로 변환하는 첫 관문을 구축하기 위함입니다.
 * HOW: UTF-8 바이트 인코딩 -> 정규식 단어 분할 -> BPE 페어 병합 -> Special Tokens(<|startoftext|>=49406, <|endoftext|>=49407) 삽입 -> 77길이 패딩.
 */
export interface TokenizerOutput {
    tokenIds: Int32Array;
    tokenCount: number;
    words: string[];
}
export declare class CLIPTokenizer {
    static readonly BOS_TOKEN = 49406;
    static readonly EOS_TOKEN = 49407;
    static readonly PAD_TOKEN = 0;
    static readonly MAX_LENGTH = 77;
    private byteEncoder;
    private vocab;
    private bpeRanks;
    constructor(customVocab?: Record<string, number>, customMerges?: string[]);
    private initByteEncoder;
    private initDefaultVocab;
    /**
     * 텍스트 문자열을 77개 길이의 Int32Array 토큰 시퀀스로 인코딩합니다.
     */
    encode(text: string): TokenizerOutput;
    /**
     * 토큰 시퀀스를 읽기 가능한 텍스트로 디코딩합니다.
     */
    decode(tokenIds: Int32Array | number[]): string;
}
